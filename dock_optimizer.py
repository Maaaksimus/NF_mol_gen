import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

sys.path.insert(0,'..')
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from copy import deepcopy
from dataloader import PretrainDataset
from models.MolHF import MolHF
from torch.utils.data import DataLoader
from multiprocessing import Pool
from time import time, ctime
import optimize_property as op
from envs import environment as env
from envs.timereport import TimeReport
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

# РџСЂРµРґРѕР±СѓС‡РµРЅРЅР°СЏ РјРѕРґРµР»СЊ РґР»СЏ РѕРїС‚РёРјРёР·Р°С†РёРё

GEN_RATIO = 1
DS_RATIO = 0.5
SA_RATIO = 0.25
TD_RATIO = 0.25

parser = argparse.ArgumentParser()

#РѕР±С‰РёРµ РїР°СЂР°РјРµС‚СЂС‹
parser.dataset = 'zinc250k' 
parser.device = 'cpu' 
parser.seed = 42
parser.save = True
parser.model = 'MolHF'
parser.order = 'bfs'
parser.net_type = 'my'

parser.init_checkpoint = './save_pretrain/zinc250k_model/checkpoint.pth'
parser.model_dir = './save_optimization'
parser.property_model_path = None #'qed_moflow_zinc250k_10.pth'

# РїР°СЂР°РјРµС‚СЂС‹ РјРѕРґРµР»Рё
parser.deq_scale = 0.6 
parser.batch_size = 256
parser.lr = 1e-3 
parser.squeeze_fold = 2 
parser.n_block = 4 
parser.a_num_flows = 6 
parser.num_layers = 2 
parser.hid_dim = 256 
parser.b_num_flows = 3 
parser.filter_size = 256 
parser.temperature = 0.6 
parser.learn_prior = True 
parser.inv_conv = True 
parser.inv_rotate = True 
parser.condition = True
parser.hidden = '32'

parser.num_data = None
parser.is_test_idx = False
parser.num_workers = 0
parser.deq_type = 'random'
parser.debug = 'true'
parser.lr_decay = 1
parser.weight_decay = 1e-5

# РѕРїС†РёРѕРЅР°Р»СЊРЅС‹Рµ РїР°СЂР°РјРµС‚СЂС‹ РґР»СЏ РѕРїС‚РёРјРёР·Р°С†РёРё
parser.split = 'moflow'
parser.topk = 1
parser.num_iter = 100
parser.opt_lr = 0.8
parser.topscore = False
parser.consopt = True
parser.ratio = 0.5
parser.max_epochs = 200


class PropNet(nn.Module):
    def __init__(self, input_size=512, hidden_size=[128, 32]):
        super(PropNet, self).__init__()

        self.latent_size = input_size
        self.hidden_size = hidden_size

        vh = (self.latent_size,) + tuple(hidden_size) + (1,)
        modules = []
        for i in range(len(vh)-1):
            modules.append(nn.Linear(vh[i], vh[i+1]))
            if i < len(vh) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, h):
        output = self.net(h)
        return output


class OptimModel(nn.Module):
    def __init__(self, gen_model:MolHF, hidden_size):
        super(OptimModel, self).__init__()
        
        self.model = gen_model

        self.latent_node_length = gen_model.latent_node_length
        self.latent_edge_length = gen_model.latent_edge_length
        self.latent_size = self.latent_node_length + self.latent_edge_length
        
        self.hidden_size = hidden_size

        self.ds_model = PropNet(self.latent_size, hidden_size)
        self.sa_model = PropNet(self.latent_size, hidden_size)
        self.td_model = PropNet(self.latent_size, hidden_size)

    def encode(self, x, adj):
        z, _, _  = self.model(x, adj)  # z = [h, adj_h]
        return z
    
    def forward(self, x, adj):
        z = self.encode(x, adj)
        h = self.model.to_latent_format(z)
        out_ds = self.ds_model(h)
        out_sa = self.sa_model(h)
        out_td = self.td_model(h)
        return out_ds, out_sa, out_td

    def reverse(self, z):
        out = self.model.to_molecule_format(z)
        x, adj = self.model.reverse(out, true_adj=None)
        return x, adj
    

def train_model(opt_model, optimizer, train_loader, metrics, tr, epoch, lrn_set=['DS']):
    '''
    Р”РµР»Р°РµС‚ РїСЂРѕС…РѕРґ РїРѕ РѕРґРЅРѕР№ СЌРїРѕС…Рµ СЃ С€Р°РіРѕРј РѕРїС‚РёРјРёР·Р°С‚РѕСЂР°
    '''
    log_step = 20
    train_iter_per_epoch = len(train_loader)
    global GEN_RATIO, DS_RATIO, SA_RATIO, TD_RATIO
    
    print("Training...")
    opt_model.train()

    total_pd_y = []
    total_true_y = []

    total_pd_sa = []
    total_true_sa_y = []
    
    total_pd_td = []
    total_true_td_y = []
    
    for i, batch in enumerate(train_loader):

        if i == 2:
            break

        x = batch['node'].to(parser.device)   # (bs,9,5)
        adj = batch['adj'].to(parser.device)   # (bs,4,9, 9)
        true_y = batch['property'][:,0].float().unsqueeze(1).to(parser.device)
        true_sa_y = batch['property'][:,1].float().unsqueeze(1).to(parser.device)
        true_td_y = batch['property'][:,2].float().unsqueeze(1).to(parser.device)

        # model and loss
        optimizer.zero_grad()
        y, sa_y, td_y = opt_model(x, adj)

        total_pd_y.append(y)
        total_true_y.append(true_y)

        total_pd_sa.append(sa_y)
        total_true_sa_y.append(true_sa_y)

        total_pd_td.append(td_y)
        total_true_td_y.append(true_td_y)
        
        if 'Gen' in lrn_set:
            out_z, out_logdet, _ = opt_model.model(x, adj)
            loss_node, loss_edge = opt_model.model.log_prob(out_z, out_logdet)
            loss_gen = loss_node + loss_edge
        else:
            loss_gen = torch.tensor([0], requires_grad=False)

        if 'DS' in lrn_set:
            loss_ds = metrics(y, true_y)
        else:
            loss_ds = torch.tensor([0], requires_grad=False)

        if 'SA' in lrn_set:
            loss_sa = metrics(sa_y, true_sa_y)
        else:
            loss_sa = torch.tensor([0], requires_grad=False)

        if 'TD' in lrn_set:
            loss_td = metrics(td_y, true_td_y)
        else:
            loss_td = torch.tensor([0], requires_grad=False)

        loss = loss_gen * GEN_RATIO + loss_ds * DS_RATIO + loss_sa * SA_RATIO + loss_td * TD_RATIO
        
        loss.backward()
        optimizer.step()
        tr.update()
        
        # Print log info
        if (i + 1) % log_step == 0:  # i % args.log_step == 0:
            print('Epoch [{}/{}], Iter [{}/{}], loss: {:.5f}, loss_gen: {:.5f}, loss_prop: {:.5f}, loss_sa: {:.5f}, loss_td: {:.5f}, {:.2f} sec/iter, {:.2f} iters/sec: '.
                    format(epoch + 1, parser.max_epochs, i + 1, train_iter_per_epoch,
                            loss.item(), loss_gen.item(), loss_ds.item(), loss_sa.item(), loss_td.item(),
                            tr.get_avg_time_per_iter(), tr.get_avg_iter_per_sec()))
            tr.print_summary()
    
    total_pd_y = torch.cat(total_pd_y, dim=-1)
    total_true_y = torch.cat(total_true_y, dim=-1)
    
    mse = metrics(total_pd_y, total_true_y)
    mae = mean_absolute_error(total_true_y.cpu().detach().numpy(), total_pd_y.cpu().detach().numpy())
    r2 = r2_score(total_true_y.cpu().detach().numpy(), total_pd_y.cpu().detach().numpy())

    with open('mse_tr.txt', 'a') as f:
        f.write(f'{r2},')

    print("Training, loss_mle:{}, loss_prop:{}, mse:{}, mae:{}, r2:{}".format(loss_gen.item(), loss_ds.item(), mse, mae, r2))


def validate_model(model, valid_loader, metrics, col, tr, epoch):
    log_step = 20
    valid_iter_per_epoch = len(valid_loader)
    
    print("Validating...")    
    model.eval()
    total_pd_y = []
    total_true_y = []
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):

            if i == 2:
                break

            x = batch['node'].to(args.device)   # (bs,9,5)
            adj = batch['adj'].to(args.device)   # (bs,4,9, 9)
            true_y = batch['property'][:, col].unsqueeze(1).float().to(args.device)
            # model and loss
            y, _, _ = model(x, adj)
            total_pd_y.append(y)
            total_true_y.append(true_y)
            loss_prop = metrics(y, true_y)
            tr.update()
            # Print log info
            if (i + 1) % log_step == 0:  # i % args.log_step == 0:
                print('Epoch [{}/{}], Iter [{}/{}], loss_prop: {:.5f}, {:.2f} sec/iter, {:.2f} iters/sec: '.
                        format(epoch + 1, args.max_epochs, i + 1, valid_iter_per_epoch,
                                loss_prop.item(),
                                tr.get_avg_time_per_iter(), tr.get_avg_iter_per_sec()))
                tr.print_summary()
        total_pd_y = torch.cat(total_pd_y, dim=-1)
        total_true_y = torch.cat(total_true_y, dim=-1)
        mse = metrics(total_pd_y, total_true_y)
        mae = mean_absolute_error(total_true_y.cpu().detach().numpy(), total_pd_y.cpu().detach().numpy())
        r2 = r2_score(total_true_y.cpu().detach().numpy(), total_pd_y.cpu().detach().numpy())

        with open('r2_val.txt', 'a') as f:
            f.write(f'{r2},')

        print("Validating, loss_prop:{}, mse:{}, mae:{}, r2:{}".format(loss_prop.item(), mse, mae, r2))
        
    return r2   


def fit_model(opt_model, train_loader, val_loader, args, property_model_path, lrn_set=['DS']):
    start = time()
    print("Start at Time: {}".format(ctime()))
    print('Moduls for learning: ', lrn_set)

    with open('r2_tr.txt', 'w') as f:
        pass
    with open('r2_val.txt', 'w') as f:
        pass
    
    # Loss and optimizer
    metrics = nn.MSELoss()
    best_metrics = float('-inf')
    optimizer = torch.optim.Adam(opt_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    train_iter_per_epoch = len(train_loader)
    valid_iter_per_epoch = len(val_loader)
    tr = TimeReport(total_iter = args.max_epochs * (train_iter_per_epoch+valid_iter_per_epoch))

    moduls_dict = {'Gen': opt_model.model, 'DS': opt_model.ds_model, 'SA': opt_model.sa_model, 'TD': opt_model.td_model}

    for modul in set(moduls_dict.keys()).difference(lrn_set):
        for param in moduls_dict[modul].parameters():
            param.requires_grad_(False)
    
    for epoch in range(args.max_epochs):
        print("In epoch {}, Time: {}".format(epoch + 1, ctime()))
        # op.generate_molecule(model, train_loader, args, epoch) # РїСЂРѕРІРµСЂРєР° С‚РµРєСѓС‰РµРіРѕ РєР°С‡РµСЃС‚РІР° РіРµРЅРµСЂР°С†РёРё СЃ РїСЂРёРЅС‚Р°РјРё РІР°Р»РёРґРЅРѕСЃС‚Рё Рё С‚.Рґ. 
        
        train_model(opt_model, optimizer, train_loader, metrics, tr, epoch, lrn_set)
        cur_metrics = validate_model(opt_model, valid_loader, metrics, 0, tr, epoch)
        
        if best_metrics < cur_metrics:
            best_metrics = cur_metrics
            print("Epoch {}, saving {} regression model to: {}".format(epoch+1, args.net_type, property_model_path))
            torch.save(opt_model.state_dict(), property_model_path)
        
    tr.print_summary()
    tr.end()
    
    print("The model's training is done. Start at {}, End at {}, Total {:.2f}".
          format(ctime(start), ctime(), time()-start))
    return opt_model


def load_property_csv(filename, normilize=True):

    df = pd.read_csv(filename)  # smiles, DS, SA, TD

    if normilize:
        df['DS'] = (df['DS'] - df['DS'].mean()) / df['DS'].std()
        df['SA'] = (df['SA'] - df['SA'].mean()) / df['SA'].std()
        df['TD'] = (df['TD'] - df['TD'].mean()) / df['TD'].std()
        
    tuples = [tuple(x[1:]) for x in df.values]

    print('Load {} done, length: {}'.format(filename, len(tuples)))
    return tuples


start = time()
print("Start at Time: {}".format(ctime()))
args = parser
# configuration
num2atom = {0: 6, 1: 7, 2: 8, 3: 9, 4: 15, 5: 16, 6: 17, 7: 35, 8: 53}
atom_valency = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

data_path = os.path.join('./data_preprocessed', args.dataset)
with open(os.path.join(data_path, 'config.txt'), 'r') as f:
    data_config = eval(f.read())

with open("./data_preprocessed/zinc250k/idx.json", "r") as f:
    train_idx, valid_idx = json.load(f)
dataset = PretrainDataset("./data_preprocessed/{}".format(args.dataset), data_config, args)
train_dataset = deepcopy(dataset)
train_dataset._indices = train_idx # РґР°РЅРЅС‹Рµ С…СЂР°РЅСЏС‚СЃСЏ РІСЃРµ, РЅРѕ Р±РµСЂСѓС‚СЃСЏ С‚РѕР»СЊРєРѕ С‚Рµ, РєРѕС‚РѕСЂС‹Рµ РµСЃС‚СЊ РІ СЃРїРёСЃРєРµ РёРЅРґРµРєСЃРѕРІ
valid_dataset = deepcopy(dataset)
valid_dataset._indices = valid_idx # Р°РЅР°Р»РѕРіРёС‡РЅРѕ

if args.hidden in ('', ','):
    hidden = []
else:
    hidden = [int(d) for d in args.hidden.strip(',').split(',')]
print('Hidden dim for output regression: ', hidden)

if args.property_model_path is None:
    print('in')
    mol_property = load_property_csv('./docking/DS_data/docking_dataset.csv')

    train_dataset.is_mol_property = True
    train_dataset.mol_property = mol_property
    valid_dataset.is_mol_property = True
    valid_dataset.mol_property = mol_property

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,collate_fn=PretrainDataset.collate_fn, num_workers=args.num_workers, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,collate_fn=PretrainDataset.collate_fn, num_workers=args.num_workers, drop_last=True)

    property_model_path = os.path.join(args.model_dir, '{}_{}-{}-{}-{}_{}.pth'.format(args.net_type, 1, 0.5, 0.25, 0.25, 'all'))
    
    gen_model = MolHF(data_config, args).to(args.device)
    op.initialize_from_checkpoint(gen_model, args)

    opti_model = OptimModel(gen_model, hidden).to(args.device)
    property_model = fit_model(opti_model, train_loader, valid_loader, args, property_model_path, ['DS', 'SA', 'TD'])   
# else:
#     prop_list = load_property_csv('tmp_docking_dataset.csv')
#     train_prop = [prop_list[i] for i in train_idx]

#     # DMNP
#     dmnp_smiles = 'CC1CCC(C2=C1C=CC(=C2)C)C(C)CCC(=O)O'
#     train_prop = [tuple(op.get_mol_property(dmnp_smiles) + [dmnp_smiles])]
    
#     test_prop = [prop_list[i] for i in valid_idx]
#     property_model_path = os.path.join(args.model_dir, args.property_model_path)
    
#     model = MolHF(data_config, args).to(args.device)
#     op.initialize_from_checkpoint(model, args)
    
#     property_model = OptimModel(model, hidden).to(args.device)
#     property_model.load_state_dict(torch.load(property_model_path, map_location=args.device))

#     property_model.eval()
