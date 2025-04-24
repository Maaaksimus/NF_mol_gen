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
from distutils.util import strtobool
from time import time, ctime
import optimize_property as op
from envs import environment as env
from envs.timereport import TimeReport
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from utils import set_random_seed

import warnings
warnings.filterwarnings("ignore")

# РџСЂРµРґРѕР±СѓС‡РµРЅРЅР°СЏ РјРѕРґРµР»СЊ РґР»СЏ РѕРїС‚РёРјРёР·Р°С†РёРё

# GEN_RATIO = 1
# DS_RATIO = 10
# SA_RATIO = 100
# TD_RATIO = 10

def arg_parse():
    parser = argparse.ArgumentParser(description='OptiModel')

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='./save_optimization')
    parser.add_argument('--dataset', type=str, default='zinc250k', choices=['zinc1500k', 'zinc250k'],
                        help='dataset name')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=23, help='random seed')
    parser.add_argument("--property_model_path", type=str, default=None)
    parser.add_argument('--split', type=str, default="moflow",
                        help='choose the split type')
    parser.add_argument('--is_test_idx', action='store_true', default=False, 
                        help='whether use test_idx')
    parser.add_argument('--num_data', type=int,
                        default=None, help='num of data to train')
    
    parser.add_argument('--num_workers', type=int, default=10,
                        help='num works to generate data.')
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument('--order', type=str, default='bfs',
                        help='order of atom')
    
    # ******model args******
    parser.add_argument('--deq_type', type=str,
                        default='random', help='dequantization methods.')
    parser.add_argument('--deq_scale', type=float, default=0.6,
                        help='dequantization scale.(only for deq_type random)')
    parser.add_argument('--n_block', type=int, default=4,
                        help='num block')
    parser.add_argument('--condition', action='store_false', default=True,
                        help='latent variables on condition')
    parser.add_argument('--moduls', type=str, default='Gen,DS', help='list of moduls to train')
    
    # ***atom model***
    parser.add_argument('--a_num_flows', type=int, default=6,
                        help='num of flows in RGBlock')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='num of R-GCN layer in GraphAffineCoupling')
    parser.add_argument('--hid_dim', type=int, default=256,
                        help='hidden dim of R-GCN layer')
    parser.add_argument('--st_type', type=str, default='sigmoid',
                        help='architecture of st net, choice: [exp, sigmoid]')
    parser.add_argument('--inv_rotate', action='store_false',
                        default=True, help='whether rotate node feature')
    # ***bond model***
    parser.add_argument('--b_num_flows', type=int, default=3,
                        help='num of flows in bond model')
    parser.add_argument('--filter_size', type=int, default=256,
                        help='num of filter size in AffineCoupling')
    parser.add_argument('--inv_conv', action='store_false',
                        default=True, help='whether use 1*1 conv')
    parser.add_argument('--squeeze_fold', type=int, default=2,
                        help='squeeze fold')
    
    parser.add_argument('--num_iter', type=int, default=200,
                        help='num iter of optimization')
    parser.add_argument('--learn_prior', action='store_false',
                        default=True, help='learn log-var of gaussian prior.')
    parser.add_argument('--init_checkpoint', type=str, default='./save_pretrain/zinc250k_model/checkpoint.pth',
                    help='initialize from a checkpoint, if None, do not restore')
    parser.add_argument('--lr', type=float, default=0.001, help='Base learning rate')
    parser.add_argument('--opt_lr', type=float, default=0.001, help='optimization learning rate')
    parser.add_argument('--lr_decay', type=float, default=1,
                        help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='L2 norm for the parameters')
    parser.add_argument('--hidden', type=str, default="32",
                        help='Hidden dimension list for output regression')
    parser.add_argument('--activation', type=str, default='tanh,tanh', help='Activations between layers')
    parser.add_argument('--max_epochs', type=int, default=5, help='How many epochs to run in total?')

    parser.add_argument('--temperature', type=float, default=0.6,
                        help='temperature of the gaussian distributions')
    parser.add_argument('--ratio', type=str, default='1,10,10,10', help='coefficients in loss')

    parser.add_argument('--gen_num', type=int, default=100, help='Number of generated molecules')

    return parser.parse_args()


class PropNet(nn.Module):
    def __init__(self, input_size=512, hidden_size=[128, 32], activ=[nn.Tanh(), nn.Tanh()]):
        super(PropNet, self).__init__()

        self.latent_size = input_size
        self.hidden_size = hidden_size

        vh = (self.latent_size,) + tuple(hidden_size) + (1,)
        modules = []
        for i in range(len(vh)-1):
            modules.append(nn.Linear(vh[i], vh[i+1]))
            if i < len(vh) - 2:
                modules.append(activ[i])
        self.net = nn.Sequential(*modules)

    def forward(self, h):
        output = self.net(h)
        return output


class OptimModel(nn.Module):
    def __init__(self, gen_model:MolHF, hidden_size, activ):
        super(OptimModel, self).__init__()
        
        self.model = gen_model

        self.latent_node_length = gen_model.latent_node_length
        self.latent_edge_length = gen_model.latent_edge_length
        self.latent_size = self.latent_node_length + self.latent_edge_length
        
        self.hidden_size = hidden_size

        self.ds_model = PropNet(self.latent_size, hidden_size, activ)
        self.sa_model = PropNet(self.latent_size, hidden_size, activ)
        self.td_model = PropNet(self.latent_size, hidden_size, activ)

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

        x = batch['node'].to(args.device)   # (bs,9,5)
        adj = batch['adj'].to(args.device)   # (bs,4,9, 9)
        true_y = batch['property'][:,0].float().unsqueeze(1).to(args.device)
        true_sa_y = batch['property'][:,1].float().unsqueeze(1).to(args.device)
        true_td_y = batch['property'][:,2].float().unsqueeze(1).to(args.device)

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
            loss_gen = torch.tensor([0], requires_grad=False).to(args.device)

        if 'DS' in lrn_set:
            loss_ds = metrics(y, true_y)
        else:
            loss_ds = torch.tensor([0], requires_grad=False).to(args.device)

        if 'SA' in lrn_set:
            loss_sa = metrics(sa_y, true_sa_y)
        else:
            loss_sa = torch.tensor([0], requires_grad=False).to(args.device)

        if 'TD' in lrn_set:
            loss_td = metrics(td_y, true_td_y)
        else:
            loss_td = torch.tensor([0], requires_grad=False).to(args.device)

        loss = loss_gen * GEN_RATIO + loss_ds * DS_RATIO + loss_sa * SA_RATIO + loss_td * TD_RATIO
        
        loss.backward()
        optimizer.step()
        tr.update()
        
        # Print log info
        if (i + 1) % log_step == 0:  # i % args.log_step == 0:
            print('Epoch [{}/{}], Iter [{}/{}], loss: {:.5f}, loss_gen: {:.5f}, loss_prop: {:.5f}, loss_sa: {:.5f}, loss_td: {:.5f}, {:.2f} sec/iter, {:.2f} iters/sec: '.
                    format(epoch + 1, args.max_epochs, i + 1, train_iter_per_epoch,
                            loss.item(), loss_gen.item(), loss_ds.item(), loss_sa.item(), loss_td.item(),
                            tr.get_avg_time_per_iter(), tr.get_avg_iter_per_sec()))

            t_pd_y = torch.cat(total_pd_y, dim=-1)
            t_true_y = torch.cat(total_true_y, dim=-1)
            print('Current R^2 score: ', r2_score(t_true_y.cpu().detach().numpy(), t_pd_y.cpu().detach().numpy()))

            tr.print_summary()
    
    total_pd_y = torch.cat(total_pd_y, dim=-1)
    total_true_y = torch.cat(total_true_y, dim=-1)
    
    mse = metrics(total_pd_y, total_true_y)
    mae = mean_absolute_error(total_true_y.cpu().detach().numpy(), total_pd_y.cpu().detach().numpy())
    r2 = r2_score(total_true_y.cpu().detach().numpy(), total_pd_y.cpu().detach().numpy())

    with open(f'r2_tr_{lrn_set}.txt', 'a') as f:
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

        with open('r2_val_mlp.txt', 'a') as f:
            f.write(f'{r2},')

        print("Validating, loss_prop:{}, mse:{}, mae:{}, r2:{}".format(loss_prop.item(), mse, mae, r2))
        
    return r2   


def fit_model(opt_model, train_loader, val_loader, args, property_model_path, lrn_set=['DS']):
    start = time()
    print("Start at Time: {}".format(ctime()))
    print('Moduls for learning: ', lrn_set)

    with open(f'r2_tr_{lrn_set}.txt', 'w') as f:
        pass
    with open('r2_val_mlp.txt', 'w') as f:
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
            print("Epoch {}, saving {} regression model to: {}".format(epoch+1, args.hidden, property_model_path))
            torch.save(opt_model.state_dict(), property_model_path)
        
    tr.print_summary()
    tr.end()
    
    print("The model's training is done. Start at {}, End at {}, Total {:.2f}".
          format(ctime(start), ctime(), time()-start))
    return opt_model


def load_property_csv(filename, normilize=True):

    df = pd.read_csv(filename)  # smiles, DS, SA, TD

    min_max = lambda prop: (df[prop] - df[prop].min()) / (df[prop].max() - df[prop].min())
    gauss = lambda prop: (df[prop] - df[prop].mean()) / df[prop].std()

    if normilize:
        # m = df['DS'].mean()  # 0.00026
        # std = df['DS'].std() # 2.05
        # mn = df['DS'].min()
        # mx = df['DS'].max()
        # # df['DS'] = 0.5 * (np.tanh(0.01 * ((df['DS'] - m) / std)) + 1)  # [0.35, 0.51]
        # # df['DS'] = (df['DS'] - m) / std
        # lower = -10 # -5 # -10
        # df['DS'] = df['DS'].clip(lower=lower, upper=5)
        # df['DS'] = (df['DS'] - lower) / (mx-lower)

        df['DS'] = df['DS'].clip(-12, -5)
        df['DS'] = min_max('DS')
        
        # df['SA'] = df['SA'].clip(-12, -5)
        df['SA'] = min_max('SA')
        
        df['TD'] = min_max('TD')
        
    tuples = [tuple(x[1:]) for x in df.values]

    print('Load {} done, length: {}'.format(filename, len(tuples)))
    return tuples


start = time()
print("Start at Time: {}".format(ctime()))
args = arg_parse()
# set_random_seed(args.seed)
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

if args.ratio in ('',','):
    ratio = []
else:
    GEN_RATIO, DS_RATIO, SA_RATIO, TD_RATIO = [float(d) for d in args.ratio.strip(',').split(',')]

if args.moduls in ('',','):
    raise ValueError('empty moduls list')
else:
    moduls_list = [mod for mod in args.moduls.strip(',').split(',')]

if args.moduls in ('',','):
    raise ValueError('empty activation list')
else:
    acti_dict = {'tanh': nn.Tanh(), 'sigm': nn.Sigmoid(), 'relu': nn.ReLU()}
    activ = [acti_dict[acti] for acti in args.activation.strip(',').split(',')]

if args.property_model_path is None:
    print('in')
    mol_property = load_property_csv('./docking/DS_data/docking_dataset.csv')

    train_dataset.is_mol_property = True
    train_dataset.mol_property = mol_property
    valid_dataset.is_mol_property = True
    valid_dataset.mol_property = mol_property

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,collate_fn=PretrainDataset.collate_fn, num_workers=args.num_workers, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,collate_fn=PretrainDataset.collate_fn, num_workers=args.num_workers, drop_last=True)
    
    property_model_path = os.path.join(args.model_dir, '{}_{}-{}-{}-{}_{}.pth'.format(args.hidden, GEN_RATIO, DS_RATIO, SA_RATIO, TD_RATIO, moduls_list))
    
    gen_model = MolHF(data_config, args).to(args.device)
    op.initialize_from_checkpoint(gen_model, args)

    opti_model = OptimModel(gen_model, hidden, activ).to(args.device)
    property_model = fit_model(opti_model, train_loader, valid_loader, args, property_model_path, moduls_list)   
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
