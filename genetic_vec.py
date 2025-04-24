import random
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.DataStructs import FingerprintSimilarity, TanimotoSimilarity
from rdkit.Chem import Crippen
from deap import base, creator, tools, algorithms
from docking.docking_modif import dock_score
from envs.sascorer import calculateScore
from models import MolHF
from utils import construct_mol
import argparse
import os
from optimize_property import initialize_from_checkpoint, smiles_to_adj, get_mol_data

rdLogger = RDLogger.logger()
rdLogger.setLevel(RDLogger.ERROR)


def arg_parse():
    parser = argparse.ArgumentParser(description='OptiModel')

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='./save_optimization')
    parser.add_argument('--dataset', type=str, default='zinc250k', choices=['zinc1500k', 'zinc250k'],
                        help='dataset name')
    parser.add_argument('--device', type=str, default='cpu')
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
    parser.add_argument('--max_epochs', type=int, default=5, help='How many epochs to run in total?')

    parser.add_argument('--temperature', type=float, default=0.6,
                        help='temperature of the gaussian distributions')
    parser.add_argument('--ratio', type=str, default='1,10,5,5', help='coefficients in loss')

    parser.add_argument('--gen_num', type=int, default=100, help='Number of generated molecules')

    return parser.parse_args()


N_BITS = 1024
num2atom = {0: 6, 1: 7, 2: 8, 3: 9, 4: 15, 5: 16, 6: 17, 7: 35, 8: 53}
atom_valency = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

args = arg_parse()

data_path = os.path.join('./data_preprocessed', args.dataset)
with open(os.path.join(data_path, 'config.txt'), 'r') as f:
    data_config = eval(f.read())

gen_model = MolHF(data_config, args)
initialize_from_checkpoint(gen_model, args)


def tanimoto(smiles, target_fp):
    """Функция для расчета расстояния Танимото до целевой молекулы"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0  # Штраф за невалидные молекулы
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=N_BITS)
    return TanimotoSimilarity(fp, target_fp)

# Целевая молекула для сравнения по Танимото
target_smiles = "CC1CCC(C2=C1C=CC(=C2)C)C(C)CCC(=O)O"
target_mol = Chem.MolFromSmiles(target_smiles)
target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2, nBits=N_BITS)

# Исходный набор молекул
known_mols = {
    'Cc1ccc2c(c1)C(C(C)CNC(=O)O)CCC2C': (-8.2, 3.563399759919447, 0.7333333333333333),
    'Cc1ccc2c(c1)C(C(C)COC(=O)O)CCC2C': (-7.9, 3.598782008440157, 0.7333333333333333),
    'Cc1ccc2c(c1)C(C(C)NCC(=O)O)CCC2C': (-7.4, 3.5622920676117547, 0.7049180327868853),
    'Cc1ccc2c(c1)C(C(C)CCC(=O)O)COC2C': (-7.8, 3.623824907848441, 0.7049180327868853),
    'Cc1ccc2c(c1)C(C(C)OCC(=O)O)CCC2C': (-7.6, 3.6754124818129377, 0.7049180327868853)
}

def get_initial_vec(smiles_list):

    start_vec = []

    with torch.no_grad():
        for sml in smiles_list:
            atoms, bond = smiles_to_adj(sml, args.dataset)
            atoms, bond = get_mol_data(atoms, bond, data_config)
            atoms, bond = torch.from_numpy(atoms).unsqueeze(0), torch.from_numpy(bond).unsqueeze(0)
            atoms, bond = atoms.to(args.device), bond.to(args.device)

            mol_z, _, _  = gen_model(atoms, bond)
            h = gen_model.to_latent_format(mol_z)
            start_vec.append(h.numpy())

    return start_vec


initial_vec = get_initial_vec(list(known_mols.keys()))
print(initial_vec)


# Создаем фитнес-функцию для многокритериальной оптимизации
def evaluate(individual):
    """Оцениваем молекулу по трем критериям"""

    global known_mols, num2atom, atom_valency, gen_model

    print('in')
    print(type(individual[0]))
    print(individual[0] + individual[0])
    
    # Расчет всех показателей
    # try:
    out = gen_model.to_molecule_format(torch.tensor(individual[0]))
    x, adj = gen_model.reverse(out, true_adj=None)
    mol = construct_mol(x[0], adj[0], num2atom, atom_valency)[0]
    smiles = Chem.MolToSmiles(mol)
    print(smiles)

    if smiles in known_mols.keys():
        return known_mols[smiles]
    mol = Chem.MolFromSmiles(smiles)
    ds = dock_score(smiles)
    sa = calculateScore(mol)
    tn = tanimoto(smiles, target_fp)
    print(smiles, ds, sa, tn)
    known_mols[smiles] = (ds, sa, tn)
    # except:
    #     ds = 0
    #     sa = 100
    #     tn = 0
    
    # Мы хотим максимизировать docking score и tanimoto, минимизировать SAscore
    return (ds, sa, tn)  # Возвращаем кортеж

# Настройка генетического алгоритма
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, 1.0))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

# Регистрируем функции для создания популяции
toolbox.register("attr_vec", random.choice, initial_vec)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_vec, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Регистрируем генетические операторы
# toolbox.register("mate", tools.cxTwoPoint)
# toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", evaluate)


# Кастомная мутация для нашего случая
def custom_mutate(individual):

    print('Mutation')

    noise_fraction = 0.2
    
    mutated_individual = individual.copy()
    n = len(individual)

    n_to_replace = int(n * noise_fraction)
    indices_to_replace = np.random.choice(n, size=n_to_replace, replace=False)
    gaussian_noise = np.random.normal(0, 1, size=n_to_replace)
    
    mutated_individual[indices_to_replace] = gaussian_noise    

    return mutated_individual,

toolbox.register("mutate", custom_mutate)

# Кастомное скрещивание для SMILES
def custom_crossover(ind1, ind2):

    print('Crossover')
    
    # Выбираем случайные точки раздела
    split1 = random.randint(1, len(ind1)-1)
    split2 = random.randint(1, len(ind2)-1)
    
    # Создаем новых потомков
    new_ind1 = ind1[:split1].copy() + ind2[split2:].copy()
    new_ind2 = ind2[:split2].copy() + ind1[split1:].copy()
    
    return new_ind1, new_ind2

toolbox.register("mate", custom_crossover)

# Основная функция оптимизации
def main():
    random.seed(42)
    population = toolbox.population(n=50)
    hof = tools.ParetoFront(similar=np.allclose)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    population, logbook = algorithms.eaSimple(
        population, toolbox, cxpb=0.5, mutpb=0.2, ngen=20,
        stats=stats, halloffame=hof, verbose=True)
    
    return population, logbook, hof

if __name__ == "__main__":
    pop, log, hof = main()
    
    # Выводим лучшие решения
    print("\nЛучшие решения:")
    for i, ind in enumerate(hof):
        smiles = ind[0]
        ds, sa, tn = ind.fitness.values
        print(f"{i+1}. SMILES: {smiles}")
        print(f"   Docking score: {ds:.2f}, SAscore: {sa:.2f}, Tanimoto: {tn:.2f}")
        print()