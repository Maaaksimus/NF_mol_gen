import random
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.DataStructs import FingerprintSimilarity, TanimotoSimilarity
from rdkit.Chem import Crippen
from deap import base, creator, tools, algorithms
# from docking.docking_modif import dock_score
from envs.sascorer import calculateScore

rdLogger = RDLogger.logger()
rdLogger.setLevel(RDLogger.ERROR)


N_BITS = 1024

def dock_score(sml):
    return random.randrange(-12,-3)

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
initial_smiles = [
    "CCO", "CCN", "CC(=O)O", "C1CCCCC1", "c1ccccc1",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Кофеин
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ибупрофен
    "C1=CC(=C(C=C1Cl)Cl)Cl",  # Трихлорбензол
    "C1CC1", "C=C", "C#N"
]

# Преобразуем SMILES в fingerprint
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=N_BITS)

# Создаем фитнес-функцию для многокритериальной оптимизации
def evaluate(individual):
    """Оцениваем молекулу по трем критериям"""
    smiles = individual[0]
    
    # Расчет всех показателей
    try:
        mol = Chem.MolFromSmiles(smiles)
        ds = dock_score(smiles)
        sa = calculateScore(mol)
        tn = tanimoto(smiles, target_fp)
    except:
        ds = 0
        sa = 100
        tn = 0
    
    # Мы хотим максимизировать docking score и tanimoto, минимизировать SAscore
    return (ds, sa, tn)  # Возвращаем кортеж

# Настройка генетического алгоритма
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

# Регистрируем функции для создания популяции
toolbox.register("attr_smiles", random.choice, initial_smiles)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_smiles, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Регистрируем генетические операторы
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", evaluate)

# def correct_invalid(smiles, max_attempts=5):
#     """
#     Пытается исправить невалидную SMILES строку
#     """
#     for _ in range(max_attempts):
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is not None:
#             return Chem.MolToSmiles(mol)
        
#         # Попробуем добавить/удалить скобки
#         if '(' not in smiles and ')' not in smiles:
#             modified = f'({smiles})'
#             mol = Chem.MolFromSmiles(modified)
#             if mol is not None:
#                 return Chem.MolToSmiles(mol)
        
#         # Попробуем удалить случайный символ
#         if len(smiles) > 1:
#             pos = random.randint(0, len(smiles)-1)
#             modified = smiles[:pos] + smiles[pos+1:]
#             mol = Chem.MolFromSmiles(modified)
#             if mol is not None:
#                 return Chem.MolToSmiles(mol)
        
#         # Попробуем добавить углерод
#         modified = smiles + 'C'
#         mol = Chem.MolFromSmiles(modified)
#         if mol is not None:
#             return Chem.MolToSmiles(mol)
    
#     # Если не удалось исправить, возвращаем простую валидную молекулу
#     return 'C'

# Функция для мутации SMILES
from rdkit import Chem
from rdkit.Chem import rdchem
import random

def mutate_mol(mol: rdchem.Mol, 
               mutation_types: list = ['atom', 'bond', 'fragment', 'scaffold'],
               prob: float = 0.3) -> rdchem.Mol:

    if mol is None or not mol.GetNumAtoms():
        return mol
    
    print('in mutate')
    
    # Создаем редактируемую копию молекулы
    rw_mol = Chem.RWMol(mol)
    
    # Выбираем случайный тип мутации
    mutation_type = random.choice(mutation_types) if random.random() < prob else None
    
    try:
        if mutation_type == 'atom' and rw_mol.GetNumAtoms() > 1:
            # Мутация атома (замена на другой подходящий атом)
            atom_idx = random.randint(0, rw_mol.GetNumAtoms()-1)
            atom = rw_mol.GetAtomWithIdx(atom_idx)
            new_atomic_num = random.choice([6, 7, 8, 9, 15, 16, 17, 35])  # C,N,O,F,P,S,Cl,Br
            atom.SetAtomicNum(new_atomic_num)
            
        elif mutation_type == 'bond' and rw_mol.GetNumBonds() > 0:
            # Мутация связи (изменение порядка связи)
            bond_idx = random.randint(0, rw_mol.GetNumBonds()-1)
            bond = rw_mol.GetBondWithIdx(bond_idx)
            new_bond_type = random.choice([
                Chem.BondType.SINGLE, 
                Chem.BondType.DOUBLE, 
                Chem.BondType.TRIPLE
            ])
            bond.SetBondType(new_bond_type)
            
        elif mutation_type == 'fragment':
            # Мутация фрагмента (добавление/удаление функциональных групп)
            fragments = [
                'C', 'O', 'N', 'F', 'Cl', 'Br', 
                'C=O', 'CO', 'CN', 'C#N', 'CCl', 'CBr'
            ]
            frag = random.choice(fragments)
            frag_mol = Chem.MolFromSmiles(frag)
            
            if frag_mol and rw_mol.GetNumAtoms() > 0:
                # Добавляем фрагмент к случайному атому
                atom_idx = random.randint(0, rw_mol.GetNumAtoms()-1)
                rw_mol.InsertMol(frag_mol)
                rw_mol.AddBond(
                    atom_idx, 
                    rw_mol.GetNumAtoms()-1, 
                    random.choice([Chem.BondType.SINGLE, Chem.BondType.DOUBLE])
                )
                
        elif mutation_type == 'scaffold' and rw_mol.GetNumAtoms() > 5:
            # Мутация скаффолда (изменение углеродного скелета)
            scaffold_ops = [
                'ADD_RING', 'REMOVE_RING', 
                'ADD_CHAIN', 'REMOVE_CHAIN'
            ]
            op = random.choice(scaffold_ops)
            
            if op == 'ADD_RING':
                # Добавляем 5- или 6-членное кольцо
                size = random.choice([5, 6])
                new_atoms = [rw_mol.AddAtom(Chem.Atom(6)) for _ in range(size)]
                for i in range(size):
                    rw_mol.AddBond(new_atoms[i], new_atoms[(i+1)%size], Chem.BondType.SINGLE)
                    
            elif op == 'REMOVE_RING':
                # Удаляем случайное кольцо (если есть)
                ri = mol.GetRingInfo()
                if ri.NumRings() > 0:
                    ring_atoms = list(ri.AtomRings()[random.randint(0, ri.NumRanges()-1)])
                    for atom in sorted(ring_atoms, reverse=True):
                        rw_mol.RemoveAtom(atom)
                        
        # Валидация и очистка структуры
        new_mol = rw_mol.GetMol()
        new_mol = Chem.SanitizeMol(new_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        
        # Если мутация привела к невалидной структуре, пробуем исправить
        if isinstance(new_mol, Chem.rdmolops.SanitizeFlags):
            print('YYYYYEEEE')
            return mol  # Возвращаем исправленную исходную
        
        return new_mol
    
    except:
        return mol  # В случае ошибки возвращаем исправленную исходную

# def correct_mol(mol: rdchem.Mol) -> rdchem.Mol:
    
#     if mol is None:
#         return Chem.MolFromSmiles('C')
    
#     try:
#         # Пробуем стандартную санитаризацию
#         Chem.SanitizeMol(mol)
#         return mol
#     except:
#         try:
#             # Пробуем удалить проблемные атомы
#             rw_mol = Chem.RWMol(mol)
#             problematic = []
#             for atom in rw_mol.GetAtoms():
#                 try:
#                     atom.GetExplicitValence()
#                 except:
#                     problematic.append(atom.GetIdx())
            
#             for idx in sorted(problematic, reverse=True):
#                 rw_mol.RemoveAtom(idx)
            
#             corrected = rw_mol.GetMol()
#             Chem.SanitizeMol(corrected)
#             return corrected if corrected.GetNumAtoms() > 0 else Chem.MolFromSmiles('C')
#         except:
#             # Возвращаем простейшую валидную молекулу
#             return Chem.MolFromSmiles('C')

# def mutate_smiles(smiles, num_mutations=1):
#     mol = Chem.MolFromSmiles(smiles)
#     for _ in range(num_mutations):
#         new_mol = AllChem.MutateMol(mol)  # Простая мутация (может потребовать доработки)
#     return Chem.MolToSmiles(new_mol)

# Кастомная мутация для нашего случая
def custom_mutate(individual):
    print('================')
    print(individual)
    mol = Chem.MolFromSmiles(individual[0])
    new_mol = mutate_mol(mol)
    print(type(new_mol), new_mol)
    individual[0] = Chem.MolToSmiles(new_mol)
    print('================')
    print(individual)
    # individual[0] = mutate_smiles(individual[0])
    return individual,

toolbox.register("mutate", custom_mutate)

# Кастомное скрещивание для SMILES
def custom_crossover(ind1, ind2):
    
    print('in cross')

    s1 = ind1[0]
    s2 = ind2[0]
    
    if len(s1) < 3 or len(s2) < 3:
        return ind1, ind2
    
    # Выбираем случайные точки раздела
    split1 = random.randint(1, len(s1)-1)
    split2 = random.randint(1, len(s2)-1)
    
    # Создаем новых потомков
    new_s1 = s1[:split1] + s2[split2:]
    new_s2 = s2[:split2] + s1[split1:]
    
    # Проверяем валидность
    if Chem.MolFromSmiles(new_s1) is not None:
        ind1[0] = new_s1
    if Chem.MolFromSmiles(new_s2) is not None:
        ind2[0] = new_s2
    
    return ind1, ind2

toolbox.register("mate", custom_crossover)

# Основная функция оптимизации
def main():
    random.seed(42)
    population = toolbox.population(n=50)
    hof = tools.ParetoFront()
    
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
        ds, neg_sa, tn = ind.fitness.values
        print(f"{i+1}. SMILES: {smiles}")
        print(f"   Docking score: {ds:.2f}, SAscore: {-neg_sa:.2f}, Tanimoto: {tn:.2f}")
        print()