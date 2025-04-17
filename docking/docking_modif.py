import os
from multiprocessing import Pool
from subprocess import run
import glob
import numpy as np
from tempfile import NamedTemporaryFile
import pandas as pd
from time import time
from datetime import datetime
from tqdm import tqdm


def parse_output(result, error_val):
    result_lines = result.split('\n')
    check_result = False
    affinity = error_val

    for result_line in result_lines:
        if result_line.startswith('-----+'):
            check_result = True
            continue
        if not check_result:
            continue
        if result_line.startswith('Writing output'):
            break
        if result_line.startswith('Refine time'):
            break
        lis = result_line.strip().split()
        if not lis[0].isdigit():
            break
        affinity = float(lis[1])
        break
    return affinity


def dock_score(sml):

    box_center = [147, 154, -14]
    box_size = [20, 20, 20]

    receptor = "/beegfs/home/p.zhilyaev/NFmol/NF_mol_gen/docking/receptor_chim.pdbqt"
    vina_program = "/beegfs/home/p.zhilyaev/NFmol/NF_mol_gen/docking/qvina02"

    docking_res = '/beegfs/home/p.zhilyaev/NFmol/NF_mol_gen/docking/'
    ligand = os.path.join(docking_res, 'ligand.pdbqt')
    docking_file = os.path.join(docking_res, 'docking.pdbqt')
    
    run_line = "/beegfs/home/p.zhilyaev/NFmol/obabel-3.1.1/bin/obabel -:{} --gen3D -h -opdbqt -O {} --partialcharge gasteiger".format(sml, ligand)
    result = run(run_line.split(), capture_output=True, text=True, timeout=None, env=os.environ)

    try:
        os.chmod(vina_program, 0o755)  # rwxr-xr-x permissions
    except Exception as e:
        print("Failed to set permissions for vina program: {}".format(e))
        return e
    
    exhaustiveness = 1 # РєРѕР»РёС‡РµСЃС‚РІРѕ РїРѕРїС‹С‚РѕРє РїРѕРёСЃРєР°
    num_modes = 10 # РєРѕР»-РІРѕ РІР°СЂРёР°РЅС‚РѕРІ СЃРІСЏР·С‹РІР°РЅРёСЏ РґР»СЏ СЃРѕС…СЂР°РЅРµРЅРёСЏ
    seed = 42
    run_line = vina_program
    run_line += " --receptor {} --ligand {} --out {}".format(receptor, ligand, docking_file)
    run_line += " --center_x {} --center_y {} --center_z {}".format(*box_center)
    run_line += " --size_x {} --size_y {} --size_z {}".format(*box_size)
    run_line += " --num_modes {}".format(num_modes)
    run_line += " --exhaustiveness {}".format(exhaustiveness)
    run_line += " --seed {}".format(seed)
    
    result = run(run_line.split(), capture_output=True, text=True, timeout=None)
    
    return parse_output(result.stdout, 0)



def res_out(res, path, b1, b2):

    if not os.path.exists(path):
        os.makedirs(path)
    res_arr = np.array(res)
    res_arr.dump(os.path.join(path, f'docking_scores_{b1}-{b2}.npy'))
    try:
        print('Обработано - {}, min - {}, max - {}'.format(len(res_arr[res_arr != 0]), min(res_arr[res_arr != 0]), max(res_arr[res_arr != 0])))
    except ValueError as ve:
        print(ve)


def main():
    start = time()
    box_center = [147, 154, -14]
    box_size = [20.375, 20.375, 20.375]
    
    receptor = "receptor_ADT.pdbqt"
    vina_program = "Q:\\Program Files (x86)\\The Scripps Research Institute\\Vina\\vina.exe"
    print(f"vina_program is {vina_program}")

    mol_class = 'gen'
    smiles_data = pd.read_csv('MolHF_gen.csv')
    smiles_list = smiles_data['SMILES'].values[:100].tolist()

    # mol_class = 'DMNP'
    # smiles_list = ['CC1CCC(C2=C1C=CC(=C2)C)C(C)CCC(=O)O']
    
    # mol_class = 'zinc'
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # parent_dir = os.path.dirname(script_dir)
    # file_path = os.path.join(parent_dir, './dataset/zinc250k/zinc250k.smi')
    # fp = open(file_path, 'r')
    # smiles_list = [smiles.strip() for smiles in fp]

    batch_size = 10
    start_point = 0
    res = []
    dt = datetime.now()
    np_out_path = './{}_DS/{}_{:02d}-{:02d}-{:02d}'.format(mol_class, dt.date(), dt.hour, dt.minute, dt.second)
    
    for i, smile in tqdm(enumerate(smiles_list[start_point:]), total=len(smiles_list[start_point:])):

        if i % batch_size == 0 and i != 0:
            print('Время работы: {:.2f}'.format(time() - start))
            res_out(res, np_out_path, ((i - 1) // batch_size) * batch_size, (i // batch_size) * batch_size - 1)
            # if not os.path.exists(np_out_path):
            #     os.makedirs(np_out_path)
            # res_arr = np.array(res)
            # res_arr.dump(os.path.join(np_out_path, f'docking_scores_{((i - 1) // batch_size) * batch_size}-{(i // batch_size) * batch_size - 1}.npy'))
            # try:
            #     print('Время работы: {:.2f}, обработано - {}, min - {}, max - {}'.format(time() - start, len(res_arr[res_arr != 0]), min(res_arr[res_arr != 0]), max(res_arr[res_arr != 0])))
            # except ValueError as ve:
            #     print(ve)
            res = []
        
        docking_res = ''
        ligand = os.path.join(docking_res, 'ligand')
        docking_file = os.path.join(docking_res, 'docking.pdbqt')
        
        run_line = "obabel -:{} --gen3D -h -opdbqt -O {}".format(smile, ligand)
        result = run(run_line.split(), capture_output=True, text=True, timeout=None, env=os.environ)
        # print(result.stdout)

        try:
            os.chmod(vina_program, 0o755)  # rwxr-xr-x permissions
        except Exception as e:
            print(f"Failed to set permissions for vina program: {e}")
            return e
        exhaustiveness = 1
        num_modes = 10 
        seed = 150
        run_line = vina_program
        run_line += " --receptor {} --ligand {} --out {}".format(receptor, ligand, docking_file)
        run_line += " --center_x {} --center_y {} --center_z {}".format(*box_center)
        run_line += " --size_x {} --size_y {} --size_z {}".format(*box_size)
        run_line += " --num_modes {}".format(num_modes)
        run_line += " --exhaustiveness {}".format(exhaustiveness)
        run_line += " --seed {}".format(seed)
        result = run(run_line.split(), capture_output=True, text=True, timeout=None)
        # print(result.stdout)
        res.append(parse_output(result.stdout, 0))

    res_out(res, np_out_path, ((i - 1) // batch_size) * batch_size, len(smiles_list) - 1)

    return
    
if __name__ == "__main__":
    main()