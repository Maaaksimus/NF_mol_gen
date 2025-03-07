import os
from multiprocessing import Pool
from subprocess import run
import glob
import numpy as np
from tempfile import NamedTemporaryFile


def parse_output(result, error_val):
    print(result)
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

def main():
    box_center = [10.131, 41.879, 32.097]
    box_size = [20.673, 20.198, 21.362]
    receptor = "Q:\\docking\\fa7.pdbqt"
    vina_program = "Q:\\docking\\qvina02"
    print(f"vina_program is {vina_program}")
    smile = "C=CCOc1ccc(OC(C)(C)S(=O)(=O)N2CCCCC2)c(-c2nnco2)c1"
    docking_res = "Q:\\docking\\"
    ligand = os.path.join(docking_res, 'ligand')
    docking_file = os.path.join(docking_res, 'docking.pdbqt')
    run_line = "obabel -:{} --gen3D -h -opdbqt -O {}".format(smile, ligand)
    print(run_line)
    result = run(run_line.split(), capture_output=True, text=True, timeout=None, env=os.environ)
    print(result.stdout)

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
    new_res = parse_output(result.stdout, KeyError)
    print(new_res)
    
if __name__ == "__main__":
    main()