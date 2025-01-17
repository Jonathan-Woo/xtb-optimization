import json

import numpy as np
import rdkit.Chem as Chem
import os

from tqdm import tqdm


def generate_molecules():
    # Create xyz file
    data = np.load('DFT_all.npz', allow_pickle=True)

    periodic_table = Chem.GetPeriodicTable()

    output_path = './molecules'

    coordinate_length = 10

    for i in tqdm(range(len(data['coordinates']))[:50], desc='Creating xyz files'):
        coordinates = data['coordinates'][i]
        atoms = data['atoms'][i]

        file = open(os.path.join(output_path, f'{data["compounds"][i].replace("/", "_")}.xyz'), 'w')

        file.write(f"{len(atoms)}\n\n")

        for coordinate, atom in tqdm(zip(coordinates, atoms), total=len(atoms), desc='Writing atoms', leave=False):
            x = coordinate[0]
            y = coordinate[1]
            z = coordinate[2]
            file.write(
                f"{periodic_table.GetElementSymbol(int(atom))}\t{x:>11.8f}\t{y:>11.8f}\t{z:>11.8f}\n")

        file.close()


def generate_parameters():
    base_parameters = {
        'ks': 1.85,
        'kp': 2.23,
        'kd': 2.23,
        'ksd': 2,
        'kpd': 2,
        'aesshift': 1.2,
        'aesrmax': 5,
        'a1': 0.52,
        'a2': 5,
        's8': 2.7,
        's9': 5,
        'kdiff': 2,
        'enscale': 2,
        'ipeashift': 1.78069,
        'gam3s': 1,
        'gam3p': 0.5,
        'gam3d1': 0.25,
        'gam3d2': 0.25,
        'aesexp': 4,
        'alphaj': 2,
        'aesdmp3': 3,
        'aesdmp5': 4,
        'kexp': 1.5,
        'kexplight': 1
    }

    new_parameter_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    new_parameters = {}
    for parameter_name, base_value in base_parameters.items():
        new_parameter_list = []
        for factor in new_parameter_factors:
            new_parameter_list.append(((base_value * (10 ** 5)) * (factor * 10)) / 10 ** 6)
        new_parameters[parameter_name] = {factor: value for factor, value in
                                          zip(new_parameter_factors, new_parameter_list)}

    with open("new_parameters.json", 'w') as f:
        json.dump(new_parameters, f, indent=4)


def generate_parameters_files():
    experiments_path = './experiments'

    with open("./new_parameters.json", 'r') as f:
        parameters_to_test = json.load(f)

    with open('./param_gfn2-xtb.txt', 'r') as f:
        base_xtb_parameters = f.read()

    base_xtb_parameters = base_xtb_parameters.split('\n')

    for parameter_name, parameter_values in parameters_to_test.items():
        for parameter_factor, parameter_value in parameter_values.items():

            for line_number, param_line in enumerate(base_xtb_parameters):
                if param_line.startswith(parameter_name):
                    base_xtb_parameter_line_number = line_number
                    break

            new_xtb_parameters = base_xtb_parameters.copy()
            num_white_spaces = 12 - len(parameter_name)
            new_xtb_parameters[
                base_xtb_parameter_line_number] = f"{parameter_name}{' ' * num_white_spaces}{parameter_value:>7.5f}"

            new_xtb_parameters = '\n'.join(new_xtb_parameters)
            new_xtb_parameters_path = os.path.join(experiments_path, parameter_name,
                                                   f"{parameter_factor}_{parameter_value}")

            os.makedirs(new_xtb_parameters_path, exist_ok=True)

            with open(
                    os.path.join(new_xtb_parameters_path, f'{parameter_name}_{parameter_factor}_{parameter_value}.txt'),
                    'w') as f:
                f.write(new_xtb_parameters)


if __name__ == '__main__':
    # generate_molecules()
    generate_parameters()
    generate_parameters_files()
