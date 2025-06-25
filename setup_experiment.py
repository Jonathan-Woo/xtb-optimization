import json
import os
from multiprocessing.pool import Pool, ThreadPool

import numpy as np
import rdkit.Chem as Chem
from tqdm import tqdm

from pathlib import Path


def create_molecule_xyz_file(coordinates, atoms, output_path, compound_name):
    periodic_table = Chem.GetPeriodicTable()

    file = open(
        os.path.join(output_path, f'{compound_name.replace("/", "_")}.xyz'), "w"
    )

    file.write(f"{len(atoms)}\n\n")

    for coordinate, atom in zip(coordinates, atoms):
        x = coordinate[0]
        y = coordinate[1]
        z = coordinate[2]
        file.write(
            f"{periodic_table.GetElementSymbol(int(atom))}\t{x:>11.8f}\t{y:>11.8f}\t{z:>11.8f}\n"
        )

    file.close()


def generate_molecules(seed, n_molecules, type):
    np.random.seed(seed)

    if type == "all":
        data = np.load("reference_data/DFT_all.npz", allow_pickle=True)
        molecules = list(zip(data["coordinates"], data["atoms"], data["compounds"]))
    elif type == "uniques":
        data = np.load("reference_data/DFT_uniques.npz", allow_pickle=True)
        molecules = list(
            zip(data["coordinates"], data["atoms"], data["compounds"], data["graphs"])
        )
    else:
        raise ValueError("Invalid type. Use 'all' or 'uniques'.")

    output_path = f"./{type}_{n_molecules}_molecules_{seed}_seed"
    os.makedirs(output_path, exist_ok=True)

    sampled_indices = np.random.choice(len(molecules), n_molecules, replace=False)

    results = []
    name_to_inchi = {}
    with ThreadPool() as p:
        for index in tqdm(
            sampled_indices,
            desc="Creating xyz files",
        ):
            if type == "all":
                coordinates, atoms, compound_name = molecules[index]
            elif type == "uniques":
                coordinates, atoms, compound_name, inchi = molecules[index]
                name_to_inchi[compound_name] = inchi
            else:
                raise ValueError("Invalid type. Use 'all' or 'uniques'.")

            results.append(
                p.apply_async(
                    create_molecule_xyz_file,
                    args=(coordinates, atoms, output_path, compound_name),
                )
            )

        for result in tqdm(results, desc="Results"):
            result.get()

    if type == "uniques":
        with open(os.path.join(output_path, "name_to_smiles.json"), "w") as f:
            json.dump(name_to_inchi, f, indent=4)


def generate_parameters():
    default_parameter_values = {
        "ks": 1.85,
        "kp": 2.23,
        "kd": 2.23,
        "ksd": 2,
        "kpd": 2,
        "aesshift": 1.2,
        "aesrmax": 5,
        "a1": 0.52,
        "a2": 5,
        "s8": 2.7,
        "s9": 5,
        "kdiff": 2,
        "enscale": 2,
        "ipeashift": 1.78069,
        "gam3s": 1,
        "gam3p": 0.5,
        "gam3d1": 0.25,
        "gam3d2": 0.25,
        "aesexp": 4,
        "alphaj": 2,
        "aesdmp3": 3,
        "aesdmp5": 4,
        "kexp": 1.5,
        "kexplight": 1,
    }

    # selected_parameters = ["ks", "kp", "kexp"]
    selected_parameters = default_parameter_values.keys()

    new_parameter_factors = [
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
    ]

    new_parameters = {}
    for parameter_name, base_value in default_parameter_values.items():
        if parameter_name in selected_parameters:
            new_parameter_list = []
            for factor in new_parameter_factors:
                new_parameter_list.append(
                    ((base_value * (10**5)) * (factor * 10))
                    / 10**6  # Integer arithmetic to avoid floating point errors
                )
            new_parameters[parameter_name] = {
                factor: value
                for factor, value in zip(new_parameter_factors, new_parameter_list)
            }

    with open("parameters.json", "w") as f:
        json.dump(new_parameters, f, indent=4)


def generate_xtb_parameter_file(parameter_changes, outpath):
    with open("./param_gfn2-xtb.txt", "r") as f:
        base_xtb_parameters = f.read().split("\n")

    new_xtb_parameters = base_xtb_parameters.copy()
    for parameter_name, parameter_value in parameter_changes.items():
        for line_number, param_line in enumerate(base_xtb_parameters):
            if param_line.startswith(parameter_name):
                base_xtb_parameter_line_number = line_number

                num_white_spaces = 12 - len(parameter_name)
                new_xtb_parameters[base_xtb_parameter_line_number] = (
                    f"{parameter_name}{' ' * num_white_spaces}{parameter_value:>7.5f}"
                )
                break

    new_xtb_parameters = "\n".join(new_xtb_parameters)

    os.makedirs(outpath.parent, exist_ok=True)

    with open(
        outpath,
        "w",
    ) as f:
        f.write(new_xtb_parameters)


def generate_xtb_parameters_files():
    experiments_path = Path("./experiments")

    with open("./parameters.json", "r") as f:
        parameters_to_test = json.load(f)

    with open("./param_gfn2-xtb.txt", "r") as f:
        base_xtb_parameters = f.read()

    base_xtb_parameters = base_xtb_parameters.split("\n")

    for parameter_name, parameter_values in parameters_to_test.items():
        for parameter_factor, parameter_value in parameter_values.items():

            outpath = (
                experiments_path
                / parameter_name
                / f"{parameter_factor}_{parameter_value}"
                / f"{parameter_name}_{parameter_factor}_{parameter_value}.txt"
            )
            generate_xtb_parameter_file({parameter_name: parameter_value}, outpath)


def generate_dft_input_file(xyz_path, output_path):
    with open(xyz_path, "r") as f:
        lines = f.read().strip().split("\n")

    atom_lines = lines[2:]

    molecule_block = ["molecule {"]
    molecule_block += atom_lines
    molecule_block.append("}")

    psi4_input = """
set {
  g_convergence GAU_TIGHT
  scf_type df
  geom_maxiter 100
  basis cc-pvdz
}

optimize('wb97x-d3')
""".strip()

    full_input = "\n".join(molecule_block) + "\n\n" + psi4_input

    with open(output_path, "w") as f:
        f.write(full_input)


if __name__ == "__main__":
    # generate_molecules(n_molecules=100, seed=42, type="all")
    generate_molecules(n_molecules=10000, seed=42, type="uniques")
    # generate_parameters()
    # generate_xtb_parameters_files()
