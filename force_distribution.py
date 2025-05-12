import json
from multiprocessing import Pool
import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from experiments import rdkit_generate_geometries, execute_xtb_run


if __name__ == "__main__":
    reference_xyz_path = Path(__file__).parent / "./uniques_100_molecules_42_seed"
    base_xtb_parameter_path = Path(__file__).parent / "param_gfn2-xtb.txt"
    output_dir = Path(__file__).parent / "force_distribution"

    experiments_path = Path(__file__).parent / "experiments"

    # with Pool(os.cpu_count() - 1) as p:
    #     results = []
    #     for molecule_xyz_path in tqdm(
    #         list(reference_xyz_path.glob("*.xyz")), desc="Molecules", leave=False
    #     ):
    #         results.append(
    #             p.apply_async(
    #                 execute_xtb_run,
    #                 args=(
    #                     base_xtb_parameter_path,
    #                     molecule_xyz_path,
    #                     output_dir / molecule_xyz_path.stem,
    #                 ),
    #             )
    #         )

    #     for result in tqdm(results, desc="Results"):
    #         result.get()

    total_gradients = []
    x_gradients = []
    y_gradients = []
    z_gradients = []

    for molecule in output_dir.glob("*"):
        molecule_name = molecule.stem
        grad_file = molecule / f"{molecule_name}.engrad"

        with open(grad_file, "r") as f:
            lines = f.readlines()

        lines = lines[11:]
        lines = [line.strip() for line in lines]

        cur_molecule_gradients = []

        for i in range(0, len(lines), 3):
            try:
                x, y, z = lines[i : i + 3]
                cur_molecule_gradients.append(
                    np.linalg.norm(np.array([float(x), float(y), float(z)]))
                )
                x_gradients.append(float(x))
                y_gradients.append(float(y))
                z_gradients.append(float(z))
                total_gradients.append(
                    np.linalg.norm(np.array([float(x), float(y), float(z)]))
                )

            except:
                break

    plt.clf()
    plt.hist(total_gradients, bins=100)
    plt.title("Force Distribution per Atom")
    plt.xlabel("Total Force Magnitude per Atom (Eh/bohr)")
    plt.ylabel("Count")
    plt.savefig("total_force_distribution_per_atom.png")

    plt.clf()

    plt.hist(x_gradients, bins=100)
    plt.title("X Force Distribution")
    plt.xlabel("X Force (Eh/bohr)")
    plt.ylabel("Count")
    plt.savefig("x_force_distribution.png")

    plt.clf()

    plt.hist(y_gradients, bins=100)
    plt.title("Y Force Distribution")
    plt.xlabel("Y Force (Eh/bohr)")
    plt.ylabel("Count")
    plt.savefig("y_force_distribution.png")

    plt.clf()
    plt.hist(z_gradients, bins=100)
    plt.title("Z Force Distribution")
    plt.xlabel("Z Force (Eh/bohr)")
    plt.ylabel("Count")
    plt.savefig("z_force_distribution.png")
