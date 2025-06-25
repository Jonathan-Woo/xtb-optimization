from collections import defaultdict
import json
from multiprocessing import Pool
import os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from experiments import rdkit_generate_geometries, execute_xtb_run


if __name__ == "__main__":
    reference_xyz_path = Path(__file__).parent / "./uniques_10000_molecules_42_seed"
    inchi_path = reference_xyz_path / "name_to_smiles.json"
    rdkit_xyz_path = Path(__file__).parent / f"rdkit_{reference_xyz_path.stem}"
    rdkit_xyz_path.mkdir(parents=True, exist_ok=True)

    rdkit_generate_geometries(reference_xyz_path, inchi_path, rdkit_xyz_path)

    # experiments_path = Path(__file__).parent / "experiments_no_opt"

    # single_threaded = True
    # optimize = False

    # with Pool(os.cpu_count() - 4) as p:
    #     results = []
    #     failed_experiments = defaultdict(list)
    #     for xtb_parameter_file_path in tqdm(
    #         list(experiments_path.glob("*/*/*.txt")), desc="Parameters"
    #     ):
    #         print(f"Running experiment with {xtb_parameter_file_path}")
    #         for molecule_xyz_path in tqdm(
    #             list(rdkit_xyz_path.glob("*.xyz")), desc="Molecules", leave=False
    #         ):
    #             results.append(
    #                 [
    #                     p.apply_async(
    #                         execute_xtb_run,
    #                         args=(
    #                             xtb_parameter_file_path,
    #                             molecule_xyz_path,
    #                             xtb_parameter_file_path.parent / molecule_xyz_path.stem,
    #                             single_threaded,
    #                             optimize,
    #                         ),
    #                     ),
    #                     [
    #                         xtb_parameter_file_path,
    #                         molecule_xyz_path,
    #                         xtb_parameter_file_path.parent / molecule_xyz_path.stem,
    #                     ],
    #                 ]
    #             )

    #     for result, args in tqdm(results):
    #         try:
    #             result.get()
    #         except Exception as e:
    #             failed_experiments[args[1].stem].append(args[0].stem)

    #     p.close()
    #     p.join()

    # with open(experiments_path / "failed_experiments", "w") as f:
    #     json.dump(failed_experiments, f, indent=4)
