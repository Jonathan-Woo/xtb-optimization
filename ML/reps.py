import pickle
import json
from multiprocessing.pool import Pool
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ML.cMBDF import get_convolutions, get_cmbdf, generate_mbdf
from experiments import execute_xtb_run
from processing_utils import read_energy_gradient, xyz_to_np
from ML.KernelRidge import GridSearchCV_local, KRR_local, GridSearchCV, KRR_global
from setup_experiment import generate_xtb_parameter_file
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV as GridSearchCV_sklearn
from sklearn.multioutput import MultiOutputRegressor

from qml.representations import get_slatm_mbtypes, generate_slatm
from qmllib.representations import generate_fchl19

from collections import defaultdict

import shutil

if __name__ == "__main__":
    random_seed = 42
    convs = get_convolutions(gradients=True)

    rdkit_path = Path(__file__).parent.parent / "rdkit_uniques_100_molecules_42_seed"
    reference_path = Path(__file__).parent.parent / "uniques_100_molecules_42_seed"
    BO_experiments_path = (
        Path(__file__).parent.parent / "BO" / "optimizations_forces_wider_complete"
    )
    molecules = []
    for molecule in BO_experiments_path.iterdir():
        if (molecule / "BO_results.json").exists():
            molecules.append(molecule)

    molecules = [molecule.stem for molecule in molecules]

    unique_charges = set()

    all_charges = []
    all_coords = []
    for molecule in molecules:
        coords, atoms, charges = xyz_to_np(rdkit_path / f"{molecule}.xyz")
        charges = np.array(charges)
        coords = np.array(coords)
        all_charges.append(charges)
        all_coords.append(coords)

        unique_charges.update(charges)

    mbtypes = get_slatm_mbtypes(all_charges, all_coords)

    all_charges = np.array(all_charges, dtype=object)
    all_coords = np.array(all_coords, dtype=object)

    cmbdf_global = generate_mbdf(
        all_charges, all_coords, local=False, progress_bar=True
    )
    cmbdf_local = generate_mbdf(all_charges, all_coords, local=True, progress_bar=True)

    slatm_global = [
        generate_slatm(coord, charge, mbtypes)
        for coord, charge in tqdm(
            zip(all_coords, all_charges),
            desc="Generating SLATM global",
            total=len(all_coords),
        )
    ]

    slatm_pad = 50
    slatm_local = [
        np.array(generate_slatm(coord, charge, mbtypes, local=True))
        for coord, charge in tqdm(
            zip(all_coords, all_charges),
            desc="Generating SLATM local",
            total=len(all_coords),
        )
    ]

    def pad_slatm_local(slatm, pad_len, pad_value=0.0):
        n_atoms, d = slatm.shape
        padded = np.full((pad_len, d), pad_value, dtype=slatm.dtype)
        padded[:n_atoms, :] = slatm
        return padded

    slatm_local = [pad_slatm_local(slatm, slatm_pad) for slatm in slatm_local]

    fchl = [
        generate_fchl19(charge, coord, list(unique_charges), pad=50)
        for coord, charge in tqdm(
            zip(all_coords, all_charges),
            desc="Generating FCHL representations",
            total=len(all_coords),
        )
    ]

    molecule_reps = {}
    for idx in tqdm(
        range(len(molecules)),
        desc="Generating representations",
    ):
        molecule_reps[molecules[idx]] = {
            "cmbdf_global": cmbdf_global[idx],
            "cmbdf_local": cmbdf_local[idx],
            "slatm_global": slatm_global[idx],
            "slatm_local": slatm_local[idx],
            "fchl": fchl[idx],
            "charges": all_charges[idx],
        }

    with open(Path(__file__).parent / "molecule_reps.pkl", "wb") as f:
        pickle.dump(molecule_reps, f)
