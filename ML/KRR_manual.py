import json
from pathlib import Path

import numpy as np

from ML.cMBDF import get_convolutions, get_cmbdf
from processing_utils import xyz_to_np


if __name__ == "__main__":
    random_seed = 42
    np.random.seed(random_seed)

    convs = get_convolutions(gradients=True)

    rdkit_path = Path(__file__).parent.parent / "rdkit_uniques_100_molecules_42_seed"
    reference_path = Path(__file__).parent.parent / "uniques_100_molecules_42_seed"
    BO_experiments_path = Path(__file__).parent.parent / "BO" / "optimizations"
    molecules = list(BO_experiments_path.iterdir())
    molecules = [molecule.name for molecule in molecules]
    train, test = np.split(molecules, [int(len(molecules) * 0.8)])

    x_train = []
    y_train = []
    for molecule in train:
        coords, atoms, charges = xyz_to_np(rdkit_path / f"{molecule}.xyz")
        charges = np.array(charges).reshape(-1)
        coords = np.array(coords).reshape(-1, 3)
        rep = get_cmbdf(charges, coords, convs, 50)
        x_train.append((molecule, atoms, charges, rep))
        with open(BO_experiments_path / molecule / "BO_results.json", "r") as f:
            optimized_params = json.load(f)["best_params"]
        y_train.append(
            [
                optimized_params["ksd"],
                optimized_params["kpd"],
                optimized_params["kp"],
                optimized_params["ks"],
                optimized_params["kexp"],
            ]
        )

    x_test = []
    y_test = []
    for molecule in test:
        coords, atoms, charges = xyz_to_np(rdkit_path / f"{molecule}.xyz")
        charges = np.array(charges).reshape(-1)
        coords = np.array(coords).reshape(-1, 3)
        rep = get_cmbdf(charges, coords, convs, 50)
        x_test.append((molecule, atoms, charges, rep))
        with open(BO_experiments_path / molecule / "BO_results.json", "r") as f:
            optimized_params = json.load(f)["best_params"]
        y_test.append(
            [
                optimized_params["ksd"],
                optimized_params["kpd"],
                optimized_params["kp"],
                optimized_params["ks"],
                optimized_params["kexp"],
            ]
        )

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # generate kernel matrix
    K = np.zeros((len(train), len(train)))
    for i, (_, atoms_i, _, rep_i) in enumerate(x_train):
        for j, (_, atoms_j, _, rep_j) in enumerate(x_train):
            K[i, j] = kernel(
                rep_i,
                atoms_i,
                rep_j,
                atoms_j,
            )

    lmbda = 1e-5
    alpha = np.linalg.solve(K + lmbda * np.eye(len(train)), y_train)

    preds = []
    for _, atoms_i, _, rep_i in x_test:
        k = np.array(
            [
                kernel(rep_i, atoms_i, rep_j, atoms_j)
                for _, atoms_j, _, rep_j in x_train
            ]
        )
        preds.append(k @ alpha)

    preds = np.array(preds)
    print(preds)

    losses = np.mean((preds - y_test) ** 2, axis=0)
    print("Losses:", losses)