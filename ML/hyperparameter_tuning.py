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
from sklearn.model_selection import RandomizedSearchCV as RandomizedSearchCV_sklearn
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

    with open(Path(__file__).parent / "molecule_reps.pkl", "rb") as f:
        molecule_reps = pickle.load(f)

    nested_dict = lambda: defaultdict(nested_dict)
    metrics = nested_dict()

    rng = np.random.default_rng(random_seed)

    train = molecules

    for rep_type, model in tqdm(
        [
            ("cmbdf_global", "krr"),
            ("cmbdf_global", "xgboost"),
            ("cmbdf_local", "krr"),
            ("slatm_global", "krr"),
            ("slatm_global", "xgboost"),
            # ("slatm_local", "krr"),
            ("fchl", "krr"),
        ],
        desc="Models",
        leave=False,
    ):
        x_train = []
        y_train = []
        for molecule in train:
            molecule_data = molecule_reps[molecule]
            charges, rep = (
                molecule_data["charges"],
                molecule_data[rep_type],
            )

            x_train.append((molecule, charges, rep))
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

        y_train = np.array(y_train)

        max_len = max([len(x[1]) for x in x_train])

        train_reps = np.array([x[2] for x in x_train])
        train_charges = np.array(
            [
                np.concatenate([x[1], np.array([0] * (max_len - len(x[1])))])
                for x in x_train
            ]
        )

        if model == "krr":
            param_grid = {
                "lambda": [1e-3, 1e-6, 1e-9, 1e-10],
                "length": [0.1 * (2**i) for i in range(14)],
            }

            if "local" in rep_type:
                best_params = GridSearchCV_local(
                    train_reps, train_charges, y_train, param_grid, cv=4
                )

            elif "global" in rep_type:
                best_params = GridSearchCV(train_reps, y_train, param_grid, cv=4)

        elif model == "xgboost":
            param_grid = {
                "estimator__n_estimators": [100, 200, 300],
                "estimator__max_depth": [3, 5, 7],
                "estimator__learning_rate": [0.01, 0.1, 0.2],
                "estimator__subsample": [0.8, 1.0],
                "estimator__colsample_bytree": [0.8, 1.0],
            }

            xgb_model = MultiOutputRegressor(
                XGBRegressor(objective="reg:squarederror", n_jobs=-1)
            )
            grid_search = RandomizedSearchCV_sklearn(
                xgb_model,
                param_grid,
                cv=4,
                scoring="neg_mean_squared_error",
                verbose=2,
                n_iter=25,
            )
            grid_search.fit(train_reps, y_train)

            best_params = grid_search.best_estimator_.estimator.get_params()

        metrics[rep_type][model] = best_params

    with open(Path(__file__).parent / "hyperparameters.json", "w") as f:
        json.dump(metrics, f, indent=4)
