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

    rdkit_path = Path(__file__).parent.parent / "rdkit_uniques_100_molecules_42_seed"
    reference_path = Path(__file__).parent.parent / "uniques_100_molecules_42_seed"
    BO_experiments_path = (
        Path(__file__).parent.parent / "BO" / "optimizations_forces_final"
    )
    molecules = []
    for molecule in BO_experiments_path.iterdir():
        if (molecule / "BO_results.json").exists():
            molecules.append(molecule)

    molecules = [molecule.stem for molecule in molecules]

    molecules = molecules[:20]

    with open(Path(__file__).parent / "molecule_reps.pkl", "rb") as f:
        molecule_reps = pickle.load(f)

    nested_dict = lambda: defaultdict(nested_dict)
    metrics = nested_dict()

    test_set_size = 5
    num_outer_folds = 5

    rng = np.random.default_rng(random_seed)

    for train_set_size in tqdm(
        # [50, 100, 200, 400, 800, 1600, 3200, 6400], desc="Training set sizes"
        [10]
    ):
        for fold in tqdm(range(num_outer_folds), desc="Outer folds", leave=False):
            test = rng.choice(molecules, size=test_set_size, replace=False)
            train = rng.choice(
                np.setdiff1d(molecules, test), size=train_set_size, replace=False
            )

            for rep_type, model in tqdm(
                [
                    # ("cmbdf_global", "krr"),
                    ("cmbdf_global", "xgboost"),
                    # ("slatm_global", "krr"),
                    # ("slatm_global", "xgboost"),
                    # ("slatm_local", "krr"),
                    # ("fchl", "krr"),
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
                    with open(
                        BO_experiments_path / molecule / "BO_results.json", "r"
                    ) as f:
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
                    molecule_data = molecule_reps[molecule]
                    charges, rep = (
                        molecule_data["charges"],
                        molecule_data[rep_type],
                    )

                    x_test.append((molecule, charges, rep))
                    with open(
                        BO_experiments_path / molecule / "BO_results.json", "r"
                    ) as f:
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

                max_len = max([len(x[1]) for x in x_train + x_test])

                train_reps = np.array([x[2] for x in x_train])
                train_charges = np.array(
                    [
                        np.concatenate([x[1], np.array([0] * (max_len - len(x[1])))])
                        for x in x_train
                    ]
                )

                test_reps = np.array([x[2] for x in x_test])
                test_charges = np.array(
                    [
                        np.concatenate([x[1], np.array([0] * (max_len - len(x[1])))])
                        for x in x_test
                    ]
                )

                if model == "krr":
                    # Grid search for best parameters
                    param_grid = {
                        "lambda": [1e-3, 1e-6, 1e-9, 1e-10],
                        "length": [0.1 * (2**i) for i in range(14)],
                    }

                    if "local" in rep_type:
                        best_params = GridSearchCV_local(
                            train_reps, train_charges, y_train, param_grid, cv=4
                        )
                        preds = KRR_local(
                            train_reps,
                            train_charges,
                            y_train,
                            test_reps,
                            test_charges,
                            "gaussian",
                            best_params,
                        )

                        preds = preds.squeeze()

                    elif "global" in rep_type:
                        best_params = GridSearchCV(
                            train_reps, y_train, param_grid, cv=4
                        )

                        preds = KRR_global(train_reps, y_train, test_reps, best_params)

                elif model == "xgboost":
                    param_grid = {}
                    xgb_model = MultiOutputRegressor(
                        XGBRegressor(objective="reg:squarederror", n_jobs=-1)
                    )
                    grid_search = GridSearchCV_sklearn(
                        xgb_model,
                        param_grid,
                        cv=4,
                        scoring="neg_mean_squared_error",
                    )
                    grid_search.fit(train_reps, y_train)

                    preds = grid_search.predict(test_reps)

                mae = np.mean(np.abs(preds - y_test), axis=0)
                metrics[train_set_size][rep_type][fold] = {
                    "ksd": mae[0],
                    "kpd": mae[1],
                    "kp": mae[2],
                    "ks": mae[3],
                    "kexp": mae[4],
                }

                xtb_outdir = Path(__file__).parent / "xtb" / str(train_set_size)
                shutil.rmtree(xtb_outdir, ignore_errors=True)
                with Pool(os.cpu_count() - 4) as p:
                    for molecule, infer_params in zip(test, preds):
                        params_dict = {
                            "ksd": infer_params[0],
                            "kpd": infer_params[1],
                            "kp": infer_params[2],
                            "ks": infer_params[3],
                            "kexp": infer_params[4],
                        }
                        generate_xtb_parameter_file(
                            params_dict, xtb_outdir / molecule / "xtb_params.txt"
                        )

                        p.apply_async(
                            execute_xtb_run,
                            args=(
                                reference_path / f"{molecule}.xyz",
                                xtb_outdir / molecule,
                                True,
                                False,
                            ),
                            kwds={
                                "xtb_parameters_file_path": xtb_outdir
                                / molecule
                                / "xtb_params.txt",
                            },
                        )
                    p.close()
                    p.join()

                max_atomic_forces = []
                max_atomic_forces_label = []
                for molecule in test:
                    gradient_path = xtb_outdir / molecule / f"{molecule}.engrad"
                    if not gradient_path.exists():
                        continue
                    _, _, _, energy_gradient = read_energy_gradient(gradient_path)
                    max_atomic_forces.append(np.max(np.abs(energy_gradient)))

                    with open(
                        BO_experiments_path / molecule / "BO_results.json", "r"
                    ) as f:
                        bo_results = json.load(f)
                    max_atomic_forces_label.append(bo_results["best_value"])

                metrics[train_set_size][rep_type][fold]["max_atomic_forces"] = np.mean(
                    max_atomic_forces
                ).tolist()

                metrics[train_set_size][rep_type][fold]["max_atomic_forces_label"] = (
                    np.mean(max_atomic_forces_label).tolist()
                )

    with open(Path(__file__).parent / "krr_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
