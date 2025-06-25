import json
from multiprocessing.pool import Pool
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ML.cMBDF import get_convolutions, get_cmbdf
from experiments import execute_xtb_run
from processing_utils import read_energy_gradient, xyz_to_np
from ML.KernelRidge import GridSearchCV_local, KRR_local
from setup_experiment import generate_xtb_parameter_file


if __name__ == "__main__":
    random_seed = 42
    np.random.seed(random_seed)

    convs = get_convolutions(gradients=True)

    rdkit_path = Path(__file__).parent.parent / "rdkit_uniques_1500_molecules_42_seed"
    reference_path = Path(__file__).parent.parent / "uniques_1500_molecules_42_seed"
    BO_experiments_path = (
        Path(__file__).parent.parent / "BO" / "optimizations_forces_final"
    )
    molecules = []
    for molecule in BO_experiments_path.iterdir():
        if (molecule / "BO_results.json").exists():
            molecules.append(molecule)

    # molecules = list(BO_experiments_path.iterdir())
    molecules = [molecule.stem for molecule in molecules]

    np.random.shuffle(molecules)

    metrics = {}

    test_set_size = 200
    for train_set_size in tqdm([50, 100, 200, 400, 800]):
        # Setup splits
        test = molecules[:test_set_size]
        train = molecules[test_set_size : test_set_size + train_set_size]

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

        max_len = max([len(x[2]) for x in x_train + x_test])

        train_reps = np.array([x[3] for x in x_train])
        train_charges = np.array(
            [
                np.concatenate([x[2], np.array([0] * (max_len - len(x[2])))])
                for x in x_train
            ]
        )

        test_reps = np.array([x[3] for x in x_test])
        test_charges = np.array(
            [
                np.concatenate([x[2], np.array([0] * (max_len - len(x[2])))])
                for x in x_test
            ]
        )

        # Grid search for best parameters
        param_grid = {
            "lambda": [1e-3, 1e-6, 1e-9, 1e-10],
            "length": [0.1 * (2**i) for i in range(14)],
        }
        best_params = GridSearchCV_local(
            train_reps, train_charges, y_train, param_grid, cv=4
        )

        # Infer
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

        mae = np.mean(np.abs(preds - y_test), axis=0)
        # mape = np.mean(np.abs((preds - y_test) / y_test), axis=0)
        metrics[train_set_size] = {
            "ksd": mae[0],
            "kpd": mae[1],
            "kp": mae[2],
            "ks": mae[3],
            "kexp": mae[4],
        }
        # metrics[train_set_size] = {
        #     "ksd": mape[0],
        #     "kpd": mape[1],
        #     "kp": mape[2],
        #     "ks": mape[3],
        #     "kexp": mape[4],
        # }

        xtb_outdir = Path(__file__).parent / "xtb" / str(train_set_size)

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
                        xtb_outdir / molecule / "xtb_params.txt",
                        rdkit_path / f"{molecule}.xyz",
                        xtb_outdir / molecule,
                        True,
                        False,
                    ),
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

            with open(BO_experiments_path / molecule / "BO_results.json", "r") as f:
                bo_results = json.load(f)
            max_atomic_forces_label.append(bo_results["best_value"])

        metrics[train_set_size]["max_atomic_forces"] = np.mean(
            max_atomic_forces
        ).tolist()

        metrics[train_set_size]["max_atomic_forces_label"] = np.mean(
            max_atomic_forces_label
        ).tolist()

    with open(Path(__file__).parent / "krr_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
