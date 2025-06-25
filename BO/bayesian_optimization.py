from functools import partial
import json
from multiprocessing.pool import Pool
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import optuna
import optuna.visualization as vis
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative

from experiments import execute_xtb_run
from processing_utils import pairwise_distance, read_energy_gradient, xyz_to_np
from setup_experiment import generate_xtb_parameter_file

import os

import logging

optuna.logging.set_verbosity(logging.WARNING)


def geometry_loss(xyz_path, reference_geometry):
    xtb_geom = xyz_to_np(xyz_path, ignore_hydrogen=False)[0]
    return np.linalg.norm(
        pairwise_distance(xtb_geom)[0] - pairwise_distance(reference_geometry)[0]
    )


def forces_loss(grad_path):
    _, _, _, total_gradients = read_energy_gradient(grad_path)
    return max(total_gradients)


def loss_fn(params_dict, initial_geometry_path, reference_geometry, outpath):
    generate_xtb_parameter_file(
        params_dict,
        outpath / "xtb_parameters.txt",
    )
    try:
        execute_xtb_run(
            outpath / "xtb_parameters.txt",
            initial_geometry_path,
            outpath,
            single_threaded=True,
            optimize=False,
        )
    except Exception as e:
        print(f"[{initial_geometry_path.stem}] xTB run failed: {e}")
        # return 100000
        return float("inf")

    # return geometry_loss(outpath / 'xtbopt.xyz', reference_geometry)
    return forces_loss(outpath / f"{initial_geometry_path.stem}.engrad")


class loss_fn_executable:
    def __init__(
        self, initial_geometry_path, reference_geometry, outpath, initial_trial_num=0
    ):
        self.initial_geometry_path = initial_geometry_path
        self.reference_geometry = reference_geometry
        self.outpath = outpath
        self.iter_num = initial_trial_num

    def __call__(self, x):
        params = dict(zip(["ksd", "kpd", "kp", "ks", "kexp"], x))

        cur_outpath = self.outpath / f"trial_{self.iter_num}"
        os.makedirs(self.outpath, exist_ok=True)
        self.iter_num += 1

        return loss_fn(
            params, self.initial_geometry_path, self.reference_geometry, cur_outpath
        )


def create_objective(
    molecule, initial_geometry_path, reference_geometry, outdir, n_trials
):
    gradient_norms = []

    def objective(trial):
        params = {
            # "ksd": trial.suggest_float("ksd", 1.6, 2.4),
            # "kpd": trial.suggest_float("kpd", 1.6, 2.4),
            # "kp": trial.suggest_float("kp", 1.784, 2.676),
            # "ks": trial.suggest_float("ks", 1.48, 2.22),
            # "kexp": trial.suggest_float("kexp", 1.2, 1.8),
            "ksd": trial.suggest_float("ksd", 1.2, 2.55),
            "kpd": trial.suggest_float("kpd", 0.892, 3.122),
            "kp": trial.suggest_float("kp", 1.0, 2.8),
            "ks": trial.suggest_float("ks", 1.295, 2.22),
            "kexp": trial.suggest_float("kexp", 1.2, 2.8),
        }

        trial_outpath = outdir / f"trial_{trial.number}"
        trial_outpath.mkdir(parents=True, exist_ok=True)

        loss = loss_fn(params, initial_geometry_path, reference_geometry, trial_outpath)

        if trial.number > 0 and loss < trial.study.best_value:
            # grad_out = trial_outpath / "gradients"
            # grad_out.mkdir(parents=True, exist_ok=True)

            # grad = approx_derivative(
            #     loss_fn_executable(
            #         initial_geometry_path,
            #         reference_geometry,
            #         grad_out,
            #     ),
            #     list(params.values()),
            # )

            # norm_grad = np.linalg.norm(grad)
            # grad_dict = dict(zip(params.keys(), grad.tolist()))
            # grad_dict["norm"] = norm_grad
            # with open(grad_out / "gradients.json", "w") as f:
            #     json.dump(grad_dict, f, indent=4)

            # gradient_norms.append((trial.number, norm_grad))
            # trial.set_user_attr("gradient_norm", norm_grad)

            # if norm_grad < 1e-3:
            #     trial.study.stop()

            try:
                fig = vis.plot_optimization_history(study)
                fig.write_image(
                    outdir / "optimization_history.png", format="png", scale=2
                )
            except Exception as e:
                print(f"[{molecule}] Could not save optimization history: {e}")

            # Save gradient norm plot
            if gradient_norms:
                plt.figure()
                plt.scatter(
                    [x[0] for x in gradient_norms],
                    [x[1] for x in gradient_norms],
                    label="Gradient Norm",
                )

                # Running min of gradient norm
                min_grads = [gradient_norms[0]]
                cur_min = min_grads[0][1]
                for t, g in gradient_norms[1:]:
                    cur_min = min(cur_min, g)
                    min_grads.append((t, cur_min))

                plt.plot(
                    [x[0] for x in min_grads],
                    [x[1] for x in min_grads],
                    color="red",
                    label=f"Min Gradient Norm So Far ({min_grads[-1][1]:.4f})",
                )
                plt.xlabel("Trial Number")
                plt.ylabel("Gradient Norm")
                plt.title(f"Gradient Norms — {molecule}")
                plt.legend()
                plt.grid(True)
                plt.savefig(outdir / "gradient_norms.png", format="png")
                plt.close()

        return loss

    study = optuna.create_study(direction="minimize")
    study.enqueue_trial(
        {
            "ksd": 2.0,
            "kpd": 2.0,
            "kp": 2.23,
            "ks": 1.85,
            "kexp": 1.5,
        }
    )

    study.optimize(objective, n_trials=n_trials)

    # Save best parameters
    BO_results = {
        "initial_value": study.trials[0].value,
        "best_params": study.best_params,
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
    }
    with open(outdir / "BO_results.json", "w") as f:
        json.dump(BO_results, f, indent=4)

    # Save optimization history plot
    try:
        fig = vis.plot_optimization_history(study)
        fig.write_image(outdir / "optimization_history.png", format="png", scale=2)
    except Exception as e:
        print(f"[{molecule}] Could not save optimization history: {e}")

    # Save gradient norm plot
    if gradient_norms:
        plt.figure()
        plt.scatter(
            [x[0] for x in gradient_norms],
            [x[1] for x in gradient_norms],
            label="Gradient Norm",
        )

        # Running min of gradient norm
        min_grads = [gradient_norms[0]]
        cur_min = min_grads[0][1]
        for t, g in gradient_norms[1:]:
            cur_min = min(cur_min, g)
            min_grads.append((t, cur_min))

        plt.plot(
            [x[0] for x in min_grads],
            [x[1] for x in min_grads],
            color="red",
            label="Min Gradient Norm So Far",
        )
        plt.xlabel("Trial Number")
        plt.ylabel("Gradient Norm")
        plt.title(f"Gradient Norms — {molecule}")
        plt.legend()
        plt.grid(True)
        plt.savefig(outdir / "gradient_norms.png", format="png")
        plt.close()


class FixedStepStopper:
    def __init__(self, f, max_iter):
        self.f = f
        self.max_iter = max_iter
        self.count = 0

    def __call__(self, xk):
        self.count += 1
        grad = approx_derivative(self.f, xk, abs_step=1e-5)
        print(
            f"[Step {self.count}] grad = {grad}, ||grad||₂ = {np.linalg.norm(grad):.3e}"
        )
        if self.count >= self.max_iter:
            raise StopIteration


def finite_diff_grad(f):
    return lambda x: approx_derivative(f, x, abs_step=1e-5)


if __name__ == "__main__":
    for n_trials in [2500]:
        optimization_type = f"forces_final"

        initial_geometries_path = (
            Path(__file__).parent.parent / "rdkit_uniques_10000_molecules_42_seed"
        )
        reference_geometry_path = (
            Path(__file__).parent.parent / "uniques_10000_molecules_42_seed"
        )

        constrained_list = list(initial_geometries_path.glob("*.xyz"))
        # 1. Optuna TPE for the first 100 trials
        with Pool(20) as p:
            # for path in initial_geometries_path.glob("*.xyz"):
            for path in constrained_list:
                outdir = (
                    Path(__file__).parent
                    / f"optimizations_{optimization_type}"
                    / path.stem
                )
                outdir.mkdir(parents=True, exist_ok=True)

            paths_to_process = []
            for path in constrained_list:
                outdir = (
                    Path(__file__).parent
                    / f"optimizations_{optimization_type}"
                    / path.stem
                )
                if not (outdir / "BO_results.json").exists():
                    paths_to_process.append(path)

            p.starmap(
                create_objective,
                [
                    (
                        path.stem,
                        reference_geometry_path / f"{path.stem}.xyz",
                        xyz_to_np(
                            reference_geometry_path / f"{path.stem}.xyz",
                            ignore_hydrogen=True,
                        )[0],
                        Path(__file__).parent
                        / f"optimizations_{optimization_type}"
                        / path.stem,
                        n_trials,
                    )
                    # for path in initial_geometries_path.glob("*.xyz")
                    for path in paths_to_process
                ],
            )

        # # 2. Scipy BFGS for the next 100 trials
        # bounds = [
        #     (0.6, 3.4),

        # ]
        # with Pool(8) as p:
        #     for path in constrained_list:
        #         # for path in initial_geometries_path.glob("*.xyz"):
        #         best_params_path = (
        #             Path(__file__).parent / "optimizations" / path.stem / "BO_results.json"
        #         )
        #         with open(best_params_path) as f:
        #             best_params = json.load(f)["best_params"]
        #         best_params = [
        #             best_params["ksd"],
        #             best_params["kpd"],
        #             best_params["kp"],
        #             best_params["ks"],
        #             best_params["kexp"],
        #         ]

        #         scp_instance = loss_fn_executable(
        #             initial_geometry_path=path,
        #             reference_geometry=xyz_to_np(
        #                 reference_geometry_path / f"{path.stem}.xyz"
        #             )[0],
        #             outpath=Path(__file__).parent / "optimizations" / path.stem,
        #             initial_trial_num=500,
        #         )

        #         res = minimize(
        #             scp_instance,
        #             best_params,
        #             method="L-BFGS-B",
        #             bounds=bounds,
        #             options={
        #                 "disp": True,
        #                 "maxiter": 1000,
        #             },
        #             jac=finite_diff_grad(scp_instance),
        #             callback=FixedStepStopper(scp_instance, 100),
        #         )

        #         with open(
        #             (
        #                 Path(__file__).parent
        #                 / "optimizations"
        #                 / path.stem
        #                 / "BFGS_results.json"
        #             ),
        #             "w",
        #         ) as f:
        #             json.dump(
        #                 {
        #                     "best_params": dict(
        #                         zip(
        #                             ["ksd", "kpd", "kp", "ks", "kexp"],
        #                             res.x.tolist(),
        #                         )
        #                     ),
        #                     "best_value": res.fun,
        #                     "jac": res.jac.tolist(),
        #                     "success": res.success,
        #                     "message": res.message,
        #                 },
        #                 f,
        #                 indent=4,
        #             )

        #         exit()

        #         p.apply_async(
        #             minimize,
        #             args=(scp_instance),
        #             kwds={
        #                 "x0": best_params,
        #                 "method": "BFGS",
        #                 "options": {"disp": True},
        #                 "norm": 2,
        #                 "gtol": 1e-99,
        #                 "ftol": 1e-99,
        #             },
        #         )

        #     p.close()
        #     p.join()
