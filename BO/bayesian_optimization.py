from functools import partial
import json
from multiprocessing.pool import Pool
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import optuna
import optuna.visualization as vis
from scipy.optimize import minimize

from experiments import execute_xtb_run
from processing_utils import xyz_to_np
from setup_experiment import generate_parameter_file


def loss_fn(params_dict, initial_geometry_path, reference_geometry, outpath):
    generate_parameter_file(
        params_dict,
        outpath / "xtb_parameters.txt",
    )
    try:
        execute_xtb_run(
            outpath / "xtb_parameters.txt",
            initial_geometry_path,
            outpath,
            xtb_args=["-P", "1"],
        )
    except Exception as e:
        print(f"[{initial_geometry_path.stem}] xTB run failed: {e}")
        return float("inf")

    xtb_geom = xyz_to_np(outpath / "xtbopt.xyz")[0]
    return np.linalg.norm(xtb_geom - reference_geometry)


def create_objective(molecule, initial_geometry_path, reference_geometry, outdir):
    gradient_norms = []

    def gradient(params_dict, outpath, epsilon=1e-4):
        keys = list(params_dict.keys())
        base = np.array([params_dict[k] for k in keys])
        grad = np.zeros_like(base)
        base_loss = loss_fn(
            params_dict, initial_geometry_path, reference_geometry, outpath
        )

        for i, key in enumerate(keys):
            perturbed = base.copy()
            perturbed[i] += epsilon
            perturbed_dict = dict(zip(keys, perturbed))
            grad[i] = (
                loss_fn(
                    perturbed_dict,
                    initial_geometry_path,
                    reference_geometry,
                    outpath / key,
                )
                - base_loss
            ) / epsilon

        with open(outpath / "gradients.txt", "w") as f:
            for key, value in zip(keys, grad):
                f.write(f"{key}: {value}\n")
            f.write(f"norm: {np.linalg.norm(grad)}\n")
        return grad, np.linalg.norm(grad)

    def objective(trial):
        params = {
            "ksd": trial.suggest_float("ksd", 1.6, 2.4),
            "kpd": trial.suggest_float("kpd", 1.6, 2.4),
            "kp": trial.suggest_float("kp", 1.784, 2.676),
            "ks": trial.suggest_float("ks", 1.48, 2.22),
            "kexp": trial.suggest_float("kexp", 1.2, 1.8),
        }

        trial_outpath = outdir / f"trial_{trial.number}"
        trial_outpath.mkdir(parents=True, exist_ok=True)

        loss = loss_fn(params, initial_geometry_path, reference_geometry, trial_outpath)

        if trial.number % 10 == 0 and trial.number > 0:
            grad_out = trial_outpath / "gradients"
            grad_out.mkdir(parents=True, exist_ok=True)
            grad, norm_grad = gradient(params, grad_out)

            gradient_norms.append((trial.number, norm_grad))
            trial.set_user_attr("gradient_norm", norm_grad)

            if norm_grad < 1e-3:
                trial.study.stop()

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

    study.optimize(objective, n_trials=100)

    # Save best parameters
    with open(outdir / "best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)

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


def scp_loss(x, initial_geometry_path, reference_geometry, outpath):
    params = dict(zip(["ksd", "kpd", "kp", "ks", "kexp"], x))
    return loss_fn(params, initial_geometry_path, reference_geometry, outpath)


if __name__ == "__main__":
    initial_geometries_path = (
        Path(__file__).parent.parent / "rdkit_uniques_100_molecules_42_seed"
    )
    reference_geometry_path = (
        Path(__file__).parent.parent / "uniques_100_molecules_42_seed"
    )

    constrained_list = list(initial_geometries_path.glob("*.xyz"))[:3]
    # 1. Optuna TPE for the first 100 trials
    with Pool(8) as p:
        # for path in initial_geometries_path.glob("*.xyz"):
        for path in constrained_list:
            outdir = Path(__file__).parent / "optimizations" / path.stem
            outdir.mkdir(parents=True, exist_ok=True)

        p.starmap(
            create_objective,
            [
                (
                    path.stem,
                    path,
                    xyz_to_np(reference_geometry_path / f"{path.stem}.xyz")[0],
                    Path(__file__).parent / "optimizations" / path.stem,
                )
                # for path in initial_geometries_path.glob("*.xyz")
                for path in constrained_list
            ],
        )

    # 2. Scipy BFGS for the next 100 trials
    with Pool(8) as p:
        for path in constrained_list:
            # for path in initial_geometries_path.glob("*.xyz"):
            best_params_path = (
                Path(__file__).parent / "optimizations" / path.stem / "best_params.json"
            )
            with open(best_params_path) as f:
                best_params = json.load(f)
            best_params = [
                best_params["ksd"],
                best_params["kpd"],
                best_params["kp"],
                best_params["ks"],
                best_params["kexp"],
            ]
            f = partial(
                scp_loss,
                initial_geometry_path=path,
                reference_geometry=xyz_to_np(
                    reference_geometry_path / f"{path.stem}.xyz"
                )[0],
                outpath=Path(__file__).parent / "optimizations" / path.stem,
            )
            minimize(
                f,
                best_params,
                method="BFGS",
                options={"disp": True},
            )
            # p.apply_async(
            #     minimize,
            #     args=(f),
            #     kwds={"x0": best_params, "method": "BFGS", "options": {"disp": True}},
            # )

        p.close()
        p.join()
