from multiprocessing.pool import Pool
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import optuna
import optuna.visualization as vis

from experiments import execute_xtb_run
from processing_utils import xyz_to_np
from setup_experiment import generate_parameter_file


def create_objective(molecule, initial_geometry_path, reference_geometry):
    outdir = Path(__file__).parent / "optimizations" / molecule
    outdir.mkdir(parents=True, exist_ok=True)

    def loss_fn(params_dict, outpath):
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
            print(f"[{molecule}] xTB run failed: {e}")
            return float("inf")

        xtb_geom = xyz_to_np(outpath / "xtbopt.xyz")[0]
        return np.linalg.norm(xtb_geom - reference_geometry)

    def gradient(params_dict, outpath, epsilon=1e-4):
        keys = list(params_dict.keys())
        base = np.array([params_dict[k] for k in keys])
        grad = np.zeros_like(base)
        base_loss = loss_fn(params_dict, outpath)

        for i, key in enumerate(keys):
            perturbed = base.copy()
            perturbed[i] += epsilon
            perturbed_dict = dict(zip(keys, perturbed))
            grad[i] = (loss_fn(perturbed_dict, outpath / key) - base_loss) / epsilon

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

        loss = loss_fn(params, trial_outpath)

        # Only compute gradient every 10 trials
        if trial.number % 10 == 0 and trial.number > 0:
            grad_out = trial_outpath / "gradients"
            grad_out.mkdir(parents=True, exist_ok=True)
            grad, norm_grad = gradient(params, grad_out)

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

    with open(outdir / "best_params.txt", "w") as f:
        f.write(str(study.best_params))

    fig = vis.plot_optimization_history(study)
    fig.write_image(outdir / "optimization_history.png", format="png", scale=2)


if __name__ == "__main__":
    initial_geometries_path = (
        Path(__file__).parent.parent / "rdkit_uniques_100_molecules_42_seed"
    )
    reference_geometry_path = (
        Path(__file__).parent.parent / "uniques_100_molecules_42_seed"
    )

    with Pool(1) as p:

        p.starmap(
            create_objective,
            [
                (
                    path.stem,
                    path,
                    xyz_to_np(reference_geometry_path / f"{path.stem}.xyz")[0],
                )
                for path in initial_geometries_path.glob("*.xyz")
            ],
        )
