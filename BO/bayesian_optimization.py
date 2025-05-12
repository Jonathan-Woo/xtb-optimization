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
            print(f"xTB run failed: {e}")
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
        params = [
            trial.suggest_float("ksd", 1.6, 2.4),
            trial.suggest_float("kpd", 1.6, 2.4),
            trial.suggest_float("kp", 1.784, 2.676),
            trial.suggest_float("ks", 1.48, 2.22),
            trial.suggest_float("kexp", 1.2, 1.8),
        ]

        outpath = (
            Path(__file__).parent / "optimizations" / molecule / f"trial_{trial.number}"
        )

        parameters = {
            "ksd": params[0],
            "kpd": params[1],
            "kp": params[2],
            "ks": params[3],
            "kexp": params[4],
        }

        loss = loss_fn(
            parameters,
            outpath,
        )

        if trial.number % 10 == 0 and trial.number > 0:
            grad, norm_grad = gradient(parameters, outpath / "gradients")

            gradient_norms.append((trial.number, norm_grad))

            if norm_grad < 1e-3:
                study.stop()

            fig = vis.plot_optimization_history(study)
            fig.write_image(
                outpath.parent / "optimization_history.png",
                format="png",
                scale=2,
            )

            plt.figure()
            plt.scatter(
                [x[0] for x in gradient_norms],
                [x[1] for x in gradient_norms],
                label="Gradient Norm",
            )
            min_grads = [gradient_norms[0]]
            cur_min_grad = min_grads[0][1]
            for i in range(1, len(gradient_norms)):
                if gradient_norms[i][1] < cur_min_grad:
                    cur_min_grad = gradient_norms[i][1]
                    min_grads.append(gradient_norms[i])
                else:
                    min_grads.append((gradient_norms[i][0], min_grads[-1][1]))

            plt.plot(
                [x[0] for x in min_grads],
                [x[1] for x in min_grads],
                label="Minimum Gradient Norm",
                color="red",
            )
            plt.xlabel("Trial Number")
            plt.ylabel("Gradient Norm")
            plt.title("Gradient Norms Over Trials")
            plt.savefig(
                outpath.parent / "gradient_norms.png",
                format="png",
            )

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

    gradient_norms = []

    study.optimize(objective, n_trials=100)

    outpath = Path(__file__).parent / "optimizations" / molecule

    with open(outpath / "best_params.txt", "w") as f:
        f.write(str(study.best_params))


if __name__ == "__main__":
    initial_geometries_path = (
        Path(__file__).parent.parent / "rdkit_uniques_100_molecules_42_seed"
    )
    reference_geometry_path = (
        Path(__file__).parent.parent / "uniques_100_molecules_42_seed"
    )

    with Pool() as p:
        for intial_geometry_path in initial_geometries_path.glob("*.xyz"):
            molecule_name = intial_geometry_path.stem
            reference_geometry = xyz_to_np(
                reference_geometry_path / f"{molecule_name}.xyz"
            )[0]

            p.apply_async(
                create_objective,
                args=(molecule_name, intial_geometry_path, reference_geometry),
            )
        p.close()
        p.join()
