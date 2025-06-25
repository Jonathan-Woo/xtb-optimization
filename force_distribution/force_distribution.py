import json
from multiprocessing import Pool
import os
from pathlib import Path
import shutil
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from experiments import rdkit_generate_geometries, execute_xtb_run
from processing_utils import read_energy_gradient
from setup_experiment import generate_xtb_parameter_file


if __name__ == "__main__":
    reference_xyz_path = (
        Path(__file__).parent.parent / "./uniques_100_molecules_42_seed"
    )
    base_xtb_parameter_path = Path(__file__).parent.parent / "param_gfn2-xtb.txt"
    output_dir = Path(__file__).parent
    optimized_xtb_parameters_path = Path(__file__).parent.parent / "BO"

    optimization_types = [
        "forces",
        "forces_wider_500",
        "forces_wider_complete",
        # "geometry",
        # "no_hydrogen",
    ]

    single_threaded = True
    optimize = False

    # with Pool(os.cpu_count() - 4) as p:
    #     results = []
    #     for molecule in tqdm(
    #         list(reference_xyz_path.glob("*.xyz")),
    #         desc="Molecules",
    #         leave=False,
    #     ):
    #         results.append(
    #             p.apply_async(
    #                 execute_xtb_run,
    #                 args=(
    #                     base_xtb_parameter_path,
    #                     reference_xyz_path / f"{molecule.stem}.xyz",
    #                     output_dir / "results" / "base" / molecule.stem,
    #                     single_threaded,
    #                     optimize,
    #                 ),
    #             )
    #         )

    #     for result in tqdm(results, desc="Results"):
    #         result.get()

    #     for optimization_type in optimization_types:
    #         results = []
    #         cur_optimization_path = (
    #             optimized_xtb_parameters_path / f"optimizations_{optimization_type}"
    #         )

    #         # if optimization_type == "forces":
    #         #     optimize = False
    #         # elif optimization_type == "geometry":
    #         #     optimize = True

    #         for molecule in tqdm(
    #             list(cur_optimization_path.iterdir()),
    #             desc="Molecules",
    #             leave=False,
    #         ):
    #             with open(
    #                 cur_optimization_path / molecule.stem / "BO_results.json",
    #                 "r",
    #             ) as f:
    #                 optimized_parameters = json.load(f)["best_params"]

    #             generate_xtb_parameter_file(
    #                 optimized_parameters,
    #                 output_dir
    #                 / "results"
    #                 / f"optimized_{optimization_type}"
    #                 / molecule.stem
    #                 / "xtb_parameters.txt",
    #             )

    #             results.append(
    #                 (
    #                     p.apply_async(
    #                         execute_xtb_run,
    #                         args=(
    #                             output_dir
    #                             / "results"
    #                             / f"optimized_{optimization_type}"
    #                             / molecule.stem
    #                             / "xtb_parameters.txt",
    #                             reference_xyz_path / f"{molecule.stem}.xyz",
    #                             output_dir
    #                             / "results"
    #                             / f"optimized_{optimization_type}"
    #                             / molecule.stem,
    #                             single_threaded,
    #                             optimize,
    #                         ),
    #                     ),
    #                     molecule.stem,
    #                 )
    #             )

    #         for result in tqdm(results, desc="Results"):
    #             try:
    #                 result[0].get()
    #             except Exception as e:
    #                 print(f"Error processing result: {e}")
    #                 for optimization_type in optimization_types:
    #                     shutil.rmtree(
    #                         output_dir
    #                         / "results"
    #                         / f"optimized_{optimization_type}"
    #                         / result[1],
    #                         ignore_errors=True,
    #                     )
    #                 shutil.rmtree(
    #                     output_dir / "results" / "base" / result[1], ignore_errors=True
    #                 )
    #     p.close()
    #     p.join()

    total_max_force_fig, total_max_force_ax = plt.subplots(figsize=(10, 6))
    total_max_force_fig_trunc, total_max_force_ax_trunc = plt.subplots(figsize=(10, 6))
    total_force_fig, total_force_ax = plt.subplots(figsize=(10, 6))
    x_force_fig, x_force_ax = plt.subplots(figsize=(10, 6))
    y_force_fig, y_force_ax = plt.subplots(figsize=(10, 6))
    z_force_fig, z_force_ax = plt.subplots(figsize=(10, 6))

    common_molecules = set(
        molecule.stem for molecule in (output_dir / "results" / "base").iterdir()
    )
    for optimization_type in optimization_types:
        common_molecules.intersection_update(
            set(
                molecule.stem
                for molecule in (
                    output_dir / "results" / f"optimized_{optimization_type}"
                ).iterdir()
            )
        )

    high_forces = {}

    metrics = {}
    gradients = {}
    for parameter_type in [
        "base",
        # "optimized_forces",
        # "optimized_forces_wider_500",
        # 'optimized_forces_wider_complete',
        "optimized_geometry",
        # "optimized_no_hydrogen",
    ]:
        total_max_gradients = []
        total_gradients = []
        x_gradients = []
        y_gradients = []
        z_gradients = []

        for molecule_name in common_molecules:
            molecule_path = output_dir / "results" / f"{parameter_type}" / molecule_name
            grad_file = molecule_path / f"{molecule_name}.engrad"

            cur_x_gradients, cur_y_gradients, cur_z_gradients, cur_total_gradients = (
                read_energy_gradient(grad_file)
            )
            x_gradients.extend(cur_x_gradients)
            y_gradients.extend(cur_y_gradients)
            z_gradients.extend(cur_z_gradients)
            total_gradients.extend(cur_total_gradients)
            total_max_gradients.append(max(cur_total_gradients))

            if max(cur_total_gradients) * 627.5 > 100:
                high_forces[molecule_name] = {
                    "max_force": max(cur_total_gradients).item() * 627.5,
                    "atom_index": np.argmax(cur_total_gradients).item(),
                }

        molecules = list(
            molecule.stem
            for molecule in (output_dir / "results" / f"{parameter_type}").iterdir()
        )

        total_max_gradients = np.array(total_max_gradients) * 627.5
        total_gradients = np.array(total_gradients) * 627.5
        x_gradients = np.array(x_gradients) * 627.5
        y_gradients = np.array(y_gradients) * 627.5
        z_gradients = np.array(z_gradients) * 627.5

        gradients[parameter_type] = {
            "total_max_gradients": total_max_gradients,
            "total_gradients": total_gradients,
            "x_gradients": x_gradients,
            "y_gradients": y_gradients,
            "z_gradients": z_gradients,
        }

    num_bins = 100
    shared_bins_max = np.linspace(
        np.min(
            np.array([grads["total_max_gradients"] for grads in gradients.values()])
        ),
        np.max(
            np.array([grads["total_max_gradients"] for grads in gradients.values()])
        ),
        num_bins,
    )
    shared_bins_max_trunc = np.linspace(0, 100, num_bins)
    shared_bins_total = np.linspace(
        np.min(np.array([grads["total_gradients"] for grads in gradients.values()])),
        np.max(np.array([grads["total_gradients"] for grads in gradients.values()])),
        num_bins,
    )
    shared_bins_x = np.linspace(
        np.min(np.array([grads["x_gradients"] for grads in gradients.values()])),
        np.max(np.array([grads["x_gradients"] for grads in gradients.values()])),
        num_bins,
    )
    shared_bins_y = np.linspace(
        np.min(np.array([grads["y_gradients"] for grads in gradients.values()])),
        np.max(np.array([grads["y_gradients"] for grads in gradients.values()])),
        num_bins,
    )
    shared_bins_z = np.linspace(
        np.min(np.array([grads["z_gradients"] for grads in gradients.values()])),
        np.max(np.array([grads["z_gradients"] for grads in gradients.values()])),
        num_bins,
    )

    parameter_name_map = {
        "base": "Base Parameters",
        "optimized_forces": "Optimized Forces",
        "optimized_forces_wider_500": "Optimized Forces Original",
        "optimized_forces_wider_complete": "Optimized Forces Wide",
        "optimized_geometry": "Optimized Displacement",
    }

    legend_titles = []

    for parameter_type, grads in gradients.items():
        _, _, patches = total_max_force_ax.hist(
            grads["total_max_gradients"],
            bins=shared_bins_max,
            alpha=0.5,
            label=f"{parameter_type} parameters",
        )

        # for grad in grads['total_max_gradients']:
        #     if grad > 100:
        #         total_max_force_ax.axvline(
        #             grad,
        #             linestyle="--",
        #             color="red",
        #             label=f"{parameter_type} outlier: {grad:.2f}",
        #         )

        color = patches[0].get_facecolor()

        _, _, patches = total_max_force_ax_trunc.hist(
            grads["total_max_gradients"],
            bins=shared_bins_max_trunc,
            alpha=0.5,
            label=f"{parameter_type} parameters",
        )

        total_force_ax.hist(
            grads["total_gradients"],
            bins=shared_bins_total,
            alpha=0.5,
            label=f"{parameter_type} parameters",
            color=color,
        )
        x_force_ax.hist(
            grads["x_gradients"],
            bins=shared_bins_x,
            alpha=0.5,
            label=f"{parameter_type} parameters",
            color=color,
        )
        y_force_ax.hist(
            grads["y_gradients"],
            bins=shared_bins_y,
            alpha=0.5,
            label=f"{parameter_type} parameters",
            color=color,
        )
        z_force_ax.hist(
            grads["z_gradients"],
            bins=shared_bins_z,
            alpha=0.5,
            label=f"{parameter_type} parameters",
            color=color,
        )

        total_max_force_ax.axvline(
            np.mean(grads["total_max_gradients"]),
            linestyle="--",
            label=f"{parameter_type} mean",
            color=color,
        )
        total_force_ax.axvline(
            np.mean(grads["total_gradients"]),
            linestyle="--",
            label=f"{parameter_type} mean",
            color=color,
        )
        x_force_ax.axvline(
            np.mean(grads["x_gradients"]),
            linestyle="--",
            label=f"{parameter_type} mean",
            color=color,
        )
        y_force_ax.axvline(
            np.mean(grads["y_gradients"]),
            linestyle="--",
            label=f"{parameter_type} mean",
            color=color,
        )
        z_force_ax.axvline(
            np.mean(grads["z_gradients"]),
            linestyle="--",
            label=f"{parameter_type} mean",
            color=color,
        )

        legend_titles.append(f"{parameter_name_map[parameter_type]}")
        legend_titles.append(
            f"{parameter_name_map[parameter_type]} mean = {np.mean(grads['total_max_gradients']):.2f}"
        )
    total_max_force_ax.set_title("Maximum Atomic Force Distribution")
    total_max_force_ax.set_xlabel("Maximum Atomic Force (kcal/mol/bohr)")
    total_max_force_ax.set_ylabel("Count")
    total_max_force_ax.legend(legend_titles)
    total_max_force_ax_trunc.set_title(
        "Maximum Atomic Force Distribution (0-100 kcal/mol/bohr)"
    )
    total_max_force_ax_trunc.set_xlabel("Maximum Atomic Force (kcal/mol/bohr)")
    total_max_force_ax_trunc.set_ylabel("Count")
    total_max_force_ax_trunc.legend(legend_titles)
    total_force_ax.set_title("Total Atomic Force Distribution")
    total_force_ax.set_xlabel("Total Atomic Force (kcal/mol/bohr)")
    total_force_ax.set_ylabel("Count")
    total_force_ax.legend(legend_titles)
    x_force_ax.set_title("X Force Distribution")
    x_force_ax.set_xlabel("X Force (kcal/mol/bohr)")
    x_force_ax.set_ylabel("Count")
    x_force_ax.legend(legend_titles)
    y_force_ax.set_title("Y Force Distribution")
    y_force_ax.set_xlabel("Y Force (kcal/mol/bohr)")
    y_force_ax.set_ylabel("Count")
    y_force_ax.legend(legend_titles)
    z_force_ax.set_title("Z Force Distribution")
    z_force_ax.set_xlabel("Z Force (kcal/mol/bohr)")
    z_force_ax.set_ylabel("Count")
    z_force_ax.legend(legend_titles)

    num_total_forces_below_1e4 = len([x for x in total_gradients if abs(x) < 1e-4])
    num_x_forces_below_1e4 = len([x for x in x_gradients if abs(x) < 1e-4])
    num_y_forces_below_1e4 = len([x for x in y_gradients if abs(x) < 1e-4])
    num_z_forces_below_1e4 = len([x for x in z_gradients if abs(x) < 1e-4])

    average_absolute_total_force = np.mean(np.abs(total_gradients))
    average_absolute_x_force = np.mean(np.abs(x_gradients))
    average_absolute_y_force = np.mean(np.abs(y_gradients))
    average_absolute_z_force = np.mean(np.abs(z_gradients))

    metrics[parameter_type] = {
        "total_forces_below_1e4": num_total_forces_below_1e4,
        "x_forces_below_1e4": num_x_forces_below_1e4,
        "y_forces_below_1e4": num_y_forces_below_1e4,
        "z_forces_below_1e4": num_z_forces_below_1e4,
        "average_absolute_total_force": average_absolute_total_force,
        "average_absolute_x_force": average_absolute_x_force,
        "average_absolute_y_force": average_absolute_y_force,
        "average_absolute_z_force": average_absolute_z_force,
    }

    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(output_dir / "force_distribution_metrics.csv")

    # save as svg
    total_max_force_ax.grid()
    total_max_force_ax_trunc.grid()
    total_force_ax.grid()
    x_force_ax.grid()
    y_force_ax.grid()
    z_force_ax.grid()

    total_max_force_fig.savefig(
        output_dir / "force_distribution_max_total.svg",
        dpi=300,
        bbox_inches="tight",
    )
    total_max_force_fig_trunc.savefig(
        output_dir / "force_distribution_max_total_trunc.svg",
        dpi=300,
        bbox_inches="tight",
    )
    total_force_fig.savefig(
        output_dir / "force_distribution_total.svg",
        dpi=300,
        bbox_inches="tight",
    )
    x_force_fig.savefig(
        output_dir / "force_distribution_x.svg",
        dpi=300,
        bbox_inches="tight",
    )
    y_force_fig.savefig(
        output_dir / "force_distribution_y.svg",
        dpi=300,
        bbox_inches="tight",
    )
    z_force_fig.savefig(
        output_dir / "force_distribution_z.svg",
        dpi=300,
        bbox_inches="tight",
    )

    with open(output_dir / "high_forces.json", "w") as f:
        json.dump(high_forces, f, indent=4)
