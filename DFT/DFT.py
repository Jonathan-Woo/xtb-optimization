import json
from multiprocessing.pool import Pool
import os
from pathlib import Path

from tqdm import tqdm
from experiments import execute_dft_run, execute_xtb_run
from setup_experiment import generate_dft_input_file, generate_xtb_parameter_file


if __name__ == "__main__":
    optimization_type = "forces_wider"
    BO_optimizations_path = (
        Path(__file__).parent.parent / "BO" / f"optimizations_{optimization_type}"
    )
    DFT_path = Path(__file__).parent / f"experiments_{optimization_type}"

    initial_geometry_path = (
        Path(__file__).parent.parent / "rdkit_uniques_100_molecules_42_seed"
    )

    with Pool(8) as p:
        procs = []
        for molecule in BO_optimizations_path.iterdir():
            if molecule.is_dir():
                optimized_parameters = next(molecule.rglob("BO_results.json"))
                molecule_name = molecule.name

                os.makedirs(DFT_path / molecule_name / "base" / "DFT", exist_ok=True)
                os.makedirs(DFT_path / molecule_name / "base" / "xtb", exist_ok=True)
                os.makedirs(
                    DFT_path / molecule_name / "optimized" / "DFT", exist_ok=True
                )
                os.makedirs(
                    DFT_path / molecule_name / "optimized" / "xtb", exist_ok=True
                )

                # make xtb parameter file
                with open(optimized_parameters, "r") as f:
                    optimized_parameters = json.load(f)

                optimized_xtb_parameter_path = (
                    DFT_path
                    / molecule_name
                    / "optimized"
                    / "xtb"
                    / "xtb_parameters.txt"
                )
                generate_xtb_parameter_file(
                    optimized_parameters["best_params"], optimized_xtb_parameter_path
                )
                base_xtb_parameter_path = (
                    DFT_path / molecule_name / "base" / "xtb" / "xtb_parameters.txt"
                )
                generate_xtb_parameter_file({}, base_xtb_parameter_path)

                procs.append(
                    p.apply_async(
                        execute_xtb_run,
                        args=(
                            optimized_xtb_parameter_path,
                            initial_geometry_path / f"{molecule_name}.xyz",
                            optimized_xtb_parameter_path.parent,
                        ),
                        kwds={
                            "single_threaded": True,
                            "optimize": True,
                        },
                    )
                )

                procs.append(
                    p.apply_async(
                        execute_xtb_run,
                        args=(
                            base_xtb_parameter_path,
                            initial_geometry_path / f"{molecule_name}.xyz",
                            base_xtb_parameter_path.parent,
                        ),
                        kwds={
                            "single_threaded": True,
                            "optimize": True,
                        },
                    )
                )
        [proc.wait() for proc in tqdm(procs, desc="Running xTB optimizations")]
        p.close()
        p.join()

    for molecule in DFT_path.iterdir():
        if molecule.is_dir():

            base_dft_input_file_path = molecule / "base" / "DFT" / "dft_parameters.in"
            optimized_dft_input_file_path = (
                molecule / "optimized" / "DFT" / "dft_parameters.in"
            )

            generate_dft_input_file(
                molecule / "base" / "xtb" / "xtbopt.xyz", base_dft_input_file_path
            )
            generate_dft_input_file(
                molecule / "optimized" / "xtb" / "xtbopt.xyz",
                optimized_dft_input_file_path,
            )

            print(
                f"Running DFT for {molecule.name} with base parameters at {base_dft_input_file_path}"
            )

            execute_dft_run(
                dft_input_file_path=base_dft_input_file_path,
                n_threads=os.cpu_count() - 4,
                output_dir=base_dft_input_file_path.parent,
            )

            print(
                f"Running DFT for {molecule.name} with optimized parameters at {optimized_dft_input_file_path}"
            )

            execute_dft_run(
                dft_input_file_path=optimized_dft_input_file_path,
                n_threads=os.cpu_count() - 4,
                output_dir=optimized_dft_input_file_path.parent,
            )
