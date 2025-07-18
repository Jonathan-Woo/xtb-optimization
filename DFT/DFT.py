import json
from multiprocessing.pool import Pool
import os
from pathlib import Path

from tqdm import tqdm
from experiments import execute_dft_run, execute_xtb_run
from setup_experiment import generate_dft_input_file, generate_xtb_parameter_file


if __name__ == "__main__":
    optimization_type = "forces_wider_complete"
    BO_optimizations_path = (
        Path(__file__).parent.parent / "BO" / f"optimizations_{optimization_type}"
    )
    DFT_path = Path(__file__).parent / f"experiments_{optimization_type}"

    rdkit_geometry_path = (
        Path(__file__).parent.parent / "rdkit_uniques_100_molecules_42_seed"
    )
    dft_geometry_path = Path(__file__).parent.parent / "uniques_100_molecules_42_seed"

    with Pool(os.cpu_count() - 4) as p:
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

                os.makedirs(DFT_path / molecule_name / "base_ff" / "DFT", exist_ok=True)
                os.makedirs(DFT_path / molecule_name / "base_ff" / "ff", exist_ok=True)
                os.makedirs(DFT_path / molecule_name / "base_ff" / "xtb", exist_ok=True)

                os.makedirs(
                    DFT_path / molecule_name / "optimized_ff" / "DFT", exist_ok=True
                )
                os.makedirs(
                    DFT_path / molecule_name / "optimized_ff" / "ff", exist_ok=True
                )
                os.makedirs(
                    DFT_path / molecule_name / "optimized_ff" / "xtb", exist_ok=True
                )

                os.makedirs(
                    DFT_path / molecule_name / "base_reference" / "DFT", exist_ok=True
                )
                os.makedirs(
                    DFT_path / molecule_name / "optimized_reference" / "DFT",
                    exist_ok=True,
                )

        procs = []
        for molecule in DFT_path.iterdir():
            if molecule.is_dir():
                molecule_name = molecule.name

                procs.append(
                    p.apply_async(
                        execute_xtb_run,
                        args=(
                            rdkit_geometry_path / f"{molecule_name}.xyz",
                            DFT_path / molecule_name / "base_ff" / "ff",
                        ),
                        kwds={
                            "single_threaded": True,
                            "optimize": True,
                            "xtb_args": ["--gfnff"],
                        },
                    )
                )

                procs.append(
                    p.apply_async(
                        execute_xtb_run,
                        args=(
                            rdkit_geometry_path / f"{molecule_name}.xyz",
                            DFT_path / molecule_name / "optimized_ff" / "ff",
                        ),
                        kwds={
                            "single_threaded": True,
                            "optimize": True,
                            "xtb_args": ["--gfnff"],
                        },
                    )
                )

        [proc.wait() for proc in tqdm(procs, desc="Running GFN-FF optimizations")]

        procs = []
        for molecule in BO_optimizations_path.iterdir():
            if molecule.is_dir():
                optimized_parameters = next(molecule.rglob("BO_results.json"))
                molecule_name = molecule.name
                with open(optimized_parameters, "r") as f:
                    optimized_parameters = json.load(f)

                base_xtb_parameter_path = (
                    DFT_path / molecule_name / "base" / "xtb" / "xtb_parameters.txt"
                )
                generate_xtb_parameter_file({}, base_xtb_parameter_path)

                base_ff_xtb_parameter_path = (
                    DFT_path / molecule_name / "base_ff" / "xtb" / "xtb_parameters.txt"
                )
                generate_xtb_parameter_file({}, base_ff_xtb_parameter_path)

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

                optimized_ff_xtb_parameter_path = (
                    DFT_path
                    / molecule_name
                    / "optimized_ff"
                    / "xtb"
                    / "xtb_parameters.txt"
                )
                generate_xtb_parameter_file(
                    optimized_parameters["best_params"], optimized_ff_xtb_parameter_path
                )

                base_reference_xtb_parameter_path = (
                    DFT_path
                    / molecule_name
                    / "base_reference"
                    / "xtb"
                    / "xtb_parameters.txt"
                )
                generate_xtb_parameter_file({}, base_reference_xtb_parameter_path)

                optimized_reference_xtb_parameter_path = (
                    DFT_path
                    / molecule_name
                    / "optimized_reference"
                    / "xtb"
                    / "xtb_parameters.txt"
                )
                generate_xtb_parameter_file(
                    optimized_parameters["best_params"],
                    optimized_reference_xtb_parameter_path,
                )

                procs.append(
                    p.apply_async(
                        execute_xtb_run,
                        args=(
                            rdkit_geometry_path / f"{molecule_name}.xyz",
                            base_xtb_parameter_path.parent,
                        ),
                        kwds={
                            "single_threaded": True,
                            "optimize": True,
                            "xtb_parameters_file_path": base_xtb_parameter_path,
                        },
                    )
                )
                procs.append(
                    p.apply_async(
                        execute_xtb_run,
                        args=(
                            DFT_path / molecule_name / "base_ff" / "ff" / "xtbopt.xyz",
                            base_ff_xtb_parameter_path.parent,
                        ),
                        kwds={
                            "single_threaded": True,
                            "optimize": True,
                            "xtb_parameters_file_path": base_ff_xtb_parameter_path,
                        },
                    )
                )

                procs.append(
                    p.apply_async(
                        execute_xtb_run,
                        args=(
                            rdkit_geometry_path / f"{molecule_name}.xyz",
                            optimized_xtb_parameter_path.parent,
                        ),
                        kwds={
                            "single_threaded": True,
                            "optimize": True,
                            "xtb_parameters_file_path": optimized_xtb_parameter_path,
                        },
                    )
                )
                procs.append(
                    p.apply_async(
                        execute_xtb_run,
                        args=(
                            DFT_path
                            / molecule_name
                            / "optimized_ff"
                            / "ff"
                            / "xtbopt.xyz",
                            optimized_ff_xtb_parameter_path.parent,
                        ),
                        kwds={
                            "single_threaded": True,
                            "optimize": True,
                            "xtb_parameters_file_path": optimized_ff_xtb_parameter_path,
                        },
                    )
                )

                procs.append(
                    p.apply_async(
                        execute_xtb_run,
                        args=(
                            dft_geometry_path / f"{molecule_name}.xyz",
                            base_reference_xtb_parameter_path.parent,
                        ),
                        kwds={
                            "single_threaded": True,
                            "optimize": True,
                            "xtb_parameters_file_path": base_reference_xtb_parameter_path,
                        },
                    )
                )
                procs.append(
                    p.apply_async(
                        execute_xtb_run,
                        args=(
                            dft_geometry_path / f"{molecule_name}.xyz",
                            optimized_reference_xtb_parameter_path.parent,
                        ),
                        kwds={
                            "single_threaded": True,
                            "optimize": True,
                            "xtb_parameters_file_path": optimized_reference_xtb_parameter_path,
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
            base_ff_dft_input_file_path = (
                molecule / "base_ff" / "DFT" / "dft_parameters.in"
            )
            optimized_ff_dft_input_file_path = (
                molecule / "optimized_ff" / "DFT" / "dft_parameters.in"
            )
            base_reference_dft_input_file_path = (
                molecule / "base_reference" / "DFT" / "dft_parameters.in"
            )
            optimized_reference_dft_input_file_path = (
                molecule / "optimized_reference" / "DFT" / "dft_parameters.in"
            )

            if (molecule / "base" / "xtb" / "xtbopt.xyz").exists():
                generate_dft_input_file(
                    molecule / "base" / "xtb" / "xtbopt.xyz", base_dft_input_file_path
                )

                print(
                    f"Running DFT for {molecule.name} with base parameters at {base_dft_input_file_path}"
                )
                execute_dft_run(
                    dft_input_file_path=base_dft_input_file_path,
                    n_threads=os.cpu_count() - 4,
                    output_dir=base_dft_input_file_path.parent,
                )

            if (molecule / "optimized" / "xtb" / "xtbopt.xyz").exists():
                generate_dft_input_file(
                    molecule / "optimized" / "xtb" / "xtbopt.xyz",
                    optimized_dft_input_file_path,
                )

                print(
                    f"Running DFT for {molecule.name} with optimized parameters at {optimized_dft_input_file_path}"
                )
                execute_dft_run(
                    dft_input_file_path=optimized_dft_input_file_path,
                    n_threads=os.cpu_count() - 4,
                    output_dir=optimized_dft_input_file_path.parent,
                )

            if (molecule / "base_ff" / "xtb" / "xtbopt.xyz").exists():
                generate_dft_input_file(
                    molecule / "base_ff" / "xtb" / "xtbopt.xyz",
                    base_ff_dft_input_file_path,
                )

                print(
                    f"Running DFT for {molecule.name} with base parameters + GFN-FF at {base_ff_dft_input_file_path}"
                )
                execute_dft_run(
                    dft_input_file_path=base_ff_dft_input_file_path,
                    n_threads=os.cpu_count() - 4,
                    output_dir=base_ff_dft_input_file_path.parent,
                )

            if (molecule / "optimized_ff" / "xtb" / "xtbopt.xyz").exists():
                generate_dft_input_file(
                    molecule / "optimized_ff" / "xtb" / "xtbopt.xyz",
                    optimized_ff_dft_input_file_path,
                )

                print(
                    f"Running DFT for {molecule.name} with optimized parameters + GFN-FF at {optimized_ff_dft_input_file_path}"
                )
                execute_dft_run(
                    dft_input_file_path=optimized_ff_dft_input_file_path,
                    n_threads=os.cpu_count() - 4,
                    output_dir=optimized_ff_dft_input_file_path.parent,
                )

            if (molecule / "base_reference" / "xtb" / "xtbopt.xyz").exists():
                generate_dft_input_file(
                    molecule / "base_reference" / "xtb" / "xtbopt.xyz",
                    base_reference_dft_input_file_path,
                )

                print(
                    f"Running DFT for {molecule.name} with base reference parameters at {base_reference_dft_input_file_path}"
                )
                execute_dft_run(
                    dft_input_file_path=base_reference_dft_input_file_path,
                    n_threads=os.cpu_count() - 4,
                    output_dir=base_reference_dft_input_file_path.parent,
                )

            if (molecule / "optimized_reference" / "xtb" / "xtbopt.xyz").exists():
                generate_dft_input_file(
                    molecule / "optimized_reference" / "xtb" / "xtbopt.xyz",
                    optimized_reference_dft_input_file_path,
                )

                print(
                    f"Running DFT for {molecule.name} with optimized reference parameters at {optimized_reference_dft_input_file_path}"
                )
                execute_dft_run(
                    dft_input_file_path=optimized_reference_dft_input_file_path,
                    n_threads=os.cpu_count() - 4,
                    output_dir=optimized_reference_dft_input_file_path.parent,
                )
