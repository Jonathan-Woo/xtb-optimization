import os
import subprocess
from multiprocessing import Pool
from pathlib import Path

from tqdm.auto import tqdm


def run_experiment(xtb_parameter_file_path, molecule_path):
    molecule_name = molecule_path.stem
    output_dir = xtb_parameter_file_path.parent / molecule_name

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_dir / "stdout.txt", "w") as stdout_file, open(
            output_dir / "stderr.txt", "w"
        ) as stderr_file:
            subprocess.run(
                [
                    "xtb",
                    molecule_path.resolve(),
                    "-v",
                    "--vparam",
                    xtb_parameter_file_path.resolve(),
                ],
                cwd=output_dir,
                stdout=stdout_file,
                stderr=stderr_file,
                timeout=5,
            )
    except Exception as e:
        print(e)


if __name__ == "__main__":
    molecules_path = Path("./molecules")
    experiments_path = Path("./experiments")

    with Pool() as p:
        for xtb_parameter_file_path in tqdm(
            list(experiments_path.glob("*/*/*.txt")), desc="Parameters"
        ):
            results = []
            for molecule_path in tqdm(
                list(molecules_path.glob("*")), desc="Molecules", leave=False
            ):
                results.append(
                    p.apply_async(
                        run_experiment, args=(xtb_parameter_file_path, molecule_path)
                    )
                )

            for result in tqdm(results, desc="Results"):
                result.get()
