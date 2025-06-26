from pathlib import Path
import os
from experiments import execute_xtb_run

if __name__ == '__main__':
    in_path = Path(__file__).parent / 'rdkit_uniques_100_molecules_42_seed'
    outpath = Path(__file__).parent / f"gfnff_{in_path.name}"

    os.makedirs(outpath, exist_ok=True)

    for molecule in in_path.glob('*.xyz'):
        execute_xtb_run(
            molecule_geometry_path=molecule,
            output_dir=outpath/molecule.stem,
            single_threaded=True,
            optimize=True,
            xtb_args=['--gfnff'],
        )

