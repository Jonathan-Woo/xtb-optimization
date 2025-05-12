import json
import os
from pathlib import Path
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem


def execute_xtb_run(
    xtb_parameter_file_path, molecule_geometry_path, output_dir=None, xtb_args=None
):
    molecule_name = molecule_geometry_path.stem
    if output_dir is None:
        output_dir = xtb_parameter_file_path.parent / molecule_name

    if output_dir.exists() and "xtbopt.xyz" in os.listdir(output_dir):
        print(f"Skipping {molecule_name} as it has already been processed.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_dir / "stdout.txt", "w") as stdout_file, open(
            output_dir / "stderr.txt", "w"
        ) as stderr_file:
            subprocess.run(
                [
                    "xtb",
                    molecule_geometry_path.resolve(),
                    "-v",
                    "--opt",
                    "--grad",
                    "--vparam",
                    xtb_parameter_file_path.resolve(),
                ]
                + (xtb_args if xtb_args else []),
                cwd=output_dir,
                stdout=stdout_file,
                stderr=stderr_file,
                timeout=60,
                check=True,
            )
    except Exception as e:
        raise RuntimeError(f"Error running xTB for {molecule_name}: {e}") from e


def rdkit_generate_geometries(molecules_path, name_to_smiles_path, output_path):
    with open(name_to_smiles_path, "r") as f:
        name_to_smiles = json.load(f)

    for molecule in molecules_path.glob("*.xyz"):
        smiles = name_to_smiles[molecule.stem]

        # 1. Read the molecule from a SMILES string
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # 2. Add hydrogens
        mol_h = Chem.AddHs(mol)

        # 3. Embed in 3D
        params = AllChem.ETKDGv3()

        params.randomSeed = 42
        try:
            res = AllChem.EmbedMolecule(mol_h, params)
            if res != 0:
                print("Warning: Embedding failed (code {})".format(res))
                continue
        except Exception as e:
            print("Warning: Embedding failed: {}".format(e))
            continue

        # 4. MMFF94 geometry optimization
        try:
            res = AllChem.MMFFOptimizeMolecule(mol_h)
            if res != 0:
                print("Warning: MMFF94 did not fully converge (code {})".format(res))
                continue
        except Exception as e:
            print("Warning: MMFF94 optimization failed: {}".format(e))
            continue

        # 5. Access the optimized coordinates
        conf = mol_h.GetConformer()
        coords = [list(conf.GetAtomPosition(i)) for i in range(mol_h.GetNumAtoms())]

        # 6. (Optional) Write out to an XYZ file
        with open(output_path / f"{molecule.stem}.xyz", "w") as f:
            f.write(f"{mol_h.GetNumAtoms()}\n")
            f.write(f"Optimized by MMFF94\n")
            for atom in mol_h.GetAtoms():
                idx = atom.GetIdx()
                x, y, z = conf.GetAtomPosition(idx)
                f.write(f"{atom.GetSymbol():2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")
