from multiprocessing.pool import Pool
from collections import defaultdict
import json
import os
from pathlib import Path
import pickle
import shutil
import numpy as np
from ase import Atoms
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchmetrics
import schnetpack as spk
import schnetpack.transform as trn
from pytorch_lightning.callbacks import Callback
from tqdm import tqdm
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import random_split
from ase.io import read

from experiments import execute_xtb_run
from processing_utils import read_energy_gradient
from setup_experiment import generate_xtb_parameter_file


class EpochProgressBar(Callback):
    def on_train_start(self, trainer, pl_module):
        self.pbar = tqdm(
            total=trainer.max_epochs, desc="Training Progress (epochs)", position=0
        )

    def on_train_epoch_end(self, trainer, pl_module):
        self.pbar.update(1)

    def on_train_end(self, trainer, pl_module):
        self.pbar.close()


torch.set_float32_matmul_precision("medium")


class XYZDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.all_atoms = [read(file_path) for file_path in file_paths]
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        atoms = self.all_atoms[idx]
        label = (
            self.labels[idx]
            if isinstance(self.labels, list)
            else self.labels[self.file_paths[idx].name]
        )
        return {"atoms": atoms, "label": torch.tensor(label, dtype=torch.float32)}


class XYZDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size=32, num_workers=0):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.converter = spk.interfaces.AtomsConverter(
            neighbor_list=trn.ASENeighborList(cutoff=5.0),
            device="cuda",  # or 'cpu'
            dtype=torch.float32,
        )

    def collate_fn(self, batch):
        atoms_list = [item["atoms"] for item in batch]
        labels = torch.stack([item["label"] for item in batch])
        inputs = self.converter(atoms_list)
        inputs["y"] = labels
        return inputs

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )


if __name__ == "__main__":
    random_seed = 42

    rdkit_path = Path(__file__).parent.parent / "rdkit_uniques_10000_molecules_42_seed"
    reference_path = Path(__file__).parent.parent / "uniques_10000_molecules_42_seed"
    BO_experiments_path = (
        Path(__file__).parent.parent / "BO" / "optimizations_forces_final"
    )
    molecules = []
    for molecule in BO_experiments_path.iterdir():
        if (molecule / "BO_results.json").exists():
            molecules.append(molecule)

    molecules = [molecule.stem for molecule in molecules]

    nested_dict = lambda: defaultdict(nested_dict)
    metrics = nested_dict()

    test_set_size = 200
    num_outer_folds = 5

    rng = np.random.default_rng(random_seed)

    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0),
        dtype=torch.float32,
        device="cuda",
    )

    cutoff = 5.0
    n_atom_basis = 128
    radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)

    for train_set_size in tqdm(
        [50, 100, 200, 400, 800, 1600, 3200, 6400], desc="Training set sizes"
    ):
        for fold in tqdm(range(num_outer_folds), desc="Outer folds", leave=False):
            test = rng.choice(molecules, size=test_set_size, replace=False)
            train = rng.choice(
                np.setdiff1d(molecules, test), size=train_set_size, replace=False
            )
            x_train = [rdkit_path / f"{molecule}.xyz" for molecule in train]
            x_test = [rdkit_path / f"{molecule}.xyz" for molecule in test]
            y_train = []
            for molecule in train:
                with open(BO_experiments_path / molecule / "BO_results.json", "r") as f:
                    optimized_params = json.load(f)["best_params"]
                y_train.append(
                    [
                        optimized_params["ks"],
                        optimized_params["kp"],
                        optimized_params["ksd"],
                        optimized_params["kpd"],
                        optimized_params["kexp"],
                    ]
                )
            y_test = []
            for molecule in test:
                with open(BO_experiments_path / molecule / "BO_results.json", "r") as f:
                    optimized_params = json.load(f)["best_params"]
                y_test.append(
                    [
                        optimized_params["ks"],
                        optimized_params["kp"],
                        optimized_params["ksd"],
                        optimized_params["kpd"],
                        optimized_params["kexp"],
                    ]
                )

            train_dataset = XYZDataset(
                file_paths=x_train,
                labels=y_train,
            )

            n_val = int(0.1 * len(train_dataset))
            n_train = len(train_dataset) - n_val

            train_dataset, val_dataset = random_split(train_dataset, [n_train, n_val])

            datamodule = XYZDataModule(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                batch_size=len(train_dataset),
            )

            for model_name in tqdm(["painn", "schnet"], desc="Models", leave=False):
                if model_name == "painn":
                    model = spk.representation.PaiNN(
                        n_atom_basis=n_atom_basis,
                        n_interactions=3,
                        radial_basis=radial_basis,
                        cutoff_fn=spk.nn.CosineCutoff(cutoff),
                    )
                elif model_name == "schnet":
                    model = spk.representation.SchNet(
                        n_atom_basis=n_atom_basis,
                        n_interactions=3,
                        radial_basis=radial_basis,
                        cutoff_fn=spk.nn.CosineCutoff(cutoff),
                    )

                output_key = "y"
                pred_layer = spk.atomistic.Atomwise(
                    n_in=n_atom_basis, output_key=output_key, n_out=5
                )

                nnpot = spk.model.NeuralNetworkPotential(
                    representation=model,
                    input_modules=[spk.atomistic.PairwiseDistances()],
                    output_modules=[pred_layer],
                )

                output_H = spk.task.ModelOutput(
                    name=output_key,
                    loss_fn=nn.MSELoss(),
                    loss_weight=1.0,
                    metrics={"MAE": torchmetrics.MeanAbsoluteError()},
                )

                task = spk.task.AtomisticTask(
                    model=nnpot,
                    outputs=[output_H],
                    optimizer_cls=torch.optim.AdamW,
                    optimizer_args={"lr": 1e-4},
                )

                save_dir = (
                    Path(__file__).parent
                    / "results"
                    / model_name
                    / f"train_size_{train_set_size}"
                    / f"fold_{fold}"
                )

                if not save_dir.exists():
                    print(
                        f"Training model {model_name} for train size {train_set_size} fold {fold}"
                    )
                    os.makedirs(save_dir, exist_ok=True)

                    logger = pl.loggers.TensorBoardLogger(save_dir=save_dir)
                    early_stop = EarlyStopping(
                        monitor="val_loss", patience=20, mode="min", verbose=False
                    )

                    callbacks = [
                        spk.train.ModelCheckpoint(
                            model_path=os.path.join(save_dir, "best_model"),
                            save_top_k=1,
                            monitor="val_loss",
                        ),
                        EpochProgressBar(),
                        early_stop,
                    ]

                    trainer = pl.Trainer(
                        max_epochs=2000,
                        accelerator="gpu",
                        devices=1,
                        callbacks=callbacks,
                        logger=logger,
                        default_root_dir=save_dir,
                        enable_progress_bar=False,
                    )

                    trainer.fit(
                        task,
                        datamodule=datamodule,
                    )
                else:
                    print(f"Loading existing model from {save_dir}")

                best_model = torch.load(os.path.join(save_dir, "best_model"))

                predictions = []
                for mol_path in x_test:
                    atoms = read(mol_path)
                    inputs = converter(atoms)
                    with torch.no_grad():
                        pred = best_model(inputs)
                    predictions.append(pred[output_key])

                predictions = torch.stack(predictions).cpu().numpy().squeeze()
                y_test_np = np.array(y_test)
                mae = np.mean(np.abs(predictions - y_test_np), axis=0)
                mae = [float(value) for value in mae]

                metrics[train_set_size][model_name][fold] = dict(
                    zip(["ks", "kp", "ksd", "kpd", "kexp"], mae)
                )

                for parameter, value in zip(["ks", "kp", "ksd", "kpd", "kexp"], mae):
                    metrics[train_set_size][model_name][fold][parameter] = value

                xtb_outdir = Path(__file__).parent / "xtb" / str(train_set_size)
                shutil.rmtree(xtb_outdir, ignore_errors=True)
                with Pool(os.cpu_count() - 4) as p:
                    for molecule, infer_params in zip(test, predictions):
                        params_dict = {
                            "ksd": infer_params[0],
                            "kpd": infer_params[1],
                            "kp": infer_params[2],
                            "ks": infer_params[3],
                            "kexp": infer_params[4],
                        }
                        generate_xtb_parameter_file(
                            params_dict, xtb_outdir / molecule / "xtb_params.txt"
                        )

                        p.apply_async(
                            execute_xtb_run,
                            args=(
                                reference_path / f"{molecule}.xyz",
                                xtb_outdir / molecule,
                                True,
                                False,
                            ),
                            kwds={
                                "xtb_parameters_file_path": xtb_outdir
                                / molecule
                                / "xtb_params.txt",
                            },
                        )
                    p.close()
                    p.join()

                max_atomic_forces = []
                max_atomic_forces_label = []
                for molecule in test:
                    gradient_path = xtb_outdir / molecule / f"{molecule}.engrad"
                    if not gradient_path.exists():
                        continue
                    _, _, _, energy_gradient = read_energy_gradient(gradient_path)
                    max_atomic_forces.append(np.max(np.abs(energy_gradient)))

                    with open(
                        BO_experiments_path / molecule / "BO_results.json", "r"
                    ) as f:
                        bo_results = json.load(f)
                    max_atomic_forces_label.append(bo_results["best_value"])

                metrics[train_set_size][model_name][fold]["max_atomic_forces"] = (
                    np.mean(max_atomic_forces).tolist()
                )

                metrics[train_set_size][model_name][fold]["max_atomic_forces_label"] = (
                    np.mean(max_atomic_forces_label).tolist()
                )

    with open(Path(__file__).parent / "gnn_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
