import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from rdkit.Chem import GetPeriodicTable


def xyz_to_np(file, ignore_hydrogen=False):
    with open(file, "r") as f:
        lines = f.readlines()
    lines = lines[2:]
    lines = [line.split() for line in lines]

    if ignore_hydrogen:
        lines = [line for line in lines if line[0] != "H"]

    ptable = GetPeriodicTable()

    atoms = [line[0] for line in lines]
    charges = [ptable.GetAtomicNumber(atom) for atom in atoms]
    xyz = [[float(line[1]), float(line[2]), float(line[3])] for line in lines]

    return np.array(xyz), atoms, charges


def xyz_to_df(file):
    with open(file, "r") as f:
        lines = f.readlines()
    lines = lines[2:]
    lines = [line.split() for line in lines]

    atoms = [line[0] for line in lines]
    xyz = [[float(line[1]), float(line[2]), float(line[3])] for line in lines]

    df = pd.DataFrame(xyz, columns=["x", "y", "z"], index=atoms)

    return df


def pairwise_distance(coords):
    dist = pdist(coords)
    dist = squareform(dist)
    return dist


def average_abs_dist(dist_df1, dist_df2):
    abs_dist_arr = abs(dist_df1 - dist_df2)
    avg_dist = abs_dist_arr[np.triu_indices_from(abs_dist_arr, k=1)].mean()
    return avg_dist


def read_energy_gradient(grad_file):
    with open(grad_file, "r") as f:
        lines = f.readlines()

    lines = lines[11:]
    lines = [line.strip() for line in lines]

    total_gradients = []
    x_gradients, y_gradients, z_gradients = [], [], []

    for i in range(0, len(lines), 3):
        try:
            x, y, z = lines[i : i + 3]
            x_gradients.append(float(x))
            y_gradients.append(float(y))
            z_gradients.append(float(z))
            total_gradients.append(
                np.linalg.norm(np.array([float(x), float(y), float(z)]))
            )

        except:
            break
    return x_gradients, y_gradients, z_gradients, total_gradients


def number_of_DFT_cycles(log):
    with open(log, "r") as f:
        raw_text = f.read()

    if "Could not converge geometry optimization" in raw_text:
        return -1

    lines = raw_text.split("\n")

    cycles = 0
    for line in lines:
        if "Step  " in line:
            cycles += 1

    return cycles
