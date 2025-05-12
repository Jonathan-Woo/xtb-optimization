import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform


def xyz_to_np(file):
    with open(file, "r") as f:
        lines = f.readlines()
    lines = lines[2:]
    lines = [line.split() for line in lines]

    atoms = [line[0] for line in lines]
    xyz = [[float(line[1]), float(line[2]), float(line[3])] for line in lines]

    return np.array(xyz), atoms


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
