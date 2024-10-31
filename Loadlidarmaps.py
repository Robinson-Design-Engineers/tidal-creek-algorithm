# created by Claude AI and reviewed by Sam Kraus 2024/09/15
import numpy as np
import h5py
from scipy import stats

def Loadlidarmaps(elev, elevmeta, slope, slopemeta):
    # Load slope data
    with h5py.File(slope, 'r') as f:
        Xs = f['X'][:]
        Ys = f['Y'][:]
        Zs = f['Z'][:]

    # Load slope metadata
    with h5py.File(slopemeta, 'r') as f:
        xs = f['x'][:]
        ys = f['y'][:]
        gss = f['gs'][()]
        ncolss = f['ncols'][()]
        nrowss = f['nrows'][()]

    # Load elevation data
    with h5py.File(elev, 'r') as f:
        Z = f['Z'][:]

    # Load elevation metadata
    with h5py.File(elevmeta, 'r') as f:
        X = f['x'][:]
        Y = f['y'][:]
        gs = f['gs'][()]
        xc = f['xc'][()]
        yc = f['yc'][()]

    # Resize matrix: the slope dataset may be smaller than the elevation dataset
    sizeZc1 = Z.shape[1] - Zs.shape[1]
    sizeZc2 = Z.shape[0] - Zs.shape[0]

    while Z.shape[1] > Zs.shape[1] and sizeZc1 > 0:
        Z = Z[:, :-1]
        X = X[:, :-1]
        Y = Y[:, :-1]
        sizeZc1 -= 1

    while Z.shape[0] > Zs.shape[0] and sizeZc2 > 0:
        Z = Z[:-1, :]
        Y = Y[:-1, :]
        X = X[:-1, :]
        sizeZc2 -= 1

    return Xs, Ys, Zs, X, Y, Z, gs, xc, yc