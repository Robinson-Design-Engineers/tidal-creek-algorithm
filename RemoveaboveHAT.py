# created by Claude AI and reviewed by Sam Kraus 2024/09/15
import numpy as np

def detrend_2d(Z):
    """
    This function is the 2D equivalent of detrend function in MATLAB
    Z_f = detrend_2d(Z) removes the best plane fit trend from the
    data in the 2D array Z and returns the residual in the 2D array Z_f
    
    Based on MATLAB function by Munther Gdeisat - The General Engineering
    Research Institute (GERI) at Liverpool John Moores University.
    """
    if Z.ndim != 2:
        raise ValueError('Z must be a 2D array')
    
    M, N = Z.shape
    X, Y = np.meshgrid(np.arange(1, N+1), np.arange(1, M+1))
    
    # Make the 2D data as 1D vector
    Xcolv = X.flatten()
    Ycolv = Y.flatten()
    Zcolv = Z.flatten()
    
    # Remove NaN values
    mask = ~np.isnan(Zcolv)
    Xcolv = Xcolv[mask]
    Ycolv = Ycolv[mask]
    Zcolv = Zcolv[mask]
    
    # Vector of ones for constant term
    Const = np.ones_like(Xcolv)
    
    # Find the coefficients of the best plane fit
    A = np.column_stack((Xcolv, Ycolv, Const))
    Coefficients, _, _, _ = np.linalg.lstsq(A, Zcolv, rcond=None)
    
    XCoeff, YCoeff, CCoeff = Coefficients
    
    # Detrend the data
    Z_p = XCoeff * X + YCoeff * Y + CCoeff
    Z_f = Z - Z_p
    
    return Z_f, XCoeff, YCoeff, CCoeff

def RemoveaboveHAT(Z, Zs, HAT, detrendyn):
    # Remove elevation and slope data when elevation is above HAT
    Zs[Z > HAT] = np.nan
    meandepth = np.nanmean(Z)
    meanslope = np.nanmean(Zs)

    # Detrend elevation dataset
    if detrendyn == 1:
        Zold = Z.copy()
        Z, XCoeff, YCoeff, CCoeff = detrend_2d(Z)
        Z = Z + meandepth

    Z[Z > HAT] = np.nan

    return Z, Zs, meandepth, meanslope