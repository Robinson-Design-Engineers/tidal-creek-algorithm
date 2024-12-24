import numpy as np #type:ignore
import matplotlib.pyplot as plt #type:ignore
from matplotlib.colors import ListedColormap #type:ignore
import pandas as pd #type:ignore
import h5py #type:ignore
from skimage import morphology #type:ignore
from skimage.morphology import dilation, disk # type: ignore
from scipy import ndimage

def process_creek_orders(STRAHLER, STRAIGHTDIST, Z, X, Y, creek_order, ID, shortname):
    # Calculate sinuosity
    SINUOSITY = np.divide(STRAHLER, STRAIGHTDIST, where=(STRAIGHTDIST!=0))
    
    # Remove rows and columns of zeros
    # Find indices where rows/columns are not all zeros
    rows_to_keep = np.any(STRAHLER, axis=1)
    cols_to_keep = np.any(STRAHLER, axis=0)
    
    # Filter arrays
    SINUOSITY = SINUOSITY[rows_to_keep][:, cols_to_keep]
    STRAIGHTDIST = STRAIGHTDIST[rows_to_keep][:, cols_to_keep]
    STRAHLER = STRAHLER[rows_to_keep][:, cols_to_keep]
    
    # Get number of segments (count non-zero elements in each row)
    SEGMENTS = np.sum(STRAHLER != 0, axis=1)
    
    # Create SINUOUSLENGTH with same shape as STRAHLER
    SINUOUSLENGTH = STRAHLER.copy()
    
    # Reverse ID array
    ID = np.flip(ID)
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Create contour mask
    contourmask = ~np.isnan(Z)
    # Fill holes
    contourmask = ndimage.binary_fill_holes(contourmask)
    
    # Morphological operations
    contourmask = morphology.binary_closing(contourmask)
    contourmask = morphology.thin(contourmask)  # equivalent to 'remove' in MATLAB
    # Note: 'diag' operation doesn't have direct equivalent in skimage
    contourmask = morphology.dilation(contourmask)  # equivalent to 'thicken'
    
    # Create discrete colormap: black + 5 colors
    colors = ['#000000', '#451caf', '#1878ff', '#00c2ba', '#c9c200', '#f7fd00']
    discrete_cmap = ListedColormap(colors)
    
    # Plot pcolor equivalent
    plt.pcolormesh(X, Y, creek_order, shading='auto', cmap=discrete_cmap)
    plt.clim(1, 7)
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Creek Order', fontsize=16, fontweight='bold')
    cbar.set_ticks([1, 2, 3, 4, 5, 6])
    
    # Set font properties
    plt.gca().tick_params(labelsize=16)
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontweight('bold')
    
    # Save creek mask to HDF5 format
    output_filename = f'OUTPUTS/{shortname}_creekorder.h5'
    with h5py.File(output_filename, 'w') as f:
        # Create a dataset in the HDF5 file
        f.create_dataset('creek_order', data=creek_order, compression='gzip', compression_opts=9)
        
        # Add metadata if needed
        f.attrs['creation_date'] = str(np.datetime64('now'))
        f.attrs['description'] = 'Creek order analysis results'
    
    return SINUOSITY, STRAIGHTDIST, STRAHLER, SEGMENTS, SINUOUSLENGTH