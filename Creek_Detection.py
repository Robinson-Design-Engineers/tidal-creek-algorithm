# written by Claude AI, reviewed by Sam Kraus 2024/09/15
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for displaying plots
import matplotlib.pyplot as plt
import json
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
import h5py


# Enable interactive mode
plt.ion()

# Set the DPI and figure size
plt.rcParams['figure.dpi'] = 100  # Increase DPI for retina display
# plt.rcParams['figure.figsize'] = (12, 10)  # Specify the figure size

# Set global font size
import matplotlib as mpl
mpl.rcParams['font.size'] = 10  # Change to your preferred font size
# mpl.rcParams['axes.titlesize'] = 2  # Title size for axes
# mpl.rcParams['axes.labelsize'] = 2  # Axis label size
# mpl.rcParams['xtick.labelsize'] = 2  # X tick label size
# mpl.rcParams['ytick.labelsize'] = 2  # Y tick label size
# mpl.rcParams['legend.fontsize'] = 2  # Legend font size

# Set global marker size and line width
msize = 6
plt.rcParams['lines.markersize'] = msize  # Default marker size
plt.rcParams['lines.linewidth'] = 0.5  # Default line width



# Load the variables from the JSON file
with open("variables_Creek_Detection.json", "r") as f:
    variables = json.load(f)

threshold = variables["threshold"]
Zs = variables["Zs"]
Zs = np.array(Zs) # make sure is an array
Z = variables["Z"]
gs = variables["gs"]
X = variables["X"]
Y = variables["Y"]
X2 = variables["X2"]
Y2 = variables["Y2"]
Z2 = variables["Z2"]
LZtharea = variables["LZtharea"]
LZth = variables["LZth"]
HZtharea = variables["HZtharea"]
HZth = variables["HZth"]
Ctharea = variables["Ctharea"]
Cth = variables["Cth"]

# # Now use these variables in your script
# # Example usage
# print(f"Zs: {Zs}, Z: {Z}, gs: {gs}, Cth: {Cth}")


def creek_detection(Zs, Z, gs, Cth, HZth, LZth):
    """
    This function gets the raw creek mask from predefined elevation and
    slope thresholds.
    """
    # Define a slope threshold Cth
    Zs = np.abs(Zs)
    binranges = np.linspace(np.nanmin(Zs), np.nanmax(Zs), 200)
    # bincounts, _ = np.histogram(Zs, bins=binranges) - from old Claude AI, gave 1 less values than binranges
    # Use np.digitize to mimic MATLAB's histc behavior
    bincounts = np.zeros(len(binranges))
    inds = np.digitize(Zs, binranges, right=True)
    for i in range(1, len(binranges)):
        bincounts[i-1] = np.sum(inds == i)
    PixArea = bincounts * gs**2
    CumAreas = np.cumsum(PixArea)

    # Extract the outercreek by removing all low slope pixels
    outercreek = Z.copy()
    # Convert the outercreek list to a NumPy array
    outercreek = np.array(outercreek)
    outercreek = outercreek.astype(float)
    outercreek[Zs < Cth] = np.nan
    outercreekind = np.where(~np.isnan(outercreek))

    # Define the upper and lower elevation thresholds HZth and LZth
    # # Check if np.max(Z) is greater than 0 before using np.arange
    # print('Max of z = ', np.max(Z))
    # if np.max(Z) > 0:
    #     binrange = np.arange(0.030, 0.030 + np.max(Z), 0.030)
    # else:
    #     raise ValueError("np.max(Z) must be greater than 0 for a valid range.")

    binrange = np.arange(0.030, 0.030 + np.nanmax(Z), 0.030)
    # bincount, _ = np.histogram(Z, bins=binrange) - from old Claude AI, gave 1 less values than binranges
    # Use np.digitize to mimic MATLAB's histc behavior
    bincount = np.zeros(len(binrange))
    ind = np.digitize(Z, binrange, right=True)
    for i in range(1, len(binrange)):
        bincount[i-1] = np.sum(ind == i)
    PixArea = bincount * gs**2
    CumArea = np.cumsum(PixArea)

    # Remove the points of high elevation from the outercreek
    outercreek[outercreek >= HZth] = np.nan
    # Create a innercreek matrix containing all values below LZth. Set all
    # the values from outercreek to NaN so values don't appear twice
    innercreek = Z.copy()
    # Ensure innercreek is a numpy array before performing element-wise comparison
    innercreek = np.array(innercreek)
    noinnercreek_ind = np.where(innercreek > LZth)
    innercreek[noinnercreek_ind] = np.nan
    innercreek[outercreekind] = np.nan

    # Combine the thresholds to get the creek map = sum of innercreek and outercreek
    innercreek = np.nan_to_num(innercreek)
    outercreek = np.nan_to_num(outercreek)
    creek = innercreek + outercreek
    creek[creek == 0] = np.nan

    return creek, CumArea, binrange, CumAreas, binranges, innercreek, outercreek

def process_detect_threshold(CumArea, binrange, titlefig, ylabtxt):
    """
    Manually selects a point in the hypsometry curve and stores it as a
    slope or elevation creek detection threshold.
    """
    plt.ion()
    plt.figure(figsize=(12,10))
    plt.plot(CumArea, binrange, marker='.')
    plt.title(titlefig)
    plt.xlabel('Cumulative Area (km^2)')
    plt.ylabel(ylabtxt)
    plt.grid(True)

    print("Please click on the plot to select the threshold.")
    point = plt.ginput(1)[0]
    Acreeksel, Zcreeksel = point

    # Find the closest point on hypsometry curve
    testdif = np.abs(CumArea - Acreeksel)
    Thind = np.argmin(testdif)

    # Find threshold value and corresponding cumulative area
    Tharea = CumArea[Thind]
    Th = binrange[Thind]

    plt.plot(Tharea, Th, marker='+', color='r')
    plt.show()

    return Thind, Th, Tharea

def creek_detection_manual(Zs, Z, gs):
    """
    This function finds the raw creek mask after having manually selected the
    elevation and slope thresholds.
    """
    # Define a slope threshold Cth
    Zs = np.abs(Zs)
    binranges = np.linspace(np.nanmin(Zs), np.nanmax(Zs), 200)
    # bincounts, _ = np.histogram(Zs, bins=binranges) - from old Claude AI, gave 199 values though
    # Use np.digitize to mimic MATLAB's histc behavior
    bincounts = np.zeros(len(binranges))
    inds = np.digitize(Zs, binranges, right=True)
    for i in range(1, len(binranges)):
        bincounts[i-1] = np.sum(inds == i)
    PixArea = bincounts * gs**2
    CumAreas = np.cumsum(PixArea)


    # Calculate the distribution function of the slope
    ylabtxt = 'Slope (degree)'
    titlefig = 'Select the slope threshold Sth'
    Cthind, Cth, Ctharea = process_detect_threshold(CumAreas, binranges, titlefig, ylabtxt)

    # Extract the outercreek by removing all low slope pixels
    outercreek = Z.copy()
    # Convert the outercreek list to a NumPy array
    outercreek = np.array(outercreek)
    outercreek = outercreek.astype(float)
    # Make sure Zs and outercreek are of compatible shape
    assert Zs.shape == outercreek.shape, "Zs and outercreek must have the same shape."

    # Set all values in outercreek where Zs is below the threshold to NaN
    outercreek[Zs < Cth] = np.nan
    outercreekind = np.where(~np.isnan(outercreek))

    # Define the upper and lower elevation thresholds HZth and LZth
    ylabtxt = 'Elevation (m)'
    titlefig = 'Select the higher elevation threshold HZth'
    binrange = np.arange(0.030, 0.030 + np.nanmax(Z), 0.030)
    # bincount, _ = np.histogram(Z, bins=binrange)
    # Use np.digitize to mimic MATLAB's histc behavior
    bincount = np.zeros(len(binrange))
    ind = np.digitize(Z, binrange, right=True)
    for i in range(1, len(binrange)):
        bincount[i-1] = np.sum(ind == i)
    PixArea = bincount * gs**2
    CumArea = np.cumsum(PixArea)
    HZthind, HZth, HZtharea = process_detect_threshold(CumArea, binrange, titlefig, ylabtxt)

    # Remove the points of high elevation from the outercreek
    outercreek[outercreek >= HZth] = np.nan

    titlefig = 'Select the lower elevation threshold LZth'
    LZthind, LZth, LZtharea = process_detect_threshold(CumArea, binrange, titlefig, ylabtxt)
    innercreek = Z.copy()
    # Ensure innercreek is a numpy array before performing element-wise comparison
    innercreek = np.array(innercreek)
    noinnercreek_ind = np.where(innercreek > LZth)
    innercreek[noinnercreek_ind] = np.nan
    innercreek[outercreekind] = np.nan

    # Combine the thresholds to get the creek map = sum of innercreek and outercreek
    innercreek = np.nan_to_num(innercreek)
    outercreek = np.nan_to_num(outercreek)
    creek = innercreek + outercreek
    creek[creek == 0] = np.nan

    return creek, CumArea, binrange, CumAreas, binranges, HZth, LZth, Cth, HZtharea, LZtharea, Ctharea

def figure_creek_detection(X, Y, creek, X2, Y2, Z2, CumArea, binrange, CumAreas, binranges, LZtharea, LZth, HZtharea, HZth, Ctharea, Cth):
    """
    This function creates a figure showing the raw creek mask as calculated by
    RawCreekMask or RawCreekMaskManual
    """
    plt.ion()

    # Create custom colormap like in MATLAB
    autumn_colors = plt.cm.autumn(np.linspace(0, 1, 292))[::-1]  # Reversed autumn
    winter_colors = plt.cm.winter(np.linspace(0, 1, 293))
    white_colors = np.array([[1, 1, 1]] * 15)
    mycolormap = np.vstack((winter_colors, autumn_colors))
    cm = LinearSegmentedColormap.from_list('custom', mycolormap)


    # ChatGPT 2D 10/10/24:
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))

    # Main plot
    # Plot the 2D plane (equivalent of pcolor and shading interp in MATLAB)
    p = ax.pcolormesh(X2, Y2, Z2, cmap=cm, shading='gouraud')  # 'shading'='gouraud' for interpolation
    # Add a colorbar and set its label
    cbar = plt.colorbar(p, ax=ax, pad=0.03)
    # cbar.set_label('Elevation (ft)', fontsize=14) # Anna made it ft not m?
    cbar.set_label('Elevation (m)', fontsize=14)
    # Plot the surface (creek), equivalent to MATLAB's surf command
    ax.contour(X2, Y2, creek, colors='black', linewidths=0.5, alpha=0.5)
    # Mesh with black gridlines
    mask = ~np.isnan(creek)
    ax.pcolormesh(X, Y, mask, cmap='binary', alpha=0.3, edgecolors='none', linewidths=0.5)

    # Customize the axis labels and formatting
    ax.set_xlabel('Distance (m)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Distance (m)', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.subplots_adjust(right=0.75)  # plot will take up X% of the total figure width

    # Hypsometry curve
    ax1 = fig.add_axes([0.8, 0.15, 0.18, 0.22]) # The argument passed is a list: [left, bottom, width, height]
    ax1.plot(CumArea, binrange, marker='.')
    ax1.plot(LZtharea, LZth, marker='x', markersize=msize*1.3, color='r')
    ax1.plot(HZtharea, HZth, marker='x', markersize=msize*1.3, color='r')
    ax1.set_xlabel('Cumulative Area (sqm)')
    ax1.set_ylabel('Elevation (m)')

    # Slope curve
    ax2 = fig.add_axes([0.8, 0.65, 0.18, 0.22])
    ax2.plot(CumAreas, binranges, marker='.')
    ax2.plot(Ctharea, Cth, marker='o', markersize=msize*1.3, markerfacecolor='none', markeredgecolor='blue')
    ax2.set_xlabel('Cumulative Area (sqm)')
    ax2.set_ylabel('Slope (degree)')

    # plt.tight_layout()
    plt.show()


    # ChatGPT 3D 10/10/24:
    # Create a figure and 3D axis
    fig = plt.figure(figsize=(12, 10))
    
    # Main plot
    ax = fig.add_subplot(111, projection='3d')
    # Plot the 2D plane with color interpolation
    # Mask the NaN values in Z2 for color mapping
    Z2_masked = np.ma.masked_invalid(Z2)
    # Create a flat Z value (can be at any constant height, e.g., Z=0)
    Z_plane = np.zeros_like(Z2_masked)
    # Plot the surface with the custom colormap and use Z2 values for color
    p = ax.plot_surface(X2, Y2, Z_plane, facecolors=cm(Z2), rstride=5, cstride=5, edgecolor='none', alpha=0.8)
    # Plot the 3D surface for the creek with mesh gridlines in black
    surf = ax.plot_surface(X2, Y2, creek, cmap='winter', edgecolor='black', alpha=0.6)
    # Add color bar for the 2D plane
    # Create a ScalarMappable for the colorbar based on the custom colormap
    sm = ScalarMappable(cmap=cm)
    sm.set_array(Z2_masked)  # Associate the data (Z2_masked) with the colorbar
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Elevation (ft)')
    # Customize the axis labels and formatting
    ax.set_xlabel('Distance (m)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Distance (m)', fontsize=16, fontweight='bold')
    ax.set_zlabel('Creek Elevation (m)', fontsize=16, fontweight='bold')
    # Set the viewing angle (camera position) to look directly down on the 2D plane
    ax.view_init(elev=90, azim=-90)
    # Set the plot's font size and mesh gridline color for the creek
    ax.tick_params(axis='both', which='major', labelsize=14)
    # Set Z-axis ticks
    ax.set_zticks(np.arange(np.floor(np.nanmin(creek)*2)/2, np.ceil(np.nanmax(creek)*2/2)+0.5, 0.5))
    # Set the aspect ratio
    ax.set_box_aspect([1, 1, 0.2])  # Keep Z aspect small to flatten
    # turn off gridlines
    ax.grid(False)

    # Hypsometry curve
    ax1 = fig.add_axes([0.65, 0.1, 0.18, 0.22]) # The argument passed is a list: [left, bottom, width, height]
    ax1.plot(CumArea, binrange, marker='.')
    ax1.plot(LZtharea, LZth, marker='x', markersize=msize*1.3, color='r')
    ax1.plot(HZtharea, HZth, marker='x', markersize=msize*1.3, color='r')
    ax1.set_xlabel('Cumulative Area (sqm)')
    ax1.set_ylabel('Elevation (m)')
    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_label_position('top')

    # Slope curve
    ax2 = fig.add_axes([0.05, 0.7, 0.15, 0.22])
    ax2.plot(CumAreas, binranges, marker='.')
    ax2.plot(Ctharea, Cth, marker='o', markersize=msize*1.3, markerfacecolor='none', markeredgecolor='blue')
    ax2.set_xlabel('Cumulative Area (sqm)')
    ax2.set_ylabel('Slope (degree)')
    ax2.xaxis.set_ticks_position('top')
    ax2.xaxis.set_label_position('top')

    plt.tight_layout()
    plt.show()

def figure_raw_creek_mask(creekmask):
    """
    Show raw creek area mask figure
    """
    creekmaskpic = creekmask.copy()
    creekmaskpic[creekmask == 1] = 0
    creekmaskpic[creekmask == 0] = 1

    plt.ion()
    plt.figure(figsize=(4,5))
    plt.imshow(creekmaskpic, cmap='gray')
    plt.title('Raw creek area mask')
    plt.axis('off')
    # The commented out lines in the MATLAB code are not translated
    # as they use the 'imoverlay' function which doesn't have a direct
    # equivalent in matplotlib. If needed, we can implement a custom solution.
    plt.show()

def main(threshold, Zs, Z, gs, Cth, HZth, LZth, X, Y, X2, Y2, HZtharea, LZtharea, Ctharea):
    if threshold == 0:
        creek, CumArea, binrange, CumAreas, binranges, _, _ = creek_detection(Zs, Z, gs, Cth, HZth, LZth)
    else:
        creek, CumArea, binrange, CumAreas, binranges, HZth, LZth, Cth, HZtharea, LZtharea, Ctharea = creek_detection_manual(Zs, Z, gs)

    # Create dictionary with all variables
    variables_correction = {
        "creek": creek,
        "CumArea": CumArea,
        "binrange": binrange,
        "CumAreas": CumAreas,
        "binranges": binranges
    }

    # Convert all numpy arrays to lists
    variables_correction = {
        key: value.tolist() if isinstance(value, np.ndarray) else value 
        for key, value in variables_correction.items()
    }

    # Save to a JSON file
    with open("processed_variables_Creek_Detection.json", "w") as f:
        json.dump(variables_correction, f)

    # Map the creek network based on the threshold criteria
    # Ensure Z is a numpy array
    Z = np.array(Z)
    X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    X2, Y2 = X, Y  # Assuming X2, Y2 are the same as X, Y
    figure_creek_detection(X, Y, creek, X2, Y2, Z, CumArea, binrange, CumAreas, binranges, LZtharea, LZth, HZtharea, HZth, Ctharea, Cth)

    # Extract the raw creek mask
    creekmask = ~np.isnan(creek)
    figure_raw_creek_mask(creekmask)

    return creekmask

# Usage example:
# threshold = 0  # or 1 for manual detection
# Zs = ...  # your slope data
# Z = ...   # your elevation data
# gs = ...  # your grid size
# Cth = ... # your slope threshold
# HZth = ... # your high elevation threshold
# LZth = ... # your low elevation threshold
# creekmask = main(threshold, Zs, Z, gs, Cth, HZth, LZth)

creekmask = main(threshold, Zs, Z, gs, Cth, HZth, LZth, X, Y, X2, Y2, HZtharea, LZtharea, Ctharea)
with h5py.File('creekmask_raw.h5', 'w') as hf:
    hf.create_dataset("creekmask_raw", data=creekmask)

# Prevent script from closing immediately
input("Press Enter to close...")