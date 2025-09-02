# creek orders check plots functions

# check creek orders
import numpy as np #type:ignore 
import matplotlib.pyplot as plt #type:ignore
from matplotlib.colors import ListedColormap #type:ignore

def plot_creek_orders(skeleton, creekorder, X=None, Y=None, colors=None, figsize=(10, 8), dpi=100):
    # Input validation
    if not isinstance(skeleton, np.ndarray) or not isinstance(creekorder, np.ndarray):
        raise ValueError("skeleton and creekorder must be numpy arrays")
    
    if skeleton.shape != creekorder.shape:
        raise ValueError("skeleton and creekorder must have the same shape")
        
    # Default color scheme if none provided
    if colors is None:
        colors = ['#000000',  # Black for order 1
                 '#451caf',  # Deep purple for order 2
                 '#1878ff',  # Blue for order 3
                 '#00c2ba',  # Turquoise for order 4
                 '#c9c200',  # Yellow-green for order 5
                 '#f7fd00']  # Bright yellow for order 6
    
    discrete_cmap = ListedColormap(colors)
    
    # Create coordinate arrays if not provided
    if X is None or Y is None:
        ny, nx = skeleton.shape
        # Create coordinates with one more point than the data dimensions
        x = np.arange(nx + 1)
        y = np.arange(ny + 1)
        X, Y = np.meshgrid(x, y)
    else:
        # If coordinates are provided, verify/adjust their dimensions
        if X.shape[0] != skeleton.shape[0] + 1 or X.shape[1] != skeleton.shape[1] + 1:
            # Adjust coordinates to have one more point than the data
            x = np.linspace(X.min(), X.max(), skeleton.shape[1] + 1)
            y = np.linspace(Y.min(), Y.max(), skeleton.shape[0] + 1)
            X, Y = np.meshgrid(x, y)
    
    # Mask the creek orders
    # masked_orders = np.where(creekorder == 0, np.nan, creekorder)
    masked_orders = np.ma.masked_where(~skeleton, creekorder)
    # masked_orders = np.ma.masked_where(creekorder == 0, creekorder)
    # masked_orders = creekorder  # No masking at all
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Set the aspect ratio to 1:1
    ax.set_aspect('equal')
    
    # Plot the data using pcolormesh with explicit shading parameter
    # im = ax.pcolormesh(X, Y, masked_orders, 
    #                   cmap=discrete_cmap, 
    #                   vmin=1, 
    #                   vmax=len(colors) + 1,
    #                   shading='flat')
    im = ax.pcolormesh(Y, X, masked_orders, 
                      cmap=discrete_cmap, 
                      vmin=1, 
                      vmax=len(colors) + 1,
                      shading='flat')
    
    # Add and customize the colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Creek Order', fontsize=10)
    
    # Center the colorbar ticks between colors
    cbar.set_ticks(np.arange(1.5, len(colors) + 1.5))
    cbar.set_ticklabels(np.arange(1, len(colors) + 1))
    
    # Add labels and title
    ax.set_xlabel('X Distance')
    ax.set_ylabel('Y Distance')
    ax.set_title('Creek Network Order Classification')
    
    # Invert the y-axis for traditional GIS visualization
    ax.invert_yaxis()
    
    return fig, ax


# # Plot the sample data
# fig, ax = plot_creek_orders(skeleton_breached, creekordersing)
# plt.show()

# check creek orders thickened skeleton
import numpy as np #type:ignore 
import matplotlib.pyplot as plt #type:ignore
from matplotlib.colors import ListedColormap #type:ignore
from skimage.morphology import dilation, disk # type: ignore

def convert_to_numpy(data, dtype=None):
    """Safely convert input data to numpy array"""
    try:
        if isinstance(data, list):
            # Handle nested lists by converting inner lists first
            if data and isinstance(data[0], list):
                data = [np.array(row) for row in data]
        return np.array(data, dtype=dtype)
    except Exception as e:
        print(f"Error converting data to numpy array: {e}")
        print("Data sample:", data[:2] if isinstance(data, list) else data)
        raise

def plot_creek_orders_big(skeleton, creekorder, ordermax, X, Y, colors, figsize, dpi):
    """Plot creek orders"""

    np.savetxt("DEBUG/ipynb_creek_order_pre_convert.csv", creekorder, delimiter=",", fmt="%.2f")
    creekorder = convert_to_numpy(creekorder, dtype=float)
    np.savetxt("DEBUG/ipynb_creek_order_post_convert.csv", creekorder, delimiter=",", fmt="%.2f")

    # Create coordinate arrays if not provided
    if X is None or Y is None:
        ny, nx = skeleton.shape
        # Create coordinates with one more point than the data dimensions
        x = np.arange(nx + 1)
        y = np.arange(ny + 1)
        X, Y = np.meshgrid(x, y)
    else:
        # If coordinates are provided, verify/adjust their dimensions
        if X.shape[0] != skeleton.shape[0] + 1 or X.shape[1] != skeleton.shape[1] + 1:
            # Adjust coordinates to have one more point than the data
            x = np.linspace(X.min(), X.max(), skeleton.shape[1] + 1)
            y = np.linspace(Y.min(), Y.max(), skeleton.shape[0] + 1)
            X, Y = np.meshgrid(x, y)
            
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Set the aspect ratio to 1:1
    ax.set_aspect('equal')
    # ax1.clear()

    # Create discrete colormap: black + 5 colors
    # Define the 5 colors you want to use
    colors = ['#000000', '#451caf', '#1878ff', '#00c2ba', '#c9c200', '#f7fd00']
    # Create the discrete colormap
    discrete_cmap = ListedColormap(colors)

    # Dilate the skeleton for better visibility
    np.savetxt("DEBUG/ipynb_skeleton.csv", skeleton, delimiter=",", fmt="%.2f")
    # dilated_skeleton = dilation(skeleton, disk(1))
    dilated_skeleton = dilation(skeleton, disk(ordermax))
    np.savetxt("DEBUG/ipynb_dilated_skeleton.csv", dilated_skeleton, delimiter=",", fmt="%.2f")
    # dilated_skeleton = dilation(dilated_skeleton, disk(1))
    # dilated_skeleton = bwmorph_thicken(skeleton, 1)

    # Swap creek orders from Strahler to Reverse Strahler
    creekorder[creekorder == 0] = np.nan
    # check max_order is actually true to creekordermask
    finite_values = creekorder[np.isfinite(creekorder)]
    if len(finite_values) > 0:
        actual_max = np.max(finite_values)
        if actual_max == ordermax:
            pass
        else:
            print(f'creekorder max ({actual_max}) != ordermax ({ordermax})')
            ordermax = actual_max # reset ordermax
    else:
        print('No finite values in creekorder')
    max_order = np.full_like(creekorder, float(ordermax + 1))
    max_order = max_order.astype(float)  # Ensure max_order is float
    max_order[np.isnan(creekorder)] = np.nan
    creek_order_swapped = max_order - creekorder

    # Dilate the creek order mask for better visibility
    np.savetxt("DEBUG/ipynb_creek_order_swapped.csv", creek_order_swapped, delimiter=",", fmt="%.2f")
    # masked_orders = np.ma.masked_where(~dilated_skeleton, creekorder)
    masked_orders = np.ma.masked_where(~dilated_skeleton, creek_order_swapped)

    # Plot the data using pcolormesh with explicit shading parameter
    np.savetxt("DEBUG/ipynb_X.csv", X, delimiter=",", fmt="%.2f")
    np.savetxt("DEBUG/ipynb_Y.csv", Y, delimiter=",", fmt="%.2f")
    # im = ax.pcolormesh(X, Y, masked_orders, 
    #                   cmap=discrete_cmap, 
    #                   vmin=1, 
    #                   vmax=len(colors) + 1,
    #                   shading='flat')
    im = ax.pcolormesh(Y, X, masked_orders, 
                      cmap=discrete_cmap, 
                      vmin=1, 
                      vmax=len(colors) + 1,
                      shading='flat')
    
    # Add and customize the colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Creek Order', fontsize=10)
    
    # Center the colorbar ticks between colors
    cbar.set_ticks(np.arange(1.5, len(colors) + 1.5))
    cbar.set_ticklabels(np.arange(1, len(colors) + 1))
    
    # Add labels and title
    ax.set_xlabel('X Distance')
    ax.set_ylabel('Y Distance')
    ax.set_title('Creek Network Order Classification')
    
    # Invert the y-axis for traditional GIS visualization
    ax.invert_yaxis()
    
    return fig, ax


# # Plot the sample data
# fig, ax = plot_creek_orders_big(skeleton_breached, creekorder, X=None, Y=None, colors=None, figsize=(10, 8), dpi=100)
# plt.show()