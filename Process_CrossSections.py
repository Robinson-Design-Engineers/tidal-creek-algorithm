import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
import numpy.ma as ma

def bresenham(x0, y0, x1, y1):
    """Implementation of Bresenham's line algorithm"""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            yield (x, y)
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            yield (x, y)
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    yield (x, y)

def process_xsects(creekmask, creek, IDXSEG, ordermax, figview=False):
    """
    Process creek cross-sections to calculate width, depth, and area.
    
    Parameters:
    creekmask : ndarray
        Binary mask of the creek
    creek : ndarray
        Elevation data of the creek
    IDXSEG : ndarray
        Segment indices array
    ordermax : int
        Maximum order of segments
    figview : bool
        Whether to display visualization figures
    
    Returns:
    tuple
        (WIDTH, DEPTH, AREA) arrays
    """
    # Replace negative values with 1
    IDXSEG[IDXSEG < 0] = 1
    
    # Determine number of cross sections
    num_segments = np.sum(IDXSEG[0, :] != 0) // 6
    
    # Initialize output arrays
    depth = np.full((ordermax, num_segments), np.nan)
    width = np.full((ordermax, num_segments), np.nan)
    area = np.full((ordermax, num_segments), np.nan)
    
    # Initialize figures if needed
    if figview:
        creekmaskpic = np.zeros_like(creekmask)
        creekmaskpic[creekmask == 1] = 0
        creekmaskpic[creekmask == 0] = 1
        
        plt.figure()
        plt.title('Creek cross-sections', fontsize=18, fontweight='bold')
        plt.imshow(creekmaskpic)
        
        fig, ax = plt.subplots()
        ax.set_xlabel('Distance (ft)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Depth (ft)', fontsize=14, fontweight='bold')
        ax.set_title('Creek cross-sections')
        colors = ['black'] + plt.cm.parula(np.linspace(0, 1, 5)).tolist()
        
        # Create legend handles
        for i, color in enumerate(colors):
            ax.plot([], [], color=color, linewidth=2, label=f'Order {i+1}')
        ax.legend()
    
    inverse_order = np.arange(ordermax, 0, -1)
    
    # Process each segment
    for order in range(ordermax):
        print('order = ', order)
        iter_idx = 0
        for seg in range(0, IDXSEG.shape[1], 6):
            print(' IDXSEG.shape[1]')
            print(' seg = ', seg)
            if IDXSEG[order, seg] == 0:  # Skip if no segment
                print('  IDXSEG[order, seg] = ', IDXSEG[order, seg])
                print('  skipped bc no segment')
                continue
                
            # Get segment coordinates
            x1 = round(IDXSEG[order, seg])
            y1 = round(IDXSEG[order, seg + 1])
            x2 = round(IDXSEG[order, seg + 2])
            y2 = round(IDXSEG[order, seg + 3])
            x3 = round(IDXSEG[order, seg + 4])
            y3 = round(IDXSEG[order, seg + 5])
            
            if creekmask[x1, y1] == 1:  # Check if middle point is on creek
                # Generate line points using Bresenham's algorithm
                xlinestart, ylinestart = zip(*bresenham(x2, y2, x1, y1))
                xlineend, ylineend = zip(*bresenham(x3, y3, x1, y1))
                # Using zip(*...), you "unpack" the tuples into two separate lists: one for x coordinates and one for y coordinates. -ChatGPT

                # Combine the results
                xline = xlinestart + xlineend
                yline = ylinestart + ylineend

                # Remove out of bounds indices
                valid_coords = (xline >= 0) & (yline >= 0) & \
                             (xline < creekmask.shape[0]) & (yline < creekmask.shape[1])
                xline = xline[valid_coords]
                yline = yline[valid_coords]
                # BOOKMARK
                if len(xline) > 1:
                    # Calculate cross-section measurements
                    idx = np.ravel_multi_index((xline, yline), creek.shape)
                    z_profile = creek.ravel()[idx]
                    
                    if len(z_profile) > 1 and np.sum(~np.isnan(z_profile)) >= 2:
                        # Interpolate NaN values
                        mask = ~np.isnan(z_profile)
                        x_valid = np.arange(len(z_profile))[mask]
                        z_valid = z_profile[mask]
                        f = interpolate.interp1d(x_valid, z_valid)
                        z_profile = f(np.arange(len(z_profile)))
                        
                        # Calculate width
                        width_val = np.sqrt((xline[-1] - xline[0])**2 + 
                                          (yline[-1] - yline[0])**2)
                        width[order, iter_idx] = width_val
                        
                        # Calculate depth
                        depth_val = np.max(z_profile) - np.min(z_profile)
                        depth[order, iter_idx] = depth_val
                        
                        # Calculate area
                        z_top = np.linspace(z_profile[0], z_profile[-1], len(z_profile))
                        area_val = np.sum(z_top - z_profile)
                        area[order, iter_idx] = area_val
                        
                        if figview:
                            z_vals = z_top - z_profile
                            x_vals = np.arange(len(z_vals))
                            plt.figure(2)
                            plt.plot(x_vals, -z_vals, linewidth=2, 
                                   color=colors[inverse_order[order]])
                    
                    elif len(z_profile) == 1:
                        # Handle single point case
                        width[order, iter_idx] = 999
                        depth[order, iter_idx] = 999
                        area[order, iter_idx] = 999
                        
            iter_idx += 1
    
    # Final formatting
    # Remove columns with all NaN values
    valid_cols = ~np.all(np.isnan(area), axis=0)
    width = width[:, valid_cols]
    depth = depth[:, valid_cols]
    area = area[:, valid_cols]
    
    # Replace 999 values with mean of non-999 values in the same row
    for i in range(area.shape[0]):
        for arr in [width, depth, area]:
            row = arr[i, :]
            mask = row == 999
            if np.any(mask):
                valid_vals = row[~mask & ~np.isnan(row)]
                if len(valid_vals) > 0:
                    arr[i, mask] = np.mean(valid_vals)
    
    return width, depth, area