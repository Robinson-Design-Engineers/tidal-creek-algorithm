import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
import numpy.ma as ma
from scipy import ndimage

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
    num_segments = np.sum(IDXSEG[1, :] != 0) // 6
    
    # Initialize output arrays
    DEPTH = np.full((ordermax+1, num_segments), np.nan) # do ordermax+1 bc Python is zero-based
    WIDTH = np.full((ordermax+1, num_segments), np.nan) # do ordermax+1 bc Python is zero-based
    AREA = np.full((ordermax+1, num_segments), np.nan) # do ordermax+1 bc Python is zero-based
    
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
    for order in range(1,ordermax+1):
        iter = 0
        for seg in range(0, IDXSEG.shape[1], 6):
            if IDXSEG[order, seg] == 0:  # Skip if no segment
                continue
                
            # Get segment coordinates
            x1 = round(IDXSEG[order, seg])
            y1 = round(IDXSEG[order, seg + 1])
            x2 = round(IDXSEG[order, seg + 2])
            y2 = round(IDXSEG[order, seg + 3])
            x3 = round(IDXSEG[order, seg + 4])
            y3 = round(IDXSEG[order, seg + 5])


            if creekmask[y1, x1] == 1:  # Check if middle point is on creek
                # Get points from Bresenham's algorithm for start and end segments
                line_start = list(bresenham(x2, y2, x1, y1))
                line_end = list(bresenham(x3, y3, x1, y1))

                # Unzip the (x,y) tuples into separate lists and convert to numpy arrays immediately
                xlinestart, ylinestart = map(np.array, zip(*line_start))
                xlineend, ylineend = map(np.array, zip(*line_end))

                # Combine start and end segments
                xline = np.concatenate([xlinestart, xlineend])
                yline = np.concatenate([ylinestart, ylineend])
                
                # Remove out of bounds indices
                coordlogic = (xline > 0) & (yline > 0) & (xline < creekmask.shape[0]) & (yline < creekmask.shape[1])
                xline = xline[coordlogic]
                yline = yline[coordlogic]

                # Create a mask where cross-section pixels = 0.5
                segmentmask = np.zeros_like(creekmask)
                segmentmask[yline, xline] = 0.5 # xline,yline or yline,xline?

                # Find indices belonging to both segmentmask and creekmask
                creekmasktemp = creekmask/2
                creekmasktemp = creekmasktemp + segmentmask

                # In creekmasktemp, pixel=1 belongs to both the cross-section and the creek, 
                # pixel=0.5 belongs only to one of them
                creekmasktemp[creekmasktemp == 0.5] = 0
                creekmasktemp = creekmasktemp.astype(bool)
                # Get indices where creekmasktemp is True
                ytest, xtest = np.where(creekmasktemp)  # Note: np.where returns (rows, cols)

                # Get connected components labeled matrix (equivalent to bwconncomp + labelmatrix)
                L, num_features = ndimage.label(creekmasktemp, structure=np.ones((3,3)))
                # Now L contains labels 1,2,3... for each connected component, just like MATLAB
                
                # Get label number of the object containing the middle point
                objectnum = L[y1, x1] # x,y or y,x?

                # Remove elements not connected to middle point
                creekmasktemp[L != objectnum] = 0

                # Get indices of the object we want to keep
                yobj, xobj = np.where(L == objectnum)  # note: returns row, col indices

                # Create coordinate arrays
                B = np.column_stack((yobj, xobj))  # switched order
                A = np.column_stack((yline, xline))  # switched order
                # Find intersection of coordinates
                C = np.array([x for x in set(map(tuple, B)) & set(map(tuple, A))])

                # If we found any intersecting points
                if len(C) > 0:
                    xline2 = C[:, 1]
                    yline2 = C[:, 0]
                else:
                    xline2 = np.array([])
                    yline2 = np.array([])

                if len(xline2) > 1:
                    # Create binary mask
                    bw = np.zeros_like(creek, dtype=bool)
                    bw[yline2.astype(int), xline2.astype(int)] = True # x,y or y,x?
                    # Fill holes
                    bw = ndimage.binary_fill_holes(bw)

                    # Create kernel for finding neighbors
                    end_kernel = np.array([[1, 1, 1],
                                        [1, 0, 1],
                                        [1, 1, 1]])
                    # Count neighbors
                    E_neighbor_count = ndimage.convolve(bw.astype(int), end_kernel, mode='constant', cval=0)
                    # Find endpoints (points with 1 neighbor or isolated points)
                    endmembers = ((E_neighbor_count == 1) | (E_neighbor_count == 0)) & bw
                    
                    # Get coordinates of endpoints
                    Yem, Xem = np.where(endmembers)
                    
                    # Get start and end points
                    x1, x2 = Xem[0], Xem[1]
                    y1, y2 = Yem[0], Yem[1]
                    
                    # Get new line using Bresenham
                    line_points = list(bresenham(x1, y1, x2, y2))
                    xline2, yline2 = zip(*line_points)
                    xline2 = np.array(xline2)
                    yline2 = np.array(yline2)
                
                # Add NaN padding to array
                yline2_padded = np.concatenate(([np.nan], yline2, [np.nan]))

                # Find indices of NaN values
                F = np.where(np.isnan(yline2_padded))[0]

                # Calculate differences and subtract 2
                D = np.diff(F) - 2

                # Find position of maximum difference
                if len(D) > 0:  # Make sure we have differences to find max from
                    M = np.max(D)
                    L = np.argmax(D)
                    
                    # Extract the longest block
                    xline2 = xline2[F[L]:F[L] + M + 1]  # The longest block
                    yline2 = yline2[F[L]:F[L] + M + 1]
                
                # Calculate line parameters
                A3 = (yline2[-1] - yline2[0]) / (xline2[-1] - xline2[0])  # slope
                B3 = yline2[-1] - xline2[-1] * A3  # intercept

                # Calculate differences
                diffy = abs(yline2[-1] - yline2[0])
                diffx = abs(xline2[-1] - xline2[0])

                if diffx >= diffy:
                    # Create x coordinates and calculate corresponding y
                    xline3 = np.arange(int(min(xline2)), int(max(xline2)) + 1)
                    yline3 = np.round(A3 * xline3 + B3).astype(int)
                else:
                    # Create y coordinates and calculate corresponding x
                    yline3 = np.arange(int(min(yline2)), int(max(yline2)) + 1)
                    xline3 = np.round((yline3 - B3) / A3).astype(int)

                # Convert coordinates and creek to numpy arrays only if needed
                if not isinstance(xline, np.ndarray):
                    xline = np.array(xline, dtype=int)
                if not isinstance(yline, np.ndarray):
                    yline = np.array(yline, dtype=int)
                if not isinstance(creek, np.ndarray):
                    creek = np.array(creek)

                # Get elevation values directly using array indexing
                z_profile = creek[yline2.astype(int), xline2.astype(int)]

                # Remove any NaN indices if needed
                valid_indices = ~np.isnan(xline2) & ~np.isnan(yline2)
                xline2 = xline2[valid_indices]
                yline2 = yline2[valid_indices]
                
                if len(xline2) > 1:
                    # Convert coordinates and creek to numpy arrays only if needed
                    if not isinstance(xline, np.ndarray):
                        xline = np.array(xline, dtype=int)
                    if not isinstance(yline, np.ndarray):
                        yline = np.array(yline, dtype=int)
                    if not isinstance(creek, np.ndarray):
                        creek = np.array(creek)
                        
                    # Get elevation profile
                    z_profile = creek[yline2.astype(int), xline2.astype(int)]
                    
                    if len(z_profile) > 1 and np.sum(~np.isnan(z_profile)) >= 2:

                        # Interpolate NaN values
                        bd = np.isnan(z_profile) # true where z_profile has a NaN
                        # Get indices of good and bad data
                        gd = np.where(~bd)[0]  # indices where we DON'T have NaNs
                        bd_indices = np.where(bd)[0]  # indices where we DO have NaNs
                        if len(gd) >= 2:  # Need at least 2 points for interpolation
                            # Create interpolation function using good data with boundary handling
                            interp_func = interpolate.interp1d(gd, z_profile[gd], 
                                                            bounds_error=False,   # Don't raise error for out of bounds
                                                            fill_value='extrapolate')  # Extrapolate beyond bounds
                            # Fill in the NaN values with interpolated values
                            z_profile[bd] = interp_func(bd_indices)
                        elif len(gd) == 1:
                            # If only one good value, fill all NaNs with that value
                            z_profile[bd] = z_profile[gd[0]]
                        else:
                            # If no good values, leave as NaN
                            pass
                        
                        # Plot if requested
                        if figview:
                            plt.figure(2)  # equivalent to figure(f2)
                            # Plot white background line
                            plt.plot(yline2, xline2, color='white', linewidth=5)
                            # Plot colored line
                            plt.plot(yline2, xline2, 
                                    color=colors[inverse_order[order]], 
                                    linewidth=3)
                            
                    if len(z_profile) > 1:  # case where cross-section is a line
                        # Add bounds check
                        if 0 <= order < WIDTH.shape[0] and 0 <= iter < WIDTH.shape[1]:
                            # Calculate width using endpoints
                            width_val = np.sqrt((xline2[-1] - xline2[0])**2 + (yline2[-1] - yline2[0])**2)
                            WIDTH[order, iter] = width_val  # store calculated width
                            
                            # Calculate and store depth
                            depth_val = np.nanmax(z_profile) - np.nanmin(z_profile)
                            DEPTH[order, iter] = depth_val
                        else:
                            print(f"Warning: Indices {order}, {iter} out of bounds for shape {WIDTH.shape}")
                                                
                        # Get start and end points and their elevations
                        startpt = np.array([yline2[0], xline2[0]])
                        Zstart = creek[int(startpt[0]), int(startpt[1])]
                        
                        endpt = np.array([yline2[-1], xline2[-1]])
                        Zend = creek[int(endpt[0]), int(endpt[1])]
                        
                        meanheight = np.mean([Zstart, Zend])
                        
                        # Calculate Z values
                        Z = np.nanmax(z_profile) - z_profile
                        
                        # Find longest continuous section
                        yline2_padded = np.concatenate(([np.nan], yline2, [np.nan]))
                        F = np.where(np.isnan(yline2_padded))[0]
                        D = np.diff(F) - 2
                        
                        if len(D) > 0:
                            M = np.max(D)
                            L = np.argmax(D)
                            
                            # Extract longest block
                            xline2 = xline2[F[L]:F[L] + M + 1]  # The longest block
                            yline2 = yline2[F[L]:F[L] + M + 1]
                            
                            # Calculate line parameters
                            A3 = (yline2[-1] - yline2[0]) / (xline2[-1] - xline2[0])
                            B3 = yline2[-1] - xline2[-1] * A3
                            
                            # Calculate differences
                            diffy = abs(yline2[-1] - yline2[0])
                            diffx = abs(xline2[-1] - xline2[0])
                        
                        if diffx >= diffy:
                            # Create x coordinates and calculate corresponding y
                            xline3 = np.arange(int(min(xline2)), int(max(xline2)) + 1)
                            yline3 = np.round(A3 * xline3 + B3).astype(int)
                        else:
                            # Create y coordinates and calculate corresponding x
                            yline3 = np.arange(int(min(yline2)), int(max(yline2)) + 1)
                            xline3 = np.round((yline3 - B3) / A3).astype(int)

                        # Calculate radii
                        r = np.sqrt(xline3**2 + yline3**2)
                        rnorm = np.abs(r - np.max(r))
                        rnorm[rnorm == 0] = 1
                        r = rnorm

                        # Create mask first
                        mask_nonzero = r != 0

                        # Then modify the filtering code to ensure dimensions match:
                        if len(Z) == len(r):  # Only proceed if dimensions match
                            # Remove zeros
                            Z = Z[mask_nonzero]
                            r = r[mask_nonzero]

                            # Remove NaNs from both Z and r
                            mask_valid = ~np.isnan(Z)
                            r = r[mask_valid]
                            Z = Z[mask_valid]

                            # Remove NaNs from z_profile
                            z_profile = z_profile[~np.isnan(z_profile)]
                        else:
                            print(f"Warning: Z and r have different lengths. Z: {len(Z)}, r: {len(r)}")

                        # # Remove zeros
                        # mask_nonzero = r != 0
                        # Z = Z[mask_nonzero]
                        # r = r[mask_nonzero]

                        # # Remove NaNs from both Z and r
                        # mask_valid = ~np.isnan(Z)
                        # r = r[mask_valid]
                        # Z = Z[mask_valid]

                        # # Remove NaNs from z_profile
                        # z_profile = z_profile[~np.isnan(z_profile)]

                        if len(Z) == 0 or np.all(np.isnan(Z)):
                            # If Z is empty or all NaN, set measurements to NaN
                            AREA[order, iter] = np.nan
                            DEPTH[order, iter] = np.nan
                            WIDTH[order, iter] = np.nan

                        elif len(Z) == 1:
                            # If Z has only one point, use special values
                            AREA[order, iter] = 999   # AREA is that of a semi-circle
                            WIDTH[order, iter] = 999  # width = 1 pixel = 1 meter
                            DEPTH[order, iter] = 999

                        else:
                            # Create top line and calculate AREA
                            Ztop = np.linspace(z_profile[0], z_profile[-1], len(z_profile))
                            Ztop = Ztop.reshape(-1, 1)  # equivalent to Ztop'
                            
                            # Calculate AREA ignoring NaN values
                            area_val = np.nansum(Ztop - z_profile)
                            
                            # Calculate Z differences
                            Z = Ztop - z_profile
                            
                            # Create x-axis values for plotting
                            lval = np.arange(1, len(Z) + 1)
                            
                            if figview:
                                plt.figure(1)
                                # Plot and set legend visibility off
                                line = plt.plot(lval, -Z, linewidth=2, color=colors[inverse_order[order]])
                                # Hide from legend
                                plt.setp(line, label="_nolegend_")
                            
                            AREA[order, iter] = area_val

                        # Increment iterator
                        iter += 1
    
                    elif len(z_profile) == 1:  # case where cross-section is a point
                        # Find middle point
                        mid_idx = round(len(xline2) / 2)  # equivalent to round(mean(length()))
                        midsegx = xline2[mid_idx]
                        midsegy = yline2[mid_idx]
                        
                        # Define neighboring points
                        lowx = midsegx - 1
                        highx = midsegx + 1
                        lowy = midsegy - 1
                        highy = midsegy + 1
                        
                        # Set special values
                        AREA[order, iter] = 999    # area is that of a semi-circle
                        WIDTH[order, iter] = 999   # width = 1 pixel = 1 meter
                        DEPTH[order, iter] = 999
                        iter += 1

                    else:  # case where there is no cross-section
                        # Set all measurements to NaN
                        AREA[order, iter] = np.nan
                        DEPTH[order, iter] = np.nan
                        WIDTH[order, iter] = np.nan
                        iter += 1
                else:
                    AREA[order, iter] = np.nan
                    DEPTH[order, iter] = np.nan
                    WIDTH[order, iter] = np.nan
                    iter += 1
            else: # the whole cross-section is NaN
                AREA[order, iter] = np.nan
                DEPTH[order, iter] = np.nan
                WIDTH[order, iter] = np.nan
                iter += 1
    
    # Final formatting - Ensure the correct dimensions

    # Remove columns with all NaN values
    valid_cols = ~np.all(np.isnan(AREA), axis=0)
    WIDTH = WIDTH[:, valid_cols]
    DEPTH = DEPTH[:, valid_cols]
    AREA = AREA[:, valid_cols]
    
    # Replace 999 values with mean of non-999 values in the same row, only if 999s exist
    for i in range(AREA.shape[0]):  # Loop through each row
        # Handle WIDTH if 999s exist
        if 999 in WIDTH[i, :]:
            W = WIDTH[i, :].copy()
            W2 = W.copy() 
            W2[W2 == 999] = np.nan  # Replace 999s with NaN
            W[W == 999] = np.nanmean(W2)  # Replace 999s with mean of non-999 values
            WIDTH[i, :] = W
        
        # Handle DEPTH if 999s exist  
        if 999 in DEPTH[i, :]:
            D = DEPTH[i, :].copy()
            D2 = D.copy()
            D2[D2 == 999] = np.nan
            D[D == 999] = np.nanmean(D2)
            DEPTH[i, :] = D
            
        # Handle AREA if 999s exist
        if 999 in AREA[i, :]:
            A = AREA[i, :].copy()
            A2 = A.copy()
            A2[A2 == 999] = np.nan
            A[A == 999] = np.nanmean(A2)
            AREA[i, :] = A
    
    return WIDTH, DEPTH, AREA

def process_xsects_diagnostic(creekmask, creek, skeleton, IDXSEG, ordermax, figview=False):
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
    num_segments = np.sum(IDXSEG[1, :] != 0) // 6
    
    # Initialize output arrays
    DEPTH = np.full((ordermax+1, num_segments), np.nan) # do ordermax+1 bc Python is zero-based
    WIDTH = np.full((ordermax+1, num_segments), np.nan) # do ordermax+1 bc Python is zero-based
    AREA = np.full((ordermax+1, num_segments), np.nan) # do ordermax+1 bc Python is zero-based
    
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
    for order in range(1,ordermax+1):
        print('order = ', order)
        iter = 0
        print('next is "for seg in range(0, IDXSEG.shape[1], 6):", where IDXSEG.shape[1] = ', IDXSEG.shape[1])
        for seg in range(0, IDXSEG.shape[1], 6):
            print('\nseg = ', seg)
            if IDXSEG[order, seg] == 0:  # Skip if no segment
                print('IDXSEG[order, seg] = ', IDXSEG[order, seg])
                print('skipped bc no segment')
                continue
                
            # Get segment coordinates
            x1 = round(IDXSEG[order, seg])
            y1 = round(IDXSEG[order, seg + 1])
            x2 = round(IDXSEG[order, seg + 2])
            y2 = round(IDXSEG[order, seg + 3])
            x3 = round(IDXSEG[order, seg + 4])
            y3 = round(IDXSEG[order, seg + 5])
            print('x1, y1, x2, y2, x3, y3 = ', x1, y1, x2, y2, x3, y3)
            
            print('creekmask[y1, x1] = ', creekmask[y1, x1])
            # # plot creek mask and middle point to check
            # plt.figure()
            # plt.imshow(creekmask, cmap='gray')
            # # Create a custom colormap for skeleton with transparency
            # # Make a copy of skeleton as float type for alpha values
            # skeleton_rgba = np.zeros((*skeleton.shape, 4))
            # skeleton_rgba[skeleton == 1] = [0, 0, 1, 0.7]  # Blue with 0.7 alpha where True
            # skeleton_rgba[skeleton == 0] = [0, 0, 0, 0]    # Transparent where False
            # plt.imshow(skeleton_rgba, origin='upper', label='skeleton')
            # plt.plot(x1, y1, '+', markersize=10, markeredgecolor='r', markerfacecolor='none', alpha=0.6, label='point of interest')
            # plt.title('Creek Mask, Skeleton, midpoint')
            # plt.show()

            if creekmask[y1, x1] == 1:  # Check if middle point is on creek
                # Get points from Bresenham's algorithm for start and end segments
                line_start = list(bresenham(x2, y2, x1, y1))
                line_end = list(bresenham(x3, y3, x1, y1))

                # Unzip the (x,y) tuples into separate lists and convert to numpy arrays immediately
                xlinestart, ylinestart = map(np.array, zip(*line_start))
                xlineend, ylineend = map(np.array, zip(*line_end))

                # Combine start and end segments
                xline = np.concatenate([xlinestart, xlineend])
                yline = np.concatenate([ylinestart, ylineend])
                
                # Remove out of bounds indices
                coordlogic = (xline > 0) & (yline > 0) & (xline < creekmask.shape[0]) & (yline < creekmask.shape[1])
                xline = xline[coordlogic]
                yline = yline[coordlogic]

                # Create a mask where cross-section pixels = 0.5
                segmentmask = np.zeros_like(creekmask)
                segmentmask[yline, xline] = 0.5 # xline,yline or yline,xline?

                # Find indices belonging to both segmentmask and creekmask
                creekmasktemp = creekmask/2
                creekmasktemp = creekmasktemp + segmentmask

                # In creekmasktemp, pixel=1 belongs to both the cross-section and the creek, 
                # pixel=0.5 belongs only to one of them
                creekmasktemp[creekmasktemp == 0.5] = 0
                creekmasktemp = creekmasktemp.astype(bool)
                # Get indices where creekmasktemp is True
                ytest, xtest = np.where(creekmasktemp)  # Note: np.where returns (rows, cols)

                # Get connected components labeled matrix (equivalent to bwconncomp + labelmatrix)
                L, num_features = ndimage.label(creekmasktemp, structure=np.ones((3,3)))
                # Now L contains labels 1,2,3... for each connected component, just like MATLAB
                
                # Get label number of the object containing the middle point
                objectnum = L[y1, x1] # x,y or y,x?

                # Remove elements not connected to middle point
                creekmasktemp[L != objectnum] = 0

                # Get indices of the object we want to keep
                yobj, xobj = np.where(L == objectnum)  # note: returns row, col indices

                # Create coordinate arrays
                B = np.column_stack((yobj, xobj))  # switched order
                A = np.column_stack((yline, xline))  # switched order
                # Find intersection of coordinates
                C = np.array([x for x in set(map(tuple, B)) & set(map(tuple, A))])

                print('C = ', C)

                # If we found any intersecting points
                if len(C) > 0:
                    xline2 = C[:, 1]
                    yline2 = C[:, 0]
                else:
                    xline2 = np.array([])
                    yline2 = np.array([])

                if len(xline2) > 1:
                    # Create binary mask
                    bw = np.zeros_like(creek, dtype=bool)
                    bw[yline2.astype(int), xline2.astype(int)] = True # x,y or y,x?
                    # Fill holes
                    bw = ndimage.binary_fill_holes(bw)

                    # Create kernel for finding neighbors
                    end_kernel = np.array([[1, 1, 1],
                                        [1, 0, 1],
                                        [1, 1, 1]])
                    # Count neighbors
                    E_neighbor_count = ndimage.convolve(bw.astype(int), end_kernel, mode='constant', cval=0)
                    # Find endpoints (points with 1 neighbor or isolated points)
                    endmembers = ((E_neighbor_count == 1) | (E_neighbor_count == 0)) & bw
                    
                    # Get coordinates of endpoints
                    Yem, Xem = np.where(endmembers)
                    
                    # Get start and end points
                    x1, x2 = Xem[0], Xem[1]
                    y1, y2 = Yem[0], Yem[1]
                    
                    # Get new line using Bresenham
                    line_points = list(bresenham(x1, y1, x2, y2))
                    xline2, yline2 = zip(*line_points)
                    xline2 = np.array(xline2)
                    yline2 = np.array(yline2)
                    # print('xline2 after bresenham = ', xline2)
                    # print('yline2 after bresenham = ', yline2)
                
                # Add NaN padding to array
                yline2_padded = np.concatenate(([np.nan], yline2, [np.nan]))

                # Find indices of NaN values
                F = np.where(np.isnan(yline2_padded))[0]

                # Calculate differences and subtract 2
                D = np.diff(F) - 2

                # Find position of maximum difference
                if len(D) > 0:  # Make sure we have differences to find max from
                    M = np.max(D)
                    L = np.argmax(D)
                    
                    # Extract the longest block
                    xline2 = xline2[F[L]:F[L] + M + 1]  # The longest block
                    yline2 = yline2[F[L]:F[L] + M + 1]
                    # print('xline2 after longest block = ', xline2)
                    # print('yline2 after longest block = ', yline2)
                
                # Calculate line parameters
                A3 = (yline2[-1] - yline2[0]) / (xline2[-1] - xline2[0])  # slope
                B3 = yline2[-1] - xline2[-1] * A3  # intercept

                # Calculate differences
                diffy = abs(yline2[-1] - yline2[0])
                diffx = abs(xline2[-1] - xline2[0])

                if diffx >= diffy:
                    # Create x coordinates and calculate corresponding y
                    xline3 = np.arange(int(min(xline2)), int(max(xline2)) + 1)
                    yline3 = np.round(A3 * xline3 + B3).astype(int)
                else:
                    # Create y coordinates and calculate corresponding x
                    yline3 = np.arange(int(min(yline2)), int(max(yline2)) + 1)
                    xline3 = np.round((yline3 - B3) / A3).astype(int)

                # Convert coordinates and creek to numpy arrays only if needed
                if not isinstance(xline, np.ndarray):
                    xline = np.array(xline, dtype=int)
                if not isinstance(yline, np.ndarray):
                    yline = np.array(yline, dtype=int)
                if not isinstance(creek, np.ndarray):
                    creek = np.array(creek)

                # Get elevation values directly using array indexing
                z_profile = creek[yline2.astype(int), xline2.astype(int)]
                print('z_profile = ', z_profile)

                # Remove any NaN indices if needed
                valid_indices = ~np.isnan(xline2) & ~np.isnan(yline2)
                xline2 = xline2[valid_indices]
                yline2 = yline2[valid_indices]
                # print('xline2 after remove nan = ', xline2)
                # print('yline2 after remove nan = ', yline2)
                
                if len(xline2) > 1:
                    # Convert coordinates and creek to numpy arrays only if needed
                    if not isinstance(xline, np.ndarray):
                        xline = np.array(xline, dtype=int)
                    if not isinstance(yline, np.ndarray):
                        yline = np.array(yline, dtype=int)
                    if not isinstance(creek, np.ndarray):
                        creek = np.array(creek)
                        
                    # Get elevation profile
                    z_profile = creek[yline2.astype(int), xline2.astype(int)]
                    # print('z_profile = ', z_profile)
                    
                    if len(z_profile) > 1 and np.sum(~np.isnan(z_profile)) >= 2:
                        print("Current indices:", order, iter)
                        print("WIDTH shape:", WIDTH.shape)
                        print("num_segments:", num_segments)
                        print("Current len(xline2):", len(xline2))

                        # Interpolate NaN values
                        bd = np.isnan(z_profile) # true where z_profile has a NaN
                        # Get indices of good and bad data
                        gd = np.where(~bd)[0]  # indices where we DON'T have NaNs
                        bd_indices = np.where(bd)[0]  # indices where we DO have NaNs
                        if len(gd) >= 2:  # Need at least 2 points for interpolation
                            # Create interpolation function using good data with boundary handling
                            interp_func = interpolate.interp1d(gd, z_profile[gd], 
                                                            bounds_error=False,   # Don't raise error for out of bounds
                                                            fill_value='extrapolate')  # Extrapolate beyond bounds
                            # Fill in the NaN values with interpolated values
                            z_profile[bd] = interp_func(bd_indices)
                        elif len(gd) == 1:
                            # If only one good value, fill all NaNs with that value
                            z_profile[bd] = z_profile[gd[0]]
                        else:
                            # If no good values, leave as NaN
                            pass
                        
                        # Plot if requested
                        if figview:
                            plt.figure(2)  # equivalent to figure(f2)
                            # Plot white background line
                            plt.plot(yline2, xline2, color='white', linewidth=5)
                            # Plot colored line
                            plt.plot(yline2, xline2, 
                                    color=colors[inverse_order[order]], 
                                    linewidth=3)
                            
                    if len(z_profile) > 1:  # case where cross-section is a line
                        # print("Current indices:", order, iter)
                        # print("WIDTH shape:", WIDTH.shape)
                        # print("num_segments:", num_segments)
                        # print("Current len(xline2):", len(xline2))

                        # Debug prints
                        print(f"Trying to access WIDTH[{order}, {iter}]")
                        print(f"WIDTH shape is {WIDTH.shape}")
                        # print(f'WIDTH[{order}, {iter}] = ', WIDTH[order,iter])
                        
                        # Add bounds check
                        if 0 <= order < WIDTH.shape[0] and 0 <= iter < WIDTH.shape[1]:
                            # Calculate width using endpoints
                            width_val = np.sqrt((xline2[-1] - xline2[0])**2 + (yline2[-1] - yline2[0])**2)
                            WIDTH[order, iter] = width_val  # store calculated width
                            print('width_val = ', width_val)
                            
                            # Calculate and store depth
                            depth_val = np.nanmax(z_profile) - np.nanmin(z_profile)
                            print('depth_val = ', depth_val)
                            DEPTH[order, iter] = depth_val
                        else:
                            print(f"Warning: Indices {order}, {iter} out of bounds for shape {WIDTH.shape}")
                        
                        # print(f'after width_val calc, WIDTH[{order}, {iter}] = ', WIDTH[order,iter])
                        
                        # Get start and end points and their elevations
                        startpt = np.array([yline2[0], xline2[0]])
                        Zstart = creek[int(startpt[0]), int(startpt[1])]
                        
                        endpt = np.array([yline2[-1], xline2[-1]])
                        Zend = creek[int(endpt[0]), int(endpt[1])]
                        
                        meanheight = np.mean([Zstart, Zend])
                        
                        # Calculate Z values
                        Z = np.nanmax(z_profile) - z_profile
                        
                        # Find longest continuous section
                        yline2_padded = np.concatenate(([np.nan], yline2, [np.nan]))
                        F = np.where(np.isnan(yline2_padded))[0]
                        D = np.diff(F) - 2
                        
                        if len(D) > 0:
                            M = np.max(D)
                            L = np.argmax(D)
                            
                            # Extract longest block
                            xline2 = xline2[F[L]:F[L] + M + 1]  # The longest block
                            yline2 = yline2[F[L]:F[L] + M + 1]
                            
                            # Calculate line parameters
                            A3 = (yline2[-1] - yline2[0]) / (xline2[-1] - xline2[0])
                            B3 = yline2[-1] - xline2[-1] * A3
                            
                            # Calculate differences
                            diffy = abs(yline2[-1] - yline2[0])
                            diffx = abs(xline2[-1] - xline2[0])
                        
                        if diffx >= diffy:
                            # Create x coordinates and calculate corresponding y
                            xline3 = np.arange(int(min(xline2)), int(max(xline2)) + 1)
                            yline3 = np.round(A3 * xline3 + B3).astype(int)
                        else:
                            # Create y coordinates and calculate corresponding x
                            yline3 = np.arange(int(min(yline2)), int(max(yline2)) + 1)
                            xline3 = np.round((yline3 - B3) / A3).astype(int)

                        # Calculate radii
                        r = np.sqrt(xline3**2 + yline3**2)
                        rnorm = np.abs(r - np.max(r))
                        rnorm[rnorm == 0] = 1
                        r = rnorm

                        print("Z shape:", Z.shape)
                        print("r shape:", r.shape)

                        # Create mask first
                        mask_nonzero = r != 0
                        print("mask_nonzero shape:", mask_nonzero.shape)

                        # Then modify the filtering code to ensure dimensions match:
                        if len(Z) == len(r):  # Only proceed if dimensions match
                            # Remove zeros
                            Z = Z[mask_nonzero]
                            r = r[mask_nonzero]

                            # Remove NaNs from both Z and r
                            mask_valid = ~np.isnan(Z)
                            r = r[mask_valid]
                            Z = Z[mask_valid]

                            # Remove NaNs from z_profile
                            z_profile = z_profile[~np.isnan(z_profile)]
                            print('z_profile after nan removed = ', z_profile)
                        else:
                            print(f"Warning: Z and r have different lengths. Z: {len(Z)}, r: {len(r)}")
                            print('Z = ', Z)
                            print('r = ', r)
                            # print('xline = ', xline)
                            # print('yline = ', yline)
                            # print('xline2 = ', xline2)
                            # print('yline2 = ', yline2)
                            # print('xline3 = ', xline3)
                            # print('yline3 = ', yline3)

                        # # Remove zeros
                        # mask_nonzero = r != 0
                        # Z = Z[mask_nonzero]
                        # r = r[mask_nonzero]

                        # # Remove NaNs from both Z and r
                        # mask_valid = ~np.isnan(Z)
                        # r = r[mask_valid]
                        # Z = Z[mask_valid]

                        # # Remove NaNs from z_profile
                        # z_profile = z_profile[~np.isnan(z_profile)]

                        print(f'before if len(Z) == 0, WIDTH[{order}, {iter}] = ', WIDTH[order,iter])
                        if len(Z) == 0 or np.all(np.isnan(Z)):
                            # If Z is empty or all NaN, set measurements to NaN
                            AREA[order, iter] = np.nan
                            DEPTH[order, iter] = np.nan
                            WIDTH[order, iter] = np.nan

                        elif len(Z) == 1:
                            # If Z has only one point, use special values
                            AREA[order, iter] = 999   # AREA is that of a semi-circle
                            WIDTH[order, iter] = 999  # width = 1 pixel = 1 meter
                            DEPTH[order, iter] = 999

                        else:
                            # Create top line and calculate AREA
                            Ztop = np.linspace(z_profile[0], z_profile[-1], len(z_profile))
                            Ztop = Ztop.reshape(-1, 1)  # equivalent to Ztop'
                            
                            # Calculate AREA ignoring NaN values
                            area_val = np.nansum(Ztop - z_profile)
                            
                            # Calculate Z differences
                            Z = Ztop - z_profile
                            
                            # Create x-axis values for plotting
                            lval = np.arange(1, len(Z) + 1)
                            
                            if figview:
                                plt.figure(1)
                                # Plot and set legend visibility off
                                line = plt.plot(lval, -Z, linewidth=2, color=colors[inverse_order[order]])
                                # Hide from legend
                                plt.setp(line, label="_nolegend_")
                            
                            AREA[order, iter] = area_val
                        print(f'after if len(Z) == 0 all statements, WIDTH[{order}, {iter}] = ', WIDTH[order,iter])

                        # Increment iterator
                        iter += 1
    
                    elif len(z_profile) == 1:  # case where cross-section is a point
                        print("Current indices:", order, iter)
                        print("WIDTH shape:", WIDTH.shape)
                        print("num_segments:", num_segments)
                        print("Current len(xline2):", len(xline2))
                        
                        # Find middle point
                        mid_idx = round(len(xline2) / 2)  # equivalent to round(mean(length()))
                        midsegx = xline2[mid_idx]
                        midsegy = yline2[mid_idx]
                        
                        # Define neighboring points
                        lowx = midsegx - 1
                        highx = midsegx + 1
                        lowy = midsegy - 1
                        highy = midsegy + 1
                        
                        # Set special values
                        AREA[order, iter] = 999    # area is that of a semi-circle
                        WIDTH[order, iter] = 999   # width = 1 pixel = 1 meter
                        DEPTH[order, iter] = 999
                        iter += 1

                    else:  # case where there is no cross-section
                        # Set all measurements to NaN
                        AREA[order, iter] = np.nan
                        DEPTH[order, iter] = np.nan
                        WIDTH[order, iter] = np.nan
                        iter += 1
                else:
                    AREA[order, iter] = np.nan
                    DEPTH[order, iter] = np.nan
                    WIDTH[order, iter] = np.nan
                    iter += 1
            else: # the whole cross-section is NaN
                AREA[order, iter] = np.nan
                DEPTH[order, iter] = np.nan
                WIDTH[order, iter] = np.nan
                iter += 1
    
    # Final formatting - Ensure the correct dimensions

    # Remove columns with all NaN values
    valid_cols = ~np.all(np.isnan(AREA), axis=0)
    print('valid_cols = ')
    WIDTH = WIDTH[:, valid_cols]
    DEPTH = DEPTH[:, valid_cols]
    AREA = AREA[:, valid_cols]
    
    # Replace 999 values with mean of non-999 values in the same row, only if 999s exist
    for i in range(AREA.shape[0]):  # Loop through each row
        # Handle WIDTH if 999s exist
        if 999 in WIDTH[i, :]:
            W = WIDTH[i, :].copy()
            W2 = W.copy() 
            W2[W2 == 999] = np.nan  # Replace 999s with NaN
            W[W == 999] = np.nanmean(W2)  # Replace 999s with mean of non-999 values
            WIDTH[i, :] = W
        
        # Handle DEPTH if 999s exist  
        if 999 in DEPTH[i, :]:
            D = DEPTH[i, :].copy()
            D2 = D.copy()
            D2[D2 == 999] = np.nan
            D[D == 999] = np.nanmean(D2)
            DEPTH[i, :] = D
            
        # Handle AREA if 999s exist
        if 999 in AREA[i, :]:
            A = AREA[i, :].copy()
            A2 = A.copy()
            A2[A2 == 999] = np.nan
            A[A == 999] = np.nanmean(A2)
            AREA[i, :] = A
    
    return WIDTH, DEPTH, AREA