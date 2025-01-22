import numpy as np
from skimage.morphology import skeletonize
from scipy import ndimage # type: ignore
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt

def analyze_skeleton(skeleton,Z):
    # Define the 10 branch point kernels
    branch_kernels = [
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
        np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]]),
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
        np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
        np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
        np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
        np.array([[1, 0, 1], [0, 0, 0], [0, 1, 0]]),
        np.array([[0, 0, 1], [1, 0, 0], [0, 0, 1]]),
        np.array([[0, 1, 0], [0, 0, 0], [1, 0, 1]]),
        np.array([[1, 0, 0], [0, 0, 1], [1, 0, 0]])
    ]

    # Define the neighbor check kernels
    neighbor_check_kernels = [
        np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]]),
        np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]]),
        np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]]),
        np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]]),
        np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]]),
        np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]),
        np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1]]),
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]]),
        np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])
    ]

    # Define a 2x2 kernel
    square_kernel = np.array([[1, 1],
                              [1, 1]])

    # Identify branch points using the 10 kernels
    B = np.zeros_like(skeleton, dtype=bool)
    for kernel in branch_kernels:
        conv_result = ndimage.convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
        B |= (conv_result >= 3) & skeleton

    # Exclude points that match the neighbor check kernels
    for check_kernel in neighbor_check_kernels:
        check_result = ndimage.convolve(skeleton.astype(int), check_kernel, mode='constant', cval=0)
        B &= ~((check_result == 3) & skeleton)

    # Find locations of 2x2 squares and take the pixel with lowest Z value as branch point
    square_result = ndimage.convolve(skeleton.astype(int), square_kernel, mode='constant', cval=0)
    square_locations = (square_result == 4)
    square_indices = np.argwhere(square_locations)

    for top_left in square_indices:
        i, j = top_left
        square_pixels = [(i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)]
        Z_values = [Z[p] for p in square_pixels]
        min_pixel_index = np.argmin(Z_values)
        branch_pixel = square_pixels[min_pixel_index]
        B[branch_pixel] = True
    
    # Identify endpoints
    end_kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]])
    E_neighbor_count = ndimage.convolve(skeleton.astype(int), end_kernel, mode='constant', cval=0) 
    # E = (E_neighbor_count == 1) & skeleton  # Endpoints have exactly 1 neighbor
    E = ((E_neighbor_count == 1) | (E_neighbor_count == 0)) & skeleton # Endpoints have exactly 1 neighbor or are isolated points

    # Combine branch points and endpoints
    PTS = B | E

    return B, E, PTS

def calculate_chessboard_distance(mask, start_y, start_x):
    """
    Calculate chessboard distance (8-connected) from start point
    
    Parameters:
    -----------
    mask : ndarray
        Binary mask of valid regions
    start_y, start_x : int
        Starting coordinates
    
    Returns:
    --------
    distances : ndarray
        Array of distances from start point
    """
    import networkx as nx  # type: ignore # Local import to avoid naming conflicts
    
    G = nx.Graph()
    
    # Get all valid points
    y_coords, x_coords = np.where(mask)
    points = list(zip(y_coords, x_coords))
    
    # Add nodes
    for p in points:
        G.add_node(p)
    
    # Add edges (8-connected)
    for y, x in points:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                newy, newx = y + dy, x + dx
                if (newy, newx) in G.nodes():
                    G.add_edge((y, x), (newy, newx), weight=1)
    
    # Calculate distances
    distances = np.full_like(mask, np.nan, dtype=float)
    try:
        if (start_y, start_x) in G.nodes():
            path_lengths = nx.single_source_dijkstra_path_length(
                G, (start_y, start_x), weight='weight')
            for (y, x), dist in path_lengths.items():
                distances[y, x] = dist
    except Exception as e:
        print(f"Error in distance calculation: {str(e)}")
        print(f"Start point: ({start_y}, {start_x})")
        print(f"Graph nodes: {len(G.nodes())}")
    
    return distances

def calculate_geodesic_distance(mask, start_y, start_x):
    """Calculate geodesic distance using graph approach"""
    # global nx  # Add this line to explicitly use the global nx import
    import networkx as nx # type: ignore

    G = nx.Graph()
    y_coords, x_coords = np.where(mask)
    points = list(zip(y_coords, x_coords))
    
    for p in points:
        G.add_node(p)
    
    for y, x in points:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                num_y, num_x = y + dy, x + dx
                if (num_y, num_x) in G.nodes():
                    G.add_edge((y, x), (num_y, num_x), weight=1)
    
    distances = np.full_like(mask, np.nan, dtype=float)
    # # # Make sure start_y and start_x are integers, not numpy types
    # start_y = int(start_y) if not isinstance(start_y, (int, float)) else int(float(start_y))
    # start_x = int(start_x) if not isinstance(start_x, (int, float)) else int(float(start_x))
    # print('      start_y, start_x = ', (start_y, start_x))
    if (start_y, start_x) in G.nodes():
    # if (int(start_y), int(start_x)) in G.nodes():
        path_lengths = nx.single_source_dijkstra_path_length(
            G, (start_y, start_x), weight='weight')
        for (y, x), dist in path_lengths.items():
            distances[y, x] = dist
    
    return distances

# Simple implementation of imregionalmin
def find_regional_minima(D):
    from scipy.ndimage import minimum_filter
    min_filtered = minimum_filter(D, size=3)
    return (D == min_filtered) & (D != np.inf)

def process_junction_angles(skeleton, creek_mask, Z, IDXBRANCH, SINUOUSLENGTH, SEGMENTS, fig_view=False):
    """
    Calculate junction angles in creek skeleton.
    
    Args:
        skeleton: Binary array of creek skeleton
        creek_mask: Binary mask of creek
        IDXBRANCH: Array of branch point indices
        SINUOUSLENGTH: Array of sinuous lengths
        SEGMENTS: Number of SEGMENTS per order
        fig_view: Boolean to control visualization
    
    Returns:
        ANGLEORDER: Array of minimum angles at junction points
    """

    # Initialize variables
    # Initialize a table of the same size as IDXBRANCH, 
    # to associate all angles with the right creek order
    ANGLEORDER = np.zeros_like(IDXBRANCH, dtype=float)
    
    # Get branch points and endpoints
    B, _, _ = analyze_skeleton(skeleton, Z)
    
    # Setup visualization if needed
    if fig_view:
        plt.figure(figsize=(12, 8))
        creek_mask_pic = np.zeros_like(creek_mask)
        creek_mask_pic[creek_mask == 0] = 1
        creek_mask_pic[creek_mask == 1] = 0
        plt.imshow(creek_mask_pic)
        plt.title('Junction Angles', fontsize=18, fontweight='bold')

    # Find length of the segments used for angle measurement: 
    # 0.2* mean length of second Strahler order creeks
    mean_SINUOUSLENGTH_list = np.nansum(SINUOUSLENGTH, axis=1) / SEGMENTS
    mean_SINUOUSLENGTH = mean_SINUOUSLENGTH_list[1]  # Second order
    seg_length = round(mean_SINUOUSLENGTH * 0.2)
    
    # Process each branch point
    branch_points = np.argwhere(B)
    
    # Get the Strahler order and junction angle
    for i, B_loc in enumerate(branch_points): # Process each branch point
        VECTORS = []
        LENGTHS = []
        ANGLES = []
        
        # Get points at seg_length distance using geodesic distance
        # Calculate distances from this branch point
        distances = calculate_chessboard_distance(skeleton, B_loc[0], B_loc[1])
        # Find points that are at seg_length distance (with small tolerance for floating point)
        ALLIDX = np.where(np.abs(distances - seg_length) < 0.5)
        nb = 0  # no vector drawn yet

        if len(ALLIDX[0]) < 3:  # Check if we have at least 3 points
            continue  # Skip this branch point if not enough points
        else:
            skeleton_temp = skeleton.copy()  # Make a copy that will be shortened
        
        # iteratively find angles
        while nb < 3:
            distances = calculate_chessboard_distance(skeleton_temp, B_loc[0], B_loc[1])
            # Find points that are at seg_length distance (with small tolerance for floating point)
            ALLIDX = np.where(np.abs(distances - seg_length) < 0.5)

            if len(ALLIDX[0]) == 0: # PROBLEM: less than three segments of adequate length available for this branch point
                nb = 999
                break
                
            pt = (ALLIDX[0][0], ALLIDX[1][0]) # First available segment
            y0, x0 = B_loc[0], B_loc[1]
            y1, x1 = pt
            vect1 = np.array([y1 - y0, x1 - x0])
            # Add to vectors list (prepend since MATLAB used [vect1;VECT])
            VECTORS = [vect1] + VECTORS if VECTORS else [vect1]
            length1 = np.sqrt(np.sum(vect1**2))
            LENGTHS = [length1] + LENGTHS if LENGTHS else [length1]

            if fig_view:  # Python uses True instead of 1 for boolean
                plt.quiver(x0, y0,           # Start point
                        vect1[1], vect1[0],  # dx, dy (note index swap for x,y order)
                        color='g',           # Green color ([0, 0.9, 0] simplified to 'g')
                        linewidth=3,
                        angles='xy',         # Important: This preserves the correct orientation
                        scale=1,            # Scale factor for arrow size
                        scale_units='xy')    # Use data units for scaling
                
            # Calculate geodesic distances from both points
            D1 = calculate_geodesic_distance(skeleton_temp, y0, x0)  # From branch point
            D2 = calculate_geodesic_distance(skeleton_temp, y1, x1)  # From endpoint
            # Add distances and round like MATLAB
            D = D1 + D2
            D = np.round(D * 32) / 32
            # Replace NaN with infinity
            D[np.isnan(D)] = np.inf

            # Find regional minima (points along the shortest path)
            paths = find_regional_minima(D)
            # Block the path in skeleton_temp
            skeleton_temp[paths] = False
            skeleton_temp[y0, x0] = True  # Restore branch point
            nb += 1  # Increment vector count
        
        if nb == 999:  # This was the error case from earlier when not enough segments were found
            continue
        else:
            # Get the three vectors (VECT was storing them in reverse order)
            vect1 = VECTORS[0]  # First vector added
            vect2 = VECTORS[1]  # Second vector added
            vect3 = VECTORS[2]  # Third vector added

            # Calculate dot products between vector pairs
            dp1 = np.dot(vect1, vect2)
            dp2 = np.dot(vect1, vect3)
            dp3 = np.dot(vect2, vect3)

            # Calculate vector lengths using Euclidean norm
            length1 = np.sqrt(np.sum(vect1**2))
            length2 = np.sqrt(np.sum(vect2**2))
            length3 = np.sqrt(np.sum(vect3**2))

            # Extract x,y components from vectors
            y1, x1 = vect1[0], vect1[1]
            y2, x2 = vect2[0], vect2[1]
            y3, x3 = vect3[0], vect3[1]

            # Calculate angles in degrees
            # Note: np.arctan2() returns radians, so convert to degrees
            angle1 = np.abs(np.degrees(np.arctan2(y1, x1) - np.arctan2(y2, x2)))
            angle2 = np.abs(np.degrees(np.arctan2(y1, x1) - np.arctan2(y3, x3)))
            angle3 = np.abs(np.degrees(np.arctan2(y2, x2) - np.arctan2(y3, x3)))

            # Find minimum angle
            angle_min = np.min([angle1, angle2, angle3])

            # Add to angles list (prepending like before)
            ANGLES = [angle_min] + ANGLES if ANGLES else [angle_min]

            # Draw intersection point if visualization is enabled
            if fig_view:
                plt.plot(x0, y0,  # coordinates
                        color='darkred',  # [0.4, 0, 0] is a dark red
                        marker='.',
                        markersize=3,
                        linewidth=3)
                
            # Find where this branch point appears in idx_branch
            # Convert branch point to linear index for comparison
            branch_linear_idx = np.ravel_multi_index((B_loc[0], B_loc[1]), skeleton.shape)
            # Find matches in IDXBRANCH
            matches = np.where(IDXBRANCH == branch_linear_idx)
            # Save the minimum angle
            if len(matches[0]) > 0:
                for idx_row, idx_col in zip(*matches):
                    ANGLEORDER[idx_row, idx_col] = angle_min
    
    # Remove rows that are all zeros (keep rows with any non-zero values)
    ANGLEORDER = ANGLEORDER[np.any(ANGLEORDER, axis=1), :]
    # Remove columns that are all zeros (keep columns with any non-zero values)
    ANGLEORDER = ANGLEORDER[:, np.any(ANGLEORDER, axis=0)]
    
    if fig_view:
        plt.show()
    
    return ANGLEORDER

def process_junction_angles_diagnostic(skeleton, creek_mask, Z, IDXBRANCH, SINUOUSLENGTH, SEGMENTS, fig_view=False):
    """
    Calculate junction angles in creek skeleton.
    
    Args:
        skeleton: Binary array of creek skeleton
        creek_mask: Binary mask of creek
        IDXBRANCH: Array of branch point indices
        SINUOUSLENGTH: Array of sinuous lengths
        SEGMENTS: Number of SEGMENTS per order
        fig_view: Boolean to control visualization
    
    Returns:
        ANGLEORDER: Array of minimum angles at junction points
    """
    
    # Print initial shapes and non-zero counts
    print(f"Initial IDXBRANCH shape: {IDXBRANCH.shape}")
    print(f"Number of non-zero elements in IDXBRANCH: {np.count_nonzero(IDXBRANCH)}")
    
    # Initialize variables
    # Initialize a table of the same size as IDXBRANCH, 
    # to associate all angles with the right creek order
    ANGLEORDER = np.zeros_like(IDXBRANCH, dtype=float)
    print(f"Initial ANGLEORDER shape: {ANGLEORDER.shape}")
    
    # Get branch points and endpoints
    B, _, _ = analyze_skeleton(skeleton, Z)
    print(f"Number of branch points found: {np.count_nonzero(B)}")
    
    # Setup visualization if needed
    if fig_view:
        plt.figure(figsize=(12, 8))
        creek_mask_pic = np.zeros_like(creek_mask)
        creek_mask_pic[creek_mask == 0] = 1
        creek_mask_pic[creek_mask == 1] = 0
        plt.imshow(creek_mask_pic)
        plt.title('Junction Angles', fontsize=18, fontweight='bold')

    # Find length of the segments used for angle measurement: 
    # 0.2* mean length of second Strahler order creeks
    mean_SINUOUSLENGTH_list = np.nansum(SINUOUSLENGTH, axis=1) / SEGMENTS
    mean_SINUOUSLENGTH = mean_SINUOUSLENGTH_list[1]  # Second order
    seg_length = round(mean_SINUOUSLENGTH * 0.2)
    print(f"Calculated segment length: {seg_length}")
    
    # Process each branch point
    branch_points = np.argwhere(B)
    print(f"Total branch points to process: {len(branch_points)}")
    
    processed_points = 0
    skipped_points = 0
    angles_found = 0
    
    # Get the Strahler order and junction angle
    for i, B_loc in enumerate(branch_points): # Process each branch point
        print('\n i = ', i)
        VECTORS = []
        LENGTHS = []
        ANGLES = []
        
        # Get points at seg_length distance using geodesic distance
        # Calculate distances from this branch point
        distances = calculate_chessboard_distance(skeleton, B_loc[0], B_loc[1])
        # Find points that are at seg_length distance (with small tolerance for floating point)
        ALLIDX = np.where(np.abs(distances - seg_length) < 0.5)
        print("ALLIDX = ", ALLIDX)
        nb = 0  # no vector drawn yet

        if len(ALLIDX[0]) < 3:  # Check if we have at least 3 points
            print(f"Branch point {i} skipped: insufficient points at distance {seg_length}")
            print(f"Found only {len(ALLIDX[0])} points")
            skipped_points += 1
            continue  # Skip this branch point if not enough points
        else:
            skeleton_temp = skeleton.copy()  # Make a copy that will be shortened
        
        # iteratively find angles
        while nb < 3:
            distances = calculate_chessboard_distance(skeleton_temp, B_loc[0], B_loc[1])
            # Find points that are at seg_length distance (with small tolerance for floating point)
            ALLIDX = np.where(np.abs(distances - seg_length) < 0.5)

            if len(ALLIDX[0]) == 0: # PROBLEM: less than three segments of adequate length available for this branch point
                print(f"Branch point {i}: Unable to find 3 segments, found only {nb}")
                nb = 999
                break
                
            pt = (ALLIDX[0][0], ALLIDX[1][0]) # First available segment
            y0, x0 = B_loc[0], B_loc[1]
            y1, x1 = pt
            vect1 = np.array([y1 - y0, x1 - x0])
            # Add to vectors list (prepend since MATLAB used [vect1;VECT])
            VECTORS = [vect1] + VECTORS if VECTORS else [vect1]
            length1 = np.sqrt(np.sum(vect1**2))
            LENGTHS = [length1] + LENGTHS if LENGTHS else [length1]

            if fig_view:  # Python uses True instead of 1 for boolean
                plt.quiver(x0, y0,           # Start point
                        vect1[1], vect1[0],  # dx, dy (note index swap for x,y order)
                        color='g',           # Green color ([0, 0.9, 0] simplified to 'g')
                        linewidth=3,
                        angles='xy',         # Important: This preserves the correct orientation
                        scale=1,            # Scale factor for arrow size
                        scale_units='xy')    # Use data units for scaling
                
            # Calculate geodesic distances from both points
            D1 = calculate_geodesic_distance(skeleton_temp, y0, x0)  # From branch point
            D2 = calculate_geodesic_distance(skeleton_temp, y1, x1)  # From endpoint
            # Add distances and round like MATLAB
            D = D1 + D2
            D = np.round(D * 32) / 32
            # Replace NaN with infinity
            D[np.isnan(D)] = np.inf

            # Find regional minima (points along the shortest path)
            paths = find_regional_minima(D)
            # Block the path in skeleton_temp
            skeleton_temp[paths] = False
            skeleton_temp[y0, x0] = True  # Restore branch point
            nb += 1  # Increment vector count
        
        if nb == 999:  # This was the error case from earlier when not enough segments were found
            skipped_points += 1
            continue
        else:
            # Get the three vectors (VECT was storing them in reverse order)
            vect1 = VECTORS[0]  # First vector added
            vect2 = VECTORS[1]  # Second vector added
            vect3 = VECTORS[2]  # Third vector added

            # Calculate dot products between vector pairs
            dp1 = np.dot(vect1, vect2)
            dp2 = np.dot(vect1, vect3)
            dp3 = np.dot(vect2, vect3)

            # Calculate vector lengths using Euclidean norm
            length1 = np.sqrt(np.sum(vect1**2))
            length2 = np.sqrt(np.sum(vect2**2))
            length3 = np.sqrt(np.sum(vect3**2))

            # Extract x,y components from vectors
            y1, x1 = vect1[0], vect1[1]
            y2, x2 = vect2[0], vect2[1]
            y3, x3 = vect3[0], vect3[1]

            # Calculate angles in degrees
            # Note: np.arctan2() returns radians, so convert to degrees
            angle1 = np.abs(np.degrees(np.arctan2(y1, x1) - np.arctan2(y2, x2)))
            angle2 = np.abs(np.degrees(np.arctan2(y1, x1) - np.arctan2(y3, x3)))
            angle3 = np.abs(np.degrees(np.arctan2(y2, x2) - np.arctan2(y3, x3)))

            # Find minimum angle
            angle_min = np.min([angle1, angle2, angle3])
            print(f"Branch point {i}: Minimum angle found = {angle_min}")
            angles_found += 1

            # Add to angles list (prepending like before)
            ANGLES = [angle_min] + ANGLES if ANGLES else [angle_min]

            # Draw intersection point if visualization is enabled
            if fig_view:
                plt.plot(x0, y0,  # coordinates
                        color='darkred',  # [0.4, 0, 0] is a dark red
                        marker='.',
                        markersize=3,
                        linewidth=3)
                
            # Find where this branch point appears in idx_branch
            # Convert branch point to linear index for comparison
            branch_linear_idx = np.ravel_multi_index((B_loc[0], B_loc[1]), skeleton.shape)
            # Find matches in IDXBRANCH
            matches = np.where(IDXBRANCH == branch_linear_idx)
            print(f"Found {len(matches[0])} matches for branch point {i}")
            # Save the minimum angle
            if len(matches[0]) > 0:
                for idx_row, idx_col in zip(*matches):
                    ANGLEORDER[idx_row, idx_col] = angle_min
            processed_points += 1
            
        
    
    print(f"\nProcessing Summary:")
    print(f"Total branch points: {len(branch_points)}")
    print(f"Successfully processed points: {processed_points}")
    print(f"Skipped points: {skipped_points}")
    print(f"Angles found: {angles_found}")
    
    # Print state before final filtering
    print(f"\nANGLEORDER before filtering:")
    print(f"Shape: {ANGLEORDER.shape}")
    print(f"Non-zero elements: {np.count_nonzero(ANGLEORDER)}")
    
    # Remove rows that are all zeros (keep rows with any non-zero values)
    ANGLEORDER = ANGLEORDER[np.any(ANGLEORDER, axis=1), :]
    # Remove columns that are all zeros (keep columns with any non-zero values)
    ANGLEORDER = ANGLEORDER[:, np.any(ANGLEORDER, axis=0)]
    
    # Print final state
    print(f"\nFinal ANGLEORDER:")
    print(f"Shape: {ANGLEORDER.shape}")
    print(f"Non-zero elements: {np.count_nonzero(ANGLEORDER)}")
    
    if fig_view:
        plt.show()
    
    return ANGLEORDER