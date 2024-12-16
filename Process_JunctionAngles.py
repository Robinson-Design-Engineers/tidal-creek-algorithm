import numpy as np
from skimage.morphology import skeletonize
from scipy import ndimage # type: ignore
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt

def analyze_skeleton(skeleton):
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

    # Identify branch points using the 10 kernels
    B = np.zeros_like(skeleton, dtype=bool)
    for kernel in branch_kernels:
        conv_result = ndimage.convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
        B |= (conv_result >= 3) & skeleton

    # Exclude points that match the neighbor check kernels
    for check_kernel in neighbor_check_kernels:
        check_result = ndimage.convolve(skeleton.astype(int), check_kernel, mode='constant', cval=0)
        B &= ~((check_result == 3) & skeleton)
    
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

def process_junction_angles(skeleton, creek_mask, idx_branch, sinuous_length, segments, fig_view=False):
    """
    Calculate junction angles in creek skeleton.
    
    Args:
        skeleton: Binary array of creek skeleton
        creek_mask: Binary mask of creek
        idx_branch: Array of branch point indices
        sinuous_length: Array of sinuous lengths
        segments: Number of segments per order
        fig_view: Boolean to control visualization
    
    Returns:
        angle_order: Array of minimum angles at junction points
    """
    
    # Initialize variables
    angle_order = np.zeros_like(idx_branch, dtype=float)
    
    # Get branch points and endpoints
    B, E, PTS = analyze_skeleton(skeleton)
    
    # Calculate segment length for angle measurement
    mean_sinuous_length_list = np.nansum(sinuous_length, axis=1) / segments
    mean_sinuous_length = mean_sinuous_length_list[1]  # Second order
    seg_length = round(mean_sinuous_length * 0.2)
    
    # Setup visualization if needed
    if fig_view:
        plt.figure(figsize=(12, 8))
        creek_mask_pic = np.zeros_like(creek_mask)
        creek_mask_pic[creek_mask == 0] = 1
        plt.imshow(creek_mask_pic)
        plt.title('Junction Angles', fontsize=18, fontweight='bold')
    
    # Process each branch point
    for i, B_loc in enumerate(np.where(B)[0]):
        vectors = []
        lengths = []
        
        # Get points at seg_length distance using geodesic distance
        # Note: This is a simplified version of bwdistgeodesic
        dist_map = distance_transform_edt(skeleton)
        potential_points = np.where(np.abs(dist_map - seg_length) < 0.5)
        
        if len(potential_points[0]) < 3:
            continue
            
        skeleton_temp = skeleton.copy()
        nb = 0  # Number of vectors found
        
        while nb < 3:
            if len(potential_points[0]) == 0:
                break
                
            # Get first available point
            y1, x1 = potential_points[0][0], potential_points[1][0]
            y0, x0 = np.unravel_index(B_loc, skeleton.shape)
            
            # Calculate vector
            vect = np.array([y1 - y0, x1 - x0])
            vectors.append(vect)
            length = np.sqrt(np.sum(vect**2))
            lengths.append(length)
            
            if fig_view:
                plt.quiver(x0, y0, vect[1], vect[0], color='g', scale=20)
                
            # Update skeleton_temp to block this path
            # Simplified version of the MATLAB path blocking
            skeleton_temp[y1, x1] = False
            
            nb += 1
            
        if nb == 3:
            # Calculate angles between vectors
            vectors = np.array(vectors)
            angles = []
            
            for j in range(3):
                for k in range(j+1, 3):
                    v1, v2 = vectors[j], vectors[k]
                    angle = np.abs(np.degrees(np.arctan2(v1[0], v1[1]) - 
                                            np.arctan2(v2[0], v2[1])))
                    angles.append(angle)
            
            min_angle = min(angles)
            
            # Save angle in the output array
            match_idx = np.where(idx_branch == B_loc)[0]
            angle_order[match_idx] = min_angle
            
            if fig_view:
                plt.plot(x0, y0, 'r.')
    
    if fig_view:
        plt.show()
    
    # Clean up results
    angle_order = angle_order[angle_order.nonzero()]
    
    return angle_order