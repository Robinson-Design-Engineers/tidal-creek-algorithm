import numpy as np
from scipy import ndimage
from skimage import morphology, measure, segmentation, filters, draw # type: ignore
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import h5py

def reconnect_diagnostic(creek_mask, noise_threshold, reconnection_dist, connectivity):
    # Remove noise
    creek_mask = morphology.remove_small_objects(creek_mask, noise_threshold, connectivity=connectivity)
    contours = creek_mask.copy()

    max_iterations = 1000
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        labeled, num_objects = ndimage.label(contours, structure=np.ones((3,3)))
        sizes = np.bincount(labeled.ravel())[1:]
        
        print(f"Iteration {iteration}:")
        print(f"  Number of objects: {num_objects}")
        print(f"  Sizes of objects: {sizes}")
        
        if num_objects <= 1:
            print("  Only one object remains. Exiting loop.")
            break
        
        # Visualize objects
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(contours, cmap='gray')
        ax.set_title(f"Objects in Iteration {iteration}")
        
        # Find the largest object
        largest_idx = np.argmax(sizes) + 1
        largest_obj = (labeled == largest_idx)
        
        # Create a distance transform from the largest object
        dist_transform = ndimage.distance_transform_edt(~largest_obj)
        
        # Process all objects
        for idx in range(1, num_objects + 1):
            if idx == largest_idx:
                continue
            
            obj = (labeled == idx)
            obj_size = sizes[idx - 1]
            
            # Get object location
            obj_coords = np.argwhere(obj)
            obj_center = np.mean(obj_coords, axis=0)
            
            # Check circularity
            props = measure.regionprops(obj.astype(int))
            circularity = props[0].perimeter ** 2 / (4 * np.pi * props[0].area)
            
            print(f"  Object {idx}: Size = {obj_size}, Circularity = {circularity:.4f}")
            print(f"    Location: Center at ({obj_center[0]:.1f}, {obj_center[1]:.1f})")
            
            # Outline the object
            minr, minc, maxr, maxc = props[0].bbox
            rect = Rectangle((minc, minr), maxc - minc, maxr - minr,
                             fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect)
            
            # Label the object
            # ax.text(obj_center[1], obj_center[0], str(idx), color='red', 
            #         fontsize=12, ha='center', va='center')
            ax.text(minc-5, maxr-5, str(idx), color='red', 
                    fontsize=12, ha='center', va='center')
            
            # Find the minimum distance to the largest object
            min_dist = np.min(dist_transform[obj])
            print(f"    Minimum distance to largest object: {min_dist:.2f}")
            
            if circularity < 1 or min_dist > reconnection_dist or obj_size < noise_threshold:
                contours[obj] = False
                print(f"    Removed object {idx}")
            else:
                # Find the closest point on the current object to the largest object
                closest_obj_point = obj_coords[np.argmin(dist_transform[obj])]
                
                # Create a local search area
                search_radius = int(min_dist + reconnection_dist)
                local_area = np.zeros_like(largest_obj)
                rr, cc = draw.disk((closest_obj_point[0], closest_obj_point[1]), search_radius, shape=largest_obj.shape)
                local_area[rr, cc] = 1
                
                # Find the closest point on the largest object within the local area
                local_largest = largest_obj & local_area
                local_dist = ndimage.distance_transform_edt(~obj)
                local_dist[~local_largest] = np.inf
                closest_largest_point = np.unravel_index(np.argmin(local_dist), local_dist.shape)
                
                # Draw a line to connect the objects
                rr, cc = draw.line(closest_obj_point[0], closest_obj_point[1],
                                   closest_largest_point[0], closest_largest_point[1])
                contours[rr, cc] = True
                print(f"    Connected object {idx} to largest object")
                print(f"    Connection: From ({closest_obj_point[0]}, {closest_obj_point[1]}) to ({closest_largest_point[0]}, {closest_largest_point[1]})")
                
                # Visualize the connection
                ax.plot([closest_obj_point[1], closest_largest_point[1]], 
                        [closest_obj_point[0], closest_largest_point[0]], 
                        color='yellow', linewidth=2)

        plt.show()

        # Check if the number of objects has changed
        new_labeled, new_num_objects = ndimage.label(contours, structure=np.ones((3,3)))
        if new_num_objects == num_objects:
            print("  Warning: No change in number of objects. May be stuck.")
            break

        print(f"  Number of objects after operation: {new_num_objects}")
        print()

    if iteration == max_iterations:
        print("Warning: Maximum iterations reached in reconnect function.")

    return contours

def reconnect(creek_mask, noise_threshold, reconnection_dist, connectivity):
    # Remove noise
    creek_mask = morphology.remove_small_objects(creek_mask, noise_threshold, connectivity=connectivity)
    contours = creek_mask.copy()

    max_iterations = 1000
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        labeled, num_objects = ndimage.label(contours, structure=np.ones((3,3)))
        sizes = np.bincount(labeled.ravel())[1:]
        
        if num_objects <= 1:
            break
        
        # Find the largest object
        largest_idx = np.argmax(sizes) + 1
        largest_obj = (labeled == largest_idx)
        
        # Create a distance transform from the largest object
        dist_transform = ndimage.distance_transform_edt(~largest_obj)
        
        # Process all objects
        for idx in range(1, num_objects + 1):
            if idx == largest_idx:
                continue
            
            obj = (labeled == idx)
            obj_size = sizes[idx - 1]
            
            # Get object location
            obj_coords = np.argwhere(obj)
            obj_center = np.mean(obj_coords, axis=0)
            
            # Check circularity
            props = measure.regionprops(obj.astype(int))
            circularity = props[0].perimeter ** 2 / (4 * np.pi * props[0].area)
            
            # Find the minimum distance to the largest object
            min_dist = np.min(dist_transform[obj])
            
            if circularity < 1 or min_dist > reconnection_dist or obj_size < noise_threshold:
                contours[obj] = False
            else:
                # Find the closest point on the current object to the largest object
                closest_obj_point = obj_coords[np.argmin(dist_transform[obj])]
                
                # Create a local search area
                search_radius = int(min_dist + reconnection_dist)
                local_area = np.zeros_like(largest_obj)
                rr, cc = draw.disk((closest_obj_point[0], closest_obj_point[1]), search_radius, shape=largest_obj.shape)
                local_area[rr, cc] = 1
                
                # Find the closest point on the largest object within the local area
                local_largest = largest_obj & local_area
                local_dist = ndimage.distance_transform_edt(~obj)
                local_dist[~local_largest] = np.inf
                closest_largest_point = np.unravel_index(np.argmin(local_dist), local_dist.shape)
                
                # Draw a line to connect the objects
                rr, cc = draw.line(closest_obj_point[0], closest_obj_point[1],
                                   closest_largest_point[0], closest_largest_point[1])
                contours[rr, cc] = True

    if iteration == max_iterations:
        print("Warning: Maximum iterations reached in reconnect function.")

    return contours

def bwareafilt(binary_image, size_range, connectivity=2):
    """
    Replicates MATLAB's bwareafilt function in Python.
    
    Parameters:
    binary_image (ndarray): Input binary image
    size_range (list or tuple): Two-element list specifying [min_size, max_size]
    connectivity (int): Connectivity for connected components (default is 2 for 8-connectivity)
    
    Returns:
    ndarray: Filtered binary image
    """
    # Ensure the image is binary
    binary_image = binary_image.astype(bool)
    
    # Remove small objects
    filtered = morphology.remove_small_objects(binary_image, min_size=size_range[0], connectivity=connectivity)
    
    # Invert the image to remove large objects (which are now small holes)
    inverted = ~filtered
    max_hole_size = binary_image.size - size_range[1]
    filtered_inverted = morphology.remove_small_objects(inverted, min_size=max_hole_size, connectivity=connectivity)
    
    # Invert back
    result = ~filtered_inverted
    
    return result

def fill_small_holes(creekmask, hole_size_infill):
    """
    Fills small holes in the binary image.
    
    Parameters:
    creekmask (ndarray): Input binary image
    hole_size_infill (int): Maximum size of holes to fill
    
    Returns:
    ndarray: Binary image with small holes filled
    """
    # Fill all holes
    filled = ndimage.binary_fill_holes(creekmask)
    
    # Identify all holes
    holes = filled & ~creekmask
    
    # Remove small holes
    small_holes = morphology.remove_small_objects(holes, min_size=hole_size_infill, connectivity=2)
    
    # Add small holes back to the original image
    result = creekmask | small_holes
    
    return result

def morphological_operations(creekmask, smoothing):
    """Performs a gentle sequence of morphological operations."""
    # Skip spur removal and diagonal removal
    
    # Create disk-shaped structuring element for smoothing
    se = morphology.disk(smoothing)
    
    # Perform closing operation for gentle smoothing
    creekmask = morphology.binary_closing(creekmask, se)
    
    # # Optional: Very mild opening to remove single-pixel protrusions
    # small_se = morphology.disk(1)
    # creekmask = morphology.binary_opening(creekmask, small_se)
    
    return creekmask

def repair_diagnostic(creekmask, filtersmall1, filterlarge1, connectivity, smoothing, filtersmall2, filterlarge2, hole_size_infill):
    stages = []
    titles = []
    
    # Store original
    stages.append(creekmask.copy())
    titles.append("Original")

    # First bwareafilt
    creekmask = bwareafilt(creekmask, [filtersmall1, filterlarge1], connectivity)
    stages.append(creekmask.copy())
    titles.append(f"After first bwareafilt ({filtersmall1}, {filterlarge1})")

    # Morphological operations
    creekmask = morphological_operations(creekmask, smoothing)
    stages.append(creekmask.copy())
    titles.append(f"After morphological operations (smoothing={smoothing})")

    # Second bwareafilt
    creekmask = bwareafilt(creekmask, [filtersmall2, filterlarge2], connectivity)
    stages.append(creekmask.copy())
    titles.append(f"After second bwareafilt ({filtersmall2}, {filterlarge2})")

    # Fill small holes
    creekmask = fill_small_holes(creekmask, hole_size_infill)
    stages.append(creekmask.copy())
    titles.append(f"After fill_small_holes (hole_size_infill={hole_size_infill})")

    # Visualize all stages
    n_stages = len(stages)
    n_cols = 3  # You can adjust this for your preferred layout
    n_rows = (n_stages + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axs = axs.ravel()
    
    for i, (stage, title) in enumerate(zip(stages, titles)):
        axs[i].imshow(stage, cmap='gray')
        axs[i].set_title(title)
        axs[i].axis('off')
    
    for i in range(n_stages, len(axs)):
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.show()

    return creekmask

def repair(creekmask, filtersmall1, filterlarge1, connectivity, smoothing, filtersmall2, filterlarge2, hole_size_infill):
    # First bwareafilt
    creekmask = bwareafilt(creekmask, [filtersmall1, filterlarge1], connectivity)

    # Morphological operations
    creekmask = morphological_operations(creekmask, smoothing)

    # Second bwareafilt
    creekmask = bwareafilt(creekmask, [filtersmall2, filterlarge2], connectivity)

    # Fill small holes
    creekmask = fill_small_holes(creekmask, hole_size_infill)
    
    return creekmask

def process_creek_mask_diagnostic(creek_mask, reconnect_flag, noise_threshold, reconnection_dist, connectivity,
                                  filtersmall1, filterlarge1, smoothing, filtersmall2, filterlarge2, hole_size_infill):
    if reconnect_flag:
        creek_mask = reconnect_diagnostic(creek_mask, noise_threshold, reconnection_dist, connectivity)
    
    repaired_mask = repair_diagnostic(creek_mask, filtersmall1, filterlarge1, connectivity, smoothing, filtersmall2, filterlarge2, hole_size_infill)
    
    return repaired_mask

def process_creek_mask(creek_mask, reconnect_flag, noise_threshold, reconnection_dist, connectivity,
                       filter_small1, filter_large1, smoothing, filter_small2, filter_large2, hole_size_infill):
    if reconnect_flag:
        creek_mask = reconnect(creek_mask, noise_threshold, reconnection_dist, connectivity)
        figure_reconnected_creek_mask(creek_mask)

    creek_mask = repair(creek_mask, filter_small1, filter_large1, connectivity, smoothing, filter_small2, filter_large2, hole_size_infill)
    figure_repaired_creek_mask(creek_mask)

    return creek_mask

def figure_reconnected_creek_mask(creek_mask):
    plt.figure()
    plt.imshow(creek_mask, cmap='gray')
    plt.title('Reconnected creek area mask')
    # plt.axis('off')
    plt.show()

def figure_repaired_creek_mask(creek_mask):
    plt.figure()
    plt.imshow(creek_mask, cmap='gray')
    plt.title('Repaired creek area mask')
    # plt.axis('off')
    plt.show()

def save_creek_mask_h5(creek_mask, filename):
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset("creek_mask", data=creek_mask)

def load_creek_mask_h5(filename):
    with h5py.File(filename, 'r') as hf:
        creek_mask = hf['creek_mask'][:]
    return creek_mask

# Example usage with adjusted parameters:
# initial_creek_mask = load_creek_mask_h5('initial_creek_mask.h5')
# processed_mask = process_creek_mask(initial_creek_mask, 
#                                     reconnect_flag=True, 
#                                     noise_threshold=50,
#                                     reconnection_dist=5,
#                                     connectivity=2,
#                                     filter_small1=25,
#                                     filter_large1=50,
#                                     smoothing=1,
#                                     filter_small2=35,
#                                     filter_large2=75,
#                                     hole_size_infill=100)
# save_creek_mask_h5(processed_mask, 'processed_creek_mask.h5')