import numpy as np
from skimage import morphology, measure, feature, draw # type: ignore
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

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
    E = (E_neighbor_count == 1) & skeleton  # Endpoints have exactly 1 neighbor

    # Combine branch points and endpoints
    PTS = B | E

    return B, E, PTS

def figure_creek_skeleton(creekmask, skeleton):
    # Thicken the skeleton for display
    skeletonpic = morphology.dilation(skeleton, morphology.square(3))
    skeletonpic2 = np.logical_not(skeletonpic)

    B, E, _ = analyze_skeleton(skeleton)
    
    # print(f"Number of branch points: {np.sum(B)}")
    # print(f"Number of endpoints: {np.sum(E)}")

    # # Debugging: Visualize intermediate results
    # fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    # axes[0, 0].imshow(skeleton, cmap='gray')
    # axes[0, 0].set_title('Original Skeleton')
    # axes[0, 1].imshow(skeletonpic, cmap='gray')
    # axes[0, 1].set_title('Dilated Skeleton')
    # axes[1, 0].imshow(B, cmap='gray')
    # axes[1, 0].set_title('Branch Points')
    # axes[1, 1].imshow(E, cmap='gray')
    # axes[1, 1].set_title('Endpoints')
    # plt.tight_layout()
    # plt.show()

    # Main visualization
    plt.figure(figsize=(8, 8))
    plt.imshow(skeletonpic2, cmap='gray', origin='upper')

    # Plot branch points and endpoints
    y_branch, x_branch = np.where(B)
    y_end, x_end = np.where(E)
    plt.plot(x_branch, y_branch, 'r+', markersize=10)  # Branch points in red
    plt.plot(x_end, y_end, 'c+', markersize=10)  # Endpoints in cyan

    plt.title('Skeleton with Branchpoints and Endpoints')
    # plt.gca().invert_yaxis()  # Invert y-axis to match MATLAB orientation
    plt.axis('off')  # Turn off axis
    plt.tight_layout()
    plt.show()

def process_creek_ordering(ordermax, Z, skeleton, outletdetection, nbbreaches): 
    # Initialize variables
    STRAHLER = []
    STRAIGHTDIST = []
    IDXSEG = []
    IDXBRANCH = []
    creekorder = np.zeros_like(skeleton, dtype=int)
    creekordersing = np.zeros_like(skeleton, dtype=int)

    skeleton_og = skeleton
    B, E, PTS = analyze_skeleton(skeleton)
    
    if outletdetection == 1:
        idxbreach, xbreach, ybreach, skeleton_breached = process_outlet_detection(Z, skeleton, PTS, nbbreaches)
    else:
        idxbreach, xbreach, ybreach, skeleton_breached = [], [], [], []

    i = 1
    max_iterations = 1000  # Failsafe to prevent infinite loop
    
    while np.any(E) and i <= max_iterations:
        
        B_loc = np.argwhere(B)
        E_loc = np.argwhere(E)
        
        E_processed = np.zeros_like(E, dtype=bool)
        
        for x, y in E_loc:
            DD = distance_transform_edt(skeleton)
            DD[~skeleton] = np.inf
            
            distanceToBranchPt = np.min(DD[tuple(B_loc.T)]) if B_loc.size > 0 else np.inf
            
            other_endpoints = E_loc[~np.all(E_loc == [x, y], axis=1)]
            if other_endpoints.size > 0:
                distanceToEndPt = np.min(DD[tuple(other_endpoints.T)])
            else:
                distanceToEndPt = np.inf
            
            if distanceToEndPt < distanceToBranchPt:
                STRAHLER.append(distanceToEndPt)
                if other_endpoints.size > 0:
                    ptloc = np.unravel_index(np.argmin(DD[tuple(other_endpoints.T)]), DD.shape)
                else:
                    ptloc = (x, y)
                STRAIGHTDIST.append(np.linalg.norm(np.array(ptloc) - np.array([x, y])))
                IDXBRANCH.append(np.ravel_multi_index(ptloc, DD.shape))
                
                x1, y1, x2, y2, x3, y3 = normal_coord(DD, distanceToEndPt, x, y, 3)
                IDXSEG.append([x1, y1, x2, y2, x3, y3])
                
                E_processed[x, y] = True
                
            elif distanceToBranchPt < np.inf:
                STRAHLER.append(distanceToBranchPt)
                ptloc = np.unravel_index(np.argmin(DD[tuple(B_loc.T)]), DD.shape)
                STRAIGHTDIST.append(np.linalg.norm(np.array(ptloc) - np.array([x, y])))
                IDXBRANCH.append(np.ravel_multi_index(ptloc, DD.shape))
                
                x1, y1, x2, y2, x3, y3 = normal_coord(DD, distanceToBranchPt, x, y, 3)
                IDXSEG.append([x1, y1, x2, y2, x3, y3])
                
                E_processed[x, y] = True
            
            else:
                # Handle the case where the endpoint is isolated
                STRAHLER.append(1)  # Assign a length of 1 to the isolated endpoint
                STRAIGHTDIST.append(1)
                IDXBRANCH.append(np.ravel_multi_index((x, y), DD.shape))
                IDXSEG.append([x, y, x, y, x, y])
                E_processed[x, y] = True
                skeleton[x, y] = False  # Remove the isolated endpoint from the skeleton
        
        creekordermask = morphology.dilation(E_processed, morphology.square(i))
        creekordersing[creekordermask] = i
        creekorder[creekordermask] = i
        
        skeleton = skeleton ^ E_processed
        B, E, PTS = analyze_skeleton(skeleton) # why is this here? - SamK 11/5/2024
        
        if not np.any(E_processed):
            print(f"No endpoints processed in iteration {i}. Breaking loop.")
            break
        
        i += 1
    
    if i > max_iterations:
        print(f"Maximum iterations ({max_iterations}) reached. Process may not have completed.")
    
    # Convert lists to numpy arrays
    STRAHLER = np.array(STRAHLER)
    STRAIGHTDIST = np.array(STRAIGHTDIST)
    IDXSEG = np.array(IDXSEG)
    IDXBRANCH = np.array(IDXBRANCH)

    B, E, PTS = analyze_skeleton(skeleton_og)
    
    return STRAHLER, STRAIGHTDIST, IDXSEG, IDXBRANCH, idxbreach, xbreach, ybreach, skeleton_breached, creekorder, creekordersing, PTS, list(range(1, i))


def process_creek_ordering_diagnostic(ordermax, Z, skeleton, outletdetection, nbbreaches): # for debugging
    # Initialize variables
    STRAHLER = []
    STRAIGHTDIST = []
    IDXSEG = []
    IDXBRANCH = []
    creekorder = np.zeros_like(skeleton, dtype=int)
    creekordersing = np.zeros_like(skeleton, dtype=int)

    B, E, PTS = analyze_skeleton(skeleton)
    
    if outletdetection == 1:
        idxbreach, xbreach, ybreach, skeleton_breached = process_outlet_detection(Z, skeleton, PTS, nbbreaches)
    else:
        idxbreach, xbreach, ybreach, skeleton_breached = [], [], [], []

    i = 1
    max_iterations = 1000  # Failsafe to prevent infinite loop
    
    while np.any(E) and i <= max_iterations:
        print(f"Iteration {i}: {np.sum(E)} endpoints remaining")
        
        B_loc = np.argwhere(B)
        E_loc = np.argwhere(E)
        
        E_processed = np.zeros_like(E, dtype=bool)
        
        for x, y in E_loc:
            DD = distance_transform_edt(skeleton)
            DD[~skeleton] = np.inf
            
            distanceToBranchPt = np.min(DD[tuple(B_loc.T)]) if B_loc.size > 0 else np.inf
            
            other_endpoints = E_loc[~np.all(E_loc == [x, y], axis=1)]
            if other_endpoints.size > 0:
                distanceToEndPt = np.min(DD[tuple(other_endpoints.T)])
            else:
                distanceToEndPt = np.inf
            
            print(f"  Endpoint at ({x}, {y}): distanceToBranchPt = {distanceToBranchPt}, distanceToEndPt = {distanceToEndPt}")
            
            if distanceToEndPt < distanceToBranchPt:
                STRAHLER.append(distanceToEndPt)
                if other_endpoints.size > 0:
                    ptloc = np.unravel_index(np.argmin(DD[tuple(other_endpoints.T)]), DD.shape)
                else:
                    ptloc = (x, y)
                STRAIGHTDIST.append(np.linalg.norm(np.array(ptloc) - np.array([x, y])))
                IDXBRANCH.append(np.ravel_multi_index(ptloc, DD.shape))
                
                x1, y1, x2, y2, x3, y3 = normal_coord(DD, distanceToEndPt, x, y, 3)
                IDXSEG.append([x1, y1, x2, y2, x3, y3])
                
                E_processed[x, y] = True
                print(f"    Processed endpoint to endpoint")
                
            elif distanceToBranchPt < np.inf:
                STRAHLER.append(distanceToBranchPt)
                ptloc = np.unravel_index(np.argmin(DD[tuple(B_loc.T)]), DD.shape)
                STRAIGHTDIST.append(np.linalg.norm(np.array(ptloc) - np.array([x, y])))
                IDXBRANCH.append(np.ravel_multi_index(ptloc, DD.shape))
                
                x1, y1, x2, y2, x3, y3 = normal_coord(DD, distanceToBranchPt, x, y, 3)
                IDXSEG.append([x1, y1, x2, y2, x3, y3])
                
                E_processed[x, y] = True
                print(f"    Processed endpoint to branch point")
            
            else:
                # Handle the case where the endpoint is isolated
                print(f"    Isolated endpoint detected. Processing and removing.")
                STRAHLER.append(1)  # Assign a length of 1 to the isolated endpoint
                STRAIGHTDIST.append(1)
                IDXBRANCH.append(np.ravel_multi_index((x, y), DD.shape))
                IDXSEG.append([x, y, x, y, x, y])
                E_processed[x, y] = True
                skeleton[x, y] = False  # Remove the isolated endpoint from the skeleton
        
        creekordermask = morphology.dilation(E_processed, morphology.square(i))
        creekordersing[creekordermask] = i
        creekorder[creekordermask] = i
        
        skeleton = skeleton ^ E_processed
        # B, E, PTS = analyze_skeleton(skeleton) # why is this here? - SamK 11/5/2024
        
        if not np.any(E_processed):
            print(f"No endpoints processed in iteration {i}. Breaking loop.")
            break
        
        i += 1
    
    if i > max_iterations:
        print(f"Maximum iterations ({max_iterations}) reached. Process may not have completed.")
    
    # Convert lists to numpy arrays
    STRAHLER = np.array(STRAHLER)
    STRAIGHTDIST = np.array(STRAIGHTDIST)
    IDXSEG = np.array(IDXSEG)
    IDXBRANCH = np.array(IDXBRANCH)
    
    return STRAHLER, STRAIGHTDIST, IDXSEG, IDXBRANCH, idxbreach, xbreach, ybreach, skeleton_breached, creekorder, creekordersing, PTS, list(range(1, i))

def process_outlet_detection(Z, skeleton, PTS, nbbreaches):
    # Create a mask for the landmass
    landmask = ~np.isnan(Z)
    
    # Create border mask (pixels exactly on the boundary)
    border_mask = np.zeros_like(landmask, dtype=bool)
    border_mask[1:-1, 1:-1] = (landmask[1:-1, 1:-1] & (
        ~landmask[:-2, 1:-1] |  # top
        ~landmask[2:, 1:-1] |   # bottom
        ~landmask[1:-1, :-2] |  # left
        ~landmask[1:-1, 2:]     # right
    ))

    # Create distance transform from the border
    distance_from_border = ndimage.distance_transform_edt(landmask)
    distance_from_border[~landmask] = 0

    # Create new border mask 5 pixels inward
    border_mask_inward = (distance_from_border >= 5) & (distance_from_border <= 6)
    border_mask_inward = border_mask_inward & landmask
    
    # # Create a distance map from the edge of the landmass
    # distance_from_edge = ndimage.distance_transform_edt(landmask)

    # Create a distance map from the contracted border
    distance_from_edge = ndimage.distance_transform_edt(border_mask_inward)
    
    # Consider points close to the edge (within 5 pixels)
    edge_region = distance_from_edge <= 5
    
    # Find skeleton points near the edge
    potential_outlets = skeleton & edge_region
    
    # Get coordinates of potential outlets
    y_outlets, x_outlets = np.where(potential_outlets)
    
    if len(y_outlets) == 0:
        print("No potential outlets found. Check the skeleton and landmass boundary.")
        return [], [], []
    
    # Get elevations of potential outlets
    outlet_elevations = Z[y_outlets, x_outlets]
    
    # Sort outlets by elevation
    sorted_indices = np.argsort(outlet_elevations)
    
    # Select the lowest nbbreaches points
    selected_indices = sorted_indices[:nbbreaches]
    
    ybreach = y_outlets[selected_indices]
    xbreach = x_outlets[selected_indices]
    
    # For each outlet point, find the nearest border point and connect to it
    new_xbreach = []
    new_ybreach = []
    
    for y, x in zip(ybreach, xbreach):
        # Find nearest border point
        border_y, border_x = np.where(border_mask)
        
        # Calculate distances to all border points
        distances = np.sqrt((border_y - y)**2 + (border_x - x)**2)
        nearest_idx = np.argmin(distances)
        
        new_y = border_y[nearest_idx]
        new_x = border_x[nearest_idx]
        
        # Draw a line to connect the original outlet to the border point
        rr, cc = draw.line(y, x, new_y, new_x)
        skeleton[rr, cc] = True
        
        # Add the new border point to our list
        new_ybreach.append(new_y)
        new_xbreach.append(new_x)
    
    # Convert lists back to arrays
    new_ybreach = np.array(new_ybreach)
    new_xbreach = np.array(new_xbreach)
    
    # Create the breach indices
    idxbreach = np.ravel_multi_index((new_ybreach, new_xbreach), Z.shape)
    
    # Visualization
    plt.figure(figsize=(12, 12))
    plt.imshow(landmask, cmap='gray')
    plt.imshow(skeleton, cmap='Blues', alpha=0.5)
    plt.plot(new_xbreach, new_ybreach, 'r+', markersize=20, markeredgewidth=2, label='New Outlets')
    plt.plot(xbreach, ybreach, 'g+', markersize=15, markeredgewidth=2, label='Original Outlets')
    plt.legend()
    plt.title('Outlet Detection with Border Correction')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 12))
    plt.imshow(landmask, cmap='gray')
    plt.imshow(skeleton, cmap='Blues', alpha=0.5)
    plt.title('Outlet Detection Final Skeleton')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    skeleton_breached = skeleton
    
    return idxbreach, new_xbreach, new_ybreach, skeleton_breached

def normal_coord(D, distanceToPt, x, y, limit):
    Dmasktemp = D < distanceToPt
    Dtemp = distance_transform_edt(Dmasktemp)
    maxDtemp = np.max(Dtemp)
    
    if maxDtemp <= limit:
        xm, ym = x, y
        endpt = np.unravel_index(np.argmax(Dtemp), Dtemp.shape)
        x1, y1 = endpt
    else:
        midpt = np.unravel_index(np.argmin(np.abs(Dtemp - np.mean(Dtemp[Dtemp > 0]))), Dtemp.shape)
        xm, ym = midpt
        
        dist = int(0.2 * maxDtemp)
        ALLIDX = np.argwhere(distance_transform_edt(Dmasktemp) == dist)
        x1, y1 = ALLIDX[0]
        x2, y2 = ALLIDX[-1]
    
    v = np.array([x1 - xm, y1 - ym])
    v_perp = np.array([-v[1], v[0]])
    
    x2, y2 = xm + v_perp[0], ym + v_perp[1]
    x3, y3 = xm - v_perp[0], ym - v_perp[1]
    
    return xm, ym, x2, y2, x3, y3

# # Main execution
# creekmask = np.load('creekmask.npy')  # Load your creekmask data
# Z = np.load('Z.npy')  # Load your Z data
# ordermax = 5  # Set your desired maximum order
# outletdetection = 1  # Set to 1 if you want to detect outlets, 0 otherwise
# nbbreaches = 3  # Set the number of breaches to detect

# skeleton = morphology.skeletonize(creekmask)
# figure_creek_skeleton(creekmask, skeleton)

# STRAHLER, STRAIGHTDIST, IDXSEG, IDXBRANCH, idxbreach, xbreach, ybreach, creekorder, creekordersing, PTS, ID = process_creek_ordering(ordermax, Z, skeleton, outletdetection, nbbreaches)

# print(creekorder)  # Display the creek order