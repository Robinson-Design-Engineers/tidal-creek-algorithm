import numpy as np # type: ignore
from skimage import morphology, measure, feature, draw # type: ignore
import matplotlib.pyplot as plt # type: ignore
from scipy import ndimage # type: ignore
from scipy.ndimage import distance_transform_edt # type: ignore
from skimage.morphology import dilation, disk # type: ignore
import networkx as nx # type: ignore
from scipy.spatial.transform import Rotation # type: ignore

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

def figure_creek_skeleton(skeleton):
    # Thicken the skeleton for display
    skeletonpic = morphology.dilation(skeleton, morphology.square(3))
    skeletonpic2 = np.logical_not(skeletonpic)

    B, E, _ = analyze_skeleton(skeleton)

    # Main visualization
    plt.figure(figsize=(8, 8))
    skeletonpic2 = np.transpose(skeletonpic2)
    plt.imshow(skeletonpic2, cmap='gray', origin='upper', label='skeleton')

    # Plot branch points and endpoints
    y_branch, x_branch = np.where(B)
    y_end, x_end = np.where(E)
    # plt.plot(x_branch, y_branch, 'r+', markersize=10, label='branch points')  # Branch points in red
    # plt.plot(x_end, y_end, 'c+', markersize=10, label='endpoints')  # Endpoints in cyan
    plt.plot(y_branch, x_branch, 'r+', markersize=10, label='branch points')  # Branch points in red
    plt.plot(y_end, x_end, 'c+', markersize=10, label='endpoints')  # Endpoints in cyan

    plt.title('Skeleton with Branchpoints and Endpoints')
    # plt.gca().invert_yaxis()  # Invert y-axis to match MATLAB orientation
    plt.axis('off')  # Turn off axis
    plt.legend()
    plt.tight_layout()
    plt.show()

def figure_creek_skeleton_diagnostic(skeleton, x_poi, y_poi):
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
    skeletonpic2 = np.transpose(skeletonpic2)
    plt.imshow(skeletonpic2, cmap='gray', origin='upper', label='skeleton')

    # Plot branch points and endpoints
    y_branch, x_branch = np.where(B)
    y_end, x_end = np.where(E)
    # plt.plot(x_branch, y_branch, 'r+', alpha = 0.7, markersize=10, label='branchpoints')  # Branch points in red
    # plt.plot(x_end, y_end, 'c+', alpha = 0.7, markersize=10, label='endpoints')  # Endpoints in cyan
    # plt.plot(x_poi, y_poi, 'o', markersize=10, markeredgecolor='r', markerfacecolor='none', label='point of interest')
    plt.plot(y_branch, x_branch, 'r+', alpha = 0.7, markersize=10, label='branchpoints')  # Branch points in red
    plt.plot(y_end, x_end, 'c+', alpha = 0.7, markersize=10, label='endpoints')  # Endpoints in cyan
    plt.plot(y_poi, x_poi, 'o', markersize=10, markeredgecolor='r', markerfacecolor='none', label='point of interest')

    plt.title('Skeleton with Branchpoints and Endpoints')
    # plt.gca().invert_yaxis()  # Invert y-axis to match MATLAB orientation
    # plt.axis('off')  # Turn off axis
    plt.legend()
    plt.tight_layout()
    plt.show()

def process_creek_ordering(ordermax, Z, skeleton, outletdetection, nbbreaches):
    
    # Automatically detect one or several outlets (deepest end points in the skeleton)
    if outletdetection == 1:
        idxbreach, xbreach, ybreach, skeleton_breached = process_outlet_detection(Z, skeleton, nbbreaches)
    else:
        idxbreach, xbreach, ybreach, skeleton_breached = [], [], [], []

    # resave skeleton as breached skeleton to ensure outlet point is connected to skeleton
    skeleton = skeleton_breached # added to debug -SamK

    # Initialize variables
    B, E, PTS = analyze_skeleton(skeleton)
    all_pts = PTS
    STRAHLER = np.zeros_like(E, dtype=float) # contains sinous length
    STRAIGHTDIST = np.zeros_like(E, dtype=float) # contains straight length
    IDXSEG = np.zeros((skeleton.shape[0], 6)) # contains coordinates of the normal vector (end and mid points)
    IDXBRANCH = np.zeros_like(skeleton, dtype=int) # contains indices of all branch points
    # IDXBRANCH = np.zeros_like(skeleton, dtype=[('row', int), ('col', int)])
    creekorder = np.zeros_like(skeleton, dtype=int)
    creekordersing = np.zeros_like(skeleton, dtype=int)
    skeleton_chopped = skeleton # we keep skeleton intact and remove segments from skeleton_chopped
    ID = []
    i = 1
    limit = 3

    # Get the Strahler order and segment lengths in a loop
    i_check = 1
    max_iterations = 1000  # Failsafe to prevent infinite loop
    skelprev = np.zeros_like(skeleton_chopped, dtype=bool)
    
    # while loop 1 out of 4
    while np.any(E) and not np.array_equal(skelprev, skeleton_chopped) and i_check<= max_iterations: # and i < ordermax - this last condition was commented out in AnnaM's MATLAB code -SamK

        # # Find entry channels, write them as branch points
        # # E.ravel()[idxbreach] = False # ravel flattens a 2D array
        # # B.ravel()[idxbreach] = True
        # Add breach point as branch point
        B[ybreach, xbreach] = True
        E[ybreach, xbreach] = False
       
        # start segment selection from end points
        # Find the locations of all branch and end points in the order i configuration
        B_loc = np.argwhere(B) # np.argwhere returns a 2D array of coordinates
        E_loc = np.argwhere(E)
        Dmask = np.zeros_like(skeleton_chopped, dtype=bool)
        col = 0 # starting column index for normal vector coordinate in IDXSEG
        coords = np.argwhere(E)
        y, x = coords[:, 0], coords[:, 1]
        
        for k in range(len(x)):
            # remove the selected kth end point from E_loc
            E_loctemp = np.delete(E_loc, k, axis=0) # assume E_loc is 2D array
            
            # Get distances from each pixel to kth end point
            DD = calculate_geodesic_distance(skeleton_chopped, y[k], x[k])
            
            # Retrieve distance from kth point to nearest branch point, using DD values
            branch_distances = DD[B_loc[:, 0], B_loc[:, 1]]
            finite_branch_distances = branch_distances[np.isfinite(branch_distances)]
            # have a check to make sure there are actual values
            try:
                distanceToBranchPt = np.min(finite_branch_distances).astype(float)
            except ValueError:
                distanceToBranchPt = np.inf  # or another appropriate value

            # Retrieve distance from kth point to nearest end point other than itself, using DD values
            end_distances = DD[E_loctemp[:, 0], E_loctemp[:, 1]]
            finite_end_distances = end_distances[np.isfinite(end_distances)]
            # have a check to make sure there are actual values
            try:
                distanceToEndPt = np.min(finite_end_distances).astype(float)
            except ValueError:
                distanceToEndPt = np.inf  # or another appropriate value
            
            # Three possibilities here: the end point we have detected can be:
            # 1) an isolated point (no creek segment to detect)
            # 2) an isolated creek segment delimited by two end points
            # 3) a creek segment delimited by a branch point and an end point

            # Remove isolated points (no creek segment to detect)
            if np.isinf(distanceToEndPt) and np.isinf(distanceToBranchPt):
                # Remove the point from skeleton_chopped
                skeleton_chopped[y[k], x[k]] = False
                # Remove point from endpoint array E
                E[y[k], x[k]] = False
                
                # Continue to next point in the original coords list
                continue
            
            # Find isolated segment (end point closer than branch point) - I guess like a vernal pool? -SamK
            elif distanceToEndPt < distanceToBranchPt:
                # Store segment real length into strahler table
                STRAHLER[i,k] = distanceToEndPt

                # Find coordinates of the closest endpoint
                ptlocy, ptlocx = np.where(DD==distanceToEndPt)
                ptloc = np.where(DD==distanceToEndPt)
                ptloc = ptloc[0]
                ptlocx = ptlocx[0]
                ptlocy = ptlocy[0]
                # # alternate:
                # E_loc_distances = DD[E_loctemp[:, 0], E_loctemp[:, 1]]
                # finite_distances = E_loc_distances[np.isfinite(end_distances)]
                # # Add check for empty array
                # if finite_distances.size == 0:
                #     # No valid distances found, skip this point
                #     continue
                # ptlocy, ptlocx = np.where(DD==np.min(finite_distances))

                # Find segment straight length
                # "Create empty image with only two points" - this is the MATLAB comment, but it seems they only create one point - SamK
                skeletoneucl = np.zeros_like(skeleton_chopped)
                skeletoneucl[E_loc[k,0], E_loc[k,1]] = 1
                euclD = ndimage.distance_transform_edt(skeletoneucl == 0) # for each pixel, euclidean dist to nearest non-zero value 
                disttopt = euclD[ptlocy, ptlocx]
                STRAIGHTDIST[i,k] = disttopt
                # Save index of branch point for junction angle assignment
                if np.isscalar(ptlocx) and np.isscalar(ptlocy):
                    # If they're already scalars
                    ptloc_linear = np.ravel_multi_index((ptlocy, ptlocx), DD.shape)
                else:
                    # If they're arrays
                    ptloc_linear = np.ravel_multi_index((ptlocy[0], ptlocx[0]), DD.shape)
                IDXBRANCH[i,k] = ptloc_linear

                # Create normal vectors to each segment and store their coordinates
                # should it be y[k], x[k] or x[k], y[k]? -SamK
                x1, y1, x2, y2, x3, y3, _, _ = normal_coord(DD, distanceToEndPt, y[k], x[k], limit)
                
                if col < IDXSEG.shape[1]:
                    IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
                else:
                    new_values = [x1, y1, x2, y2, x3, y3]
                    IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                    IDXSEG[i, -6:] = new_values

                col += 6

                # Prepare to remove the segments order i
                Dmask[DD < distanceToEndPt] = True

            # Find isolated segment (branch point doesn't exist)
            # seems exact same as if distanceToEndPt < distanceToBranchPt, except 
            # uses normal_coord function on x[k], y[k] rather than y[k], x[k] as above -SamK
            elif np.isinf(distanceToBranchPt) and np.isfinite(distanceToEndPt):
                # Store segment real length into strahler table
                STRAHLER[i,k] = distanceToEndPt

                # Find coordinates of the closest endpoint
                ptlocy, ptlocx = np.where(DD==distanceToEndPt)
                ptloc = np.where(DD==distanceToEndPt)
                ptloc = ptloc[0]
                ptlocx = ptlocx[0]
                ptlocy = ptlocy[0]
                # # alternate:
                # E_loc_distances = DD[E_loctemp[:, 0], E_loctemp[:, 1]]
                # finite_distances = E_loc_distances[np.isfinite(end_distances)]
                # # Add check for empty array
                # if finite_distances.size == 0:
                #     # No valid distances found, skip this point
                #     continue
                # ptlocy, ptlocx = np.where(DD==np.min(finite_distances))

                # Find segment straight length
                # "Create empty image with only two points" - this is the MATLAB comment, but it seems they only create one point - SamK
                skeletoneucl = np.zeros_like(skeleton_chopped)
                skeletoneucl[E_loc[k,0], E_loc[k,1]] = 1
                euclD = ndimage.distance_transform_edt(skeletoneucl == 0) # for each pixel, euclidean dist to nearest non-zero value 
                disttopt = euclD[ptlocy, ptlocx]
                STRAIGHTDIST[i,k] = disttopt
                # Save index of branch point for junction angle assignment
                if np.isscalar(ptlocx) and np.isscalar(ptlocy):
                    # If they're already scalars
                    ptloc_linear = np.ravel_multi_index((ptlocy, ptlocx), DD.shape)
                else:
                    # If they're arrays
                    ptloc_linear = np.ravel_multi_index((ptlocy[0], ptlocx[0]), DD.shape)
                IDXBRANCH[i,k] = ptloc_linear

                # Create normal vectors to each segment and store their coordinates
                x1, y1, x2, y2, x3, y3, _, _ = normal_coord(DD, distanceToEndPt, x[k], y[k], limit)
                
                if col < IDXSEG.shape[1]:
                    IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
                else:
                    new_values = [x1, y1, x2, y2, x3, y3]
                    IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                    IDXSEG[i, -6:] = new_values

                col += 6

                # Prepare to remove the segments order i
                Dmask[DD < distanceToEndPt] = True

            # Find connected segment (detect closest branch point)
            else:
                # Store segment real length into strahler table
                STRAHLER[i,k] = distanceToBranchPt

                ptlocy, ptlocx = np.where(DD==distanceToBranchPt)
                ptloc = np.where(DD==distanceToBranchPt)
                ptloc = ptloc[0]
                ptlocx = ptlocx[0]
                ptlocy = ptlocy[0]

                # Find segment straight length
                # "Create empty image with only two points" - this is the MATLAB comment, but it seems they only create one point - SamK
                skeletoneucl = np.zeros_like(skeleton_chopped)
                skeletoneucl[E_loc[k,0], E_loc[k,1]] = 1
                euclD = ndimage.distance_transform_edt(skeletoneucl == 0) # for each pixel, euclidean dist to nearest non-zero value 
                disttopt = euclD[ptlocy, ptlocx]
                STRAIGHTDIST[i,k] = disttopt
                # Save index of branch point for junction angle assignment
                if np.isscalar(ptlocx) and np.isscalar(ptlocy):
                    # If they're already scalars
                    ptloc_linear = np.ravel_multi_index((ptlocy, ptlocx), DD.shape)
                else:
                    # If they're arrays
                    ptloc_linear = np.ravel_multi_index((ptlocy[0], ptlocx[0]), DD.shape)
                IDXBRANCH[i,k] = ptloc_linear

                # Create normal vectors to each segment and store their coordinates
                # should it be y[k], x[k] or x[k], y[k]? -SamK
                x1, y1, x2, y2, x3, y3, _, _ = normal_coord(DD, distanceToBranchPt, x[k], y[k], limit)
                
                if col < IDXSEG.shape[1]:
                    IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
                else:
                    new_values = [x1, y1, x2, y2, x3, y3]
                    IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                    IDXSEG[i, -6:] = new_values

                col += 6

                # Prepare to remove the segments order i
                Dmask[DD < distanceToBranchPt] = True


        # Prepare the creek order plot
        creekordermask = bwmorph_diag(Dmask)
        # Assign creek orders
        creekordersing[creekordermask != 0] = creekordermask[creekordermask != 0] * i
        # Thicken the mask i times
        # creekordermask = bwmorph_thicken(creekordermask, i)
        creekordermask = dilation(creekordermask, disk(1))
        # Assign creek orders to thickened mask
        creekorder[creekordermask != 0] = creekordermask[creekordermask != 0] * i
        # Clear the mask
        del creekordermask

        # Remove order i segments
        # skelD = skeleton_chopped - Dmask # subtract matrices
        skelD = np.logical_xor(skeleton_chopped, Dmask) # or use skelD = skeleton_chopped ^ Dmask
        skeleton_chopped = skelD.astype(bool) # convert to boolean/logical, could also do skelchopped = skelD > 0 -SamK

        # Redo skeleton
        skeleton_chopped = morphology.skeletonize(skeleton_chopped)
        ID = np.append(ID, i) # update order tracking

        # Find the end and branch points of the order i+1 network configuration
        i += 1 # Store order number and move on to the next creek order until skeleton contains only 0
        col = 0
        B, E, PTS = analyze_skeleton(skeleton_chopped)
        # Update breach points
        E.ravel()[idxbreach] = False
        B.ravel()[idxbreach] = True

        i_check += 1
        if i_check >= max_iterations:
            print("Max iterations reached, breaking out of the loop.")
            break

    if i == ordermax:
        ID = np.append(ID, i)
    else:
        i = ordermax
        ID = np.arange(1, ordermax + 1)  # Create array from 1 to ordermax

    # Segment assignation has occured until no more end points are detected. 
    # We do a second one that also detects the outlets.
    # Segment assignation for outlets
    skelprev = np.zeros_like(skeleton, dtype=bool)
    B, E, PTS = analyze_skeleton(skeleton_chopped)
    ESIZE = np.any(E) # doesn't seem to be used later -SamK
    test = 1

    i_check = 1

    # while loop 2 out of 4 - outlets
    # ensure there are endpoints and that skelprev and skeleton_chopped are not identical
    while np.any(E) and not np.array_equal(skelprev, skeleton_chopped) and i_check<= max_iterations:
        test = test + 1

        # Find the locations of all branch and end points in the order i configuration
        B_loc = np.argwhere(B) # np.argwhere returns a 2D array of coordinates
        E_loc = np.argwhere(E)
        Dmask = np.zeros_like(skeleton_chopped, dtype=bool)
        # col = 0
        coords = np.argwhere(E)
        y, x = coords[:, 0], coords[:, 1]
        
        # Loop through each end point in order i configuration
        for k2 in range(len(x)):
            k = k + 1

            # Remove selected end point from E_loc
            E_loctemp = np.delete(E_loc, k2, axis=0) # assume E_loc is 2D array

            # get quasi-euclidean distance from selected end point to the rest of the creek network and find closest branch or end point
            # Get distances to kth end point
            DD = calculate_geodesic_distance(skeleton_chopped, y[k2], x[k2])
            
            # Retrieve distance from kth point to nearest branch point, using DD values
            branch_distances = DD[B_loc[:, 0], B_loc[:, 1]]
            finite_branch_distances = branch_distances[np.isfinite(branch_distances)]
            # have a check to make sure there are actual values
            try:
                distanceToBranchPt = np.min(finite_branch_distances).astype(float)
            except ValueError:
                distanceToBranchPt = np.inf  # or another appropriate value

            # Retrieve distance from kth point to nearest end point other than itself, using DD values
            end_distances = DD[E_loctemp[:, 0], E_loctemp[:, 1]]
            finite_end_distances = end_distances[np.isfinite(end_distances)]
            # have a check to make sure there are actual values
            try:
                distanceToEndPt = np.min(finite_end_distances).astype(float)
            except ValueError:
                distanceToEndPt = np.inf  # or another appropriate value
            
            # Three possibilities here: the end point we have detected can be:
            # 1) an isolated point (no creek segment to detect)
            # 2) an isolated creek segment delimited by two end points
            # 3) a creek segment delimited by a branch point and an end point

            # Remove isolated points (no creek segment to detect)
            if np.isinf(distanceToEndPt) and np.isinf(distanceToBranchPt):
                # Remove the point from skeleton_chopped
                skeleton_chopped[y[k2], x[k2]] = False
                # Remove point from endpoint array E
                E[y[k2], x[k2]] = False
                
                # Continue to next point in the original coords list
                continue
            
            # Find isolated segment (end point closer than branch point) - I guess like a vernal pool? -SamK
            elif distanceToEndPt < distanceToBranchPt:
                # Store segment real length into strahler table
                STRAHLER[i,k] = distanceToEndPt

                # Find coordinates of the closest endpoint
                ptlocy, ptlocx = np.where(DD==distanceToEndPt)
                ptloc = np.where(DD==distanceToEndPt)
                ptloc = ptloc[0]
                ptlocx = ptlocx[0]
                ptlocy = ptlocy[0]
                # # alternate:
                # E_loc_distances = DD[E_loctemp[:, 0], E_loctemp[:, 1]]
                # finite_distances = E_loc_distances[np.isfinite(end_distances)]
                # # Add check for empty array
                # if finite_distances.size == 0:
                #     # No valid distances found, skip this point
                #     continue
                # ptlocy, ptlocx = np.where(DD==np.min(finite_distances))

                # Find segment straight length
                # "Create empty image with only two points" - this is the MATLAB comment, but it seems they only create one point - SamK
                skeletoneucl = np.zeros_like(skeleton_chopped)
                skeletoneucl[E_loc[k2,0], E_loc[k2,1]] = 1
                euclD = ndimage.distance_transform_edt(skeletoneucl == 0) # for each pixel, euclidean dist to nearest non-zero value 
                disttopt = euclD[ptlocy, ptlocx]
                STRAIGHTDIST[i,k] = disttopt
                # Save index of branch point for junction angle assignment
                if np.isscalar(ptlocx) and np.isscalar(ptlocy):
                    # If they're already scalars
                    ptloc_linear = np.ravel_multi_index((ptlocy, ptlocx), DD.shape)
                else:
                    # If they're arrays
                    ptloc_linear = np.ravel_multi_index((ptlocy[0], ptlocx[0]), DD.shape)
                IDXBRANCH[i,k] = ptloc_linear

                # Create normal vectors to each segment and store their coordinates
                # should it be y[k2], x[k2] or x[k2], y[k2]? -SamK
                x1, y1, x2, y2, x3, y3, _, _ = normal_coord(DD, distanceToEndPt, y[k2], x[k2], limit)
                
                if col < IDXSEG.shape[1]:
                    IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
                else:
                    new_values = [x1, y1, x2, y2, x3, y3]
                    IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                    IDXSEG[i, -6:] = new_values

                col += 6

                # Prepare to remove the segments order i
                Dmask[DD < distanceToEndPt] = True

            # Find isolated segment (branch point doesn't exist)
            # seems exact same as if distanceToEndPt < distanceToBranchPt, except 
            # uses normal_coord function on x[k], y[k] rather than y[k], x[k] as above -SamK
            elif np.isinf(distanceToBranchPt) and np.isfinite(distanceToEndPt):
                # Store segment real length into strahler table
                STRAHLER[i,k] = distanceToEndPt

                # Find coordinates of the closest endpoint
                ptlocy, ptlocx = np.where(DD==distanceToEndPt)
                ptloc = np.where(DD==distanceToEndPt)
                ptloc = ptloc[0]
                ptlocx = ptlocx[0]
                ptlocy = ptlocy[0]
                # # alternate:
                # E_loc_distances = DD[E_loctemp[:, 0], E_loctemp[:, 1]]
                # finite_distances = E_loc_distances[np.isfinite(end_distances)]
                # # Add check for empty array
                # if finite_distances.size == 0:
                #     # No valid distances found, skip this point
                #     continue
                # ptlocy, ptlocx = np.where(DD==np.min(finite_distances))

                # Find segment straight length
                # "Create empty image with only two points" - this is the MATLAB comment, but it seems they only create one point - SamK
                skeletoneucl = np.zeros_like(skeleton_chopped)
                skeletoneucl[E_loc[k2,0], E_loc[k2,1]] = 1
                euclD = ndimage.distance_transform_edt(skeletoneucl == 0) # for each pixel, euclidean dist to nearest non-zero value 
                disttopt = euclD[ptlocy, ptlocx]
                STRAIGHTDIST[i,k] = disttopt
                # Save index of branch point for junction angle assignment
                if np.isscalar(ptlocx) and np.isscalar(ptlocy):
                    # If they're already scalars
                    ptloc_linear = np.ravel_multi_index((ptlocy, ptlocx), DD.shape)
                else:
                    # If they're arrays
                    ptloc_linear = np.ravel_multi_index((ptlocy[0], ptlocx[0]), DD.shape)
                IDXBRANCH[i,k] = ptloc_linear

                # Create normal vectors to each segment and store their coordinates
                x1, y1, x2, y2, x3, y3, _, _ = normal_coord(DD, distanceToEndPt, x[k2], y[k2], limit)
                
                if col < IDXSEG.shape[1]:
                    IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
                else:
                    new_values = [x1, y1, x2, y2, x3, y3]
                    IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                    IDXSEG[i, -6:] = new_values

                col += 6

                # Prepare to remove the segments order i
                Dmask[DD < distanceToEndPt] = True

            # Find connected segment (detect closest branch point)
            else:
                # Store segment real length into strahler table
                STRAHLER[i,k] = distanceToBranchPt

                ptlocy, ptlocx = np.where(DD==distanceToBranchPt)
                ptloc = np.where(DD==distanceToBranchPt)
                ptloc = ptloc[0]
                ptlocx = ptlocx[0]
                ptlocy = ptlocy[0]

                # Find segment straight length
                # "Create empty image with only two points" - this is the MATLAB comment, but it seems they only create one point - SamK
                skeletoneucl = np.zeros_like(skeleton_chopped)
                skeletoneucl[E_loc[k2,0], E_loc[k2,1]] = 1
                euclD = ndimage.distance_transform_edt(skeletoneucl == 0) # for each pixel, euclidean dist to nearest non-zero value 
                disttopt = euclD[ptlocy, ptlocx]
                STRAIGHTDIST[i,k] = disttopt
                # Save index of branch point for junction angle assignment
                if np.isscalar(ptlocx) and np.isscalar(ptlocy):
                    # If they're already scalars
                    ptloc_linear = np.ravel_multi_index((ptlocy, ptlocx), DD.shape)
                else:
                    # If they're arrays
                    ptloc_linear = np.ravel_multi_index((ptlocy[0], ptlocx[0]), DD.shape)
                IDXBRANCH[i,k] = ptloc_linear

                # Create normal vectors to each segment and store their coordinates
                # should it be y[k], x[k] or x[k], y[k]? -SamK
                x1, y1, x2, y2, x3, y3, _, _ = normal_coord(DD, distanceToBranchPt, x[k2], y[k2], limit)
                
                if col < IDXSEG.shape[1]:
                    IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
                else:
                    new_values = [x1, y1, x2, y2, x3, y3]
                    IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                    IDXSEG[i, -6:] = new_values

                col += 6

                # Prepare to remove the segments order i
                Dmask[DD < distanceToBranchPt] = True

        # Prepare the creek order plot
        creekordermask = bwmorph_diag(Dmask)
        # Assign creek orders
        creekordersing[creekordermask != 0] = creekordermask[creekordermask != 0] * i
        # Thicken the mask i times
        # creekordermask = bwmorph_thicken(creekordermask, i)
        creekordermask = dilation(creekordermask, disk(1))
        # Assign creek orders to thickened mask
        creekorder[creekordermask != 0] = creekordermask[creekordermask != 0] * i
        # Clear the mask
        del creekordermask

        # Remove order i segments
        # skelD = skeleton_chopped - Dmask # subtract matrices
        skelD = np.logical_xor(skeleton_chopped, Dmask) # or use skelD = skeleton_chopped ^ Dmask
        skeleton_chopped = skelD.astype(bool) # convert to boolean/logical, could also do skelchopped = skelD > 0 -SamK

        # Redo skeleton
        skeleton_chopped = morphology.skeletonize(skeleton_chopped)
        # ID = np.append(ID, i) # update order tracking

        # Find the end and branch points of the order i+1 network configuration
        # i += 1 # Store order number and move on to the next creek order until skeleton contains only 0
        # col = 0
        B, E, PTS = analyze_skeleton(skeleton_chopped)

        i_check += 1
        if i_check >= max_iterations:
            print("Max iterations reached, breaking out of the loop.")
            break
    
    # Outlet Segment assignation has occured until no more end points are detected. 
    # We do a THIRD one that detects all branch points and gets interconnected sections.

    skelprev = np.zeros_like(skeleton, dtype=bool)
    B, E, PTS = analyze_skeleton(skeleton_chopped)
    B[E != 0] = True # could also do: B[np.where(E)] = True -SamK
    # # Update breach points
    # E.ravel()[idxbreach] = False
    # B.ravel()[idxbreach] = True

    # Figure - not translated as of now, commented out in AnnaM's MATLAB code -SamK

    i_check = 1

    # while loop 3 out of 4 - branch points / interconnected segments
    while np.any(B) and i_check<= max_iterations:
        # Select 1 node
        k2 = 1

        # Find the locations of all branch and end points in the order i configuration
        B_loc = np.argwhere(B) # np.argwhere returns a 2D array of coordinates
        # E_loc = np.argwhere(E)
        Dmask = np.zeros_like(skeleton_chopped, dtype=bool)
        # col = 0
        coords = np.argwhere(B) # different than being based on E in first two while loops -SamK
        y, x = coords[:, 0], coords[:, 1]

        # "Plot the order i configuration" and "Check for interconnections" commented out in AnnaM's MATLAB code -SamK
        
        # for k2 in range(len(x)): # commented out in AnnaM's MATLAB code -SamK
        k = k+1

        # remove the selected kth end point from E_loc
        E_loctemp = np.delete(E_loc, k2, axis=0) # assume E_loc is 2D array
        B_loctemp = np.delete(B_loc, k2, axis=0) # assume B_loc is 2D array

        y2 = x
        x = y
        y = y2

        # get quasi-euclidean distance from selected end point to the 
        # rest of the creek network and find closest branch or end point
        # Get distances
        DD = calculate_geodesic_distance(skeleton_chopped, y[k2], x[k2]) 
        # commented out figure -SamK
            
        # Retrieve distance from kth point to nearest branch point, using DD values
        branch_distances = DD[B_loctemp[:, 0], B_loctemp[:, 1]]
        finite_branch_distances = branch_distances[np.isfinite(branch_distances)]
        # have a check to make sure there are actual values
        try:
            distanceToBranchPt = np.min(finite_branch_distances).astype(float)
        except ValueError:
            distanceToBranchPt = np.inf  # or another appropriate value

        # Retrieve distance from kth point to nearest end point other than itself, using DD values
        end_distances = DD[E_loctemp[:, 0], E_loctemp[:, 1]]
        finite_end_distances = end_distances[np.isfinite(end_distances)]
        # have a check to make sure there are actual values
        try:
            distanceToEndPt = np.min(finite_end_distances).astype(float)
        except ValueError:
            distanceToEndPt = np.inf  # or another appropriate value
            
        # Three possibilities here: the end point we have detected can be:
        # 1) an isolated point (no creek segment to detect)
        # 2) an isolated creek segment delimited by two end points
        # 3) a creek segment delimited by a branch point and an end point

        # Remove isolated points (no creek segment to detect)
        if np.isinf(distanceToEndPt) and np.isinf(distanceToBranchPt):
            # Remove the point from skeleton_chopped
            skeleton_chopped[y[k2], x[k2]] = False
            # Remove point from endpoint array E
            E[y[k2], x[k2]] = False
            
            # Continue to next point in the original coords list
            continue
        
        # Find isolated segment (end point closer than branch point) - I guess like a vernal pool? -SamK
        elif distanceToEndPt < distanceToBranchPt:
            # Store segment real length into strahler table
            STRAHLER[i,k] = distanceToEndPt

            # Find coordinates of the closest endpoint
            ptlocy, ptlocx = np.where(DD==distanceToEndPt)
            ptloc = np.where(DD==distanceToEndPt)
            ptloc = ptloc[0]
            ptlocx = ptlocx[0]
            ptlocy = ptlocy[0]
            # # alternate:
            # E_loc_distances = DD[E_loctemp[:, 0], E_loctemp[:, 1]]
            # finite_distances = E_loc_distances[np.isfinite(end_distances)]
            # # Add check for empty array
            # if finite_distances.size == 0:
            #     # No valid distances found, skip this point
            #     continue
            # ptlocy, ptlocx = np.where(DD==np.min(finite_distances))

            # Find segment straight length
            # "Create empty image with only two points" - this is the MATLAB comment, but it seems they only create one point - SamK
            skeletoneucl = np.zeros_like(skeleton_chopped)
            skeletoneucl[B_loc[k2,0], B_loc[k2,1]] = 1
            euclD = ndimage.distance_transform_edt(skeletoneucl == 0) # for each pixel, euclidean dist to nearest non-zero value 
            disttopt = euclD[ptlocy, ptlocx]
            STRAIGHTDIST[i,k] = disttopt
            # Save index of branch point for junction angle assignment
            if np.isscalar(ptlocx) and np.isscalar(ptlocy):
                # If they're already scalars
                ptloc_linear = np.ravel_multi_index((ptlocy, ptlocx), DD.shape)
            else:
                # If they're arrays
                ptloc_linear = np.ravel_multi_index((ptlocy[0], ptlocx[0]), DD.shape)
            IDXBRANCH[i,k] = ptloc_linear

            # Create normal vectors to each segment and store their coordinates
            x1, y1, x2, y2, x3, y3, BWrem = normal_coord_test(DD, distanceToEndPt, x[k2], y[k2], limit, skeleton_chopped, B_loctemp)
            
            if col < IDXSEG.shape[1]:
                    IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
            else:
                new_values = [x1, y1, x2, y2, x3, y3]
                IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                IDXSEG[i, -6:] = new_values

            col += 6

            # Prepare to remove the segments order i
            # Dmask[DD < distanceToEndPt] = True
            Dmask=BWrem

        # Find isolated segment (branch point doesn't exist)
        # seems exact same as if distanceToEndPt < distanceToBranchPt, except 
        # uses normal_coord function on x[k], y[k] rather than y[k], x[k] as above -SamK
        elif np.isinf(distanceToBranchPt) and np.isfinite(distanceToEndPt):
            # Store segment real length into strahler table
            STRAHLER[i,k] = distanceToEndPt

            # Find coordinates of the closest endpoint
            ptlocy, ptlocx = np.where(DD==distanceToEndPt)
            ptloc = np.where(DD==distanceToEndPt)
            ptloc = ptloc[0]
            ptlocx = ptlocx[0]
            ptlocy = ptlocy[0]
            # # alternate:
            # E_loc_distances = DD[E_loctemp[:, 0], E_loctemp[:, 1]]
            # finite_distances = E_loc_distances[np.isfinite(end_distances)]
            # # Add check for empty array
            # if finite_distances.size == 0:
            #     # No valid distances found, skip this point
            #     continue
            # ptlocy, ptlocx = np.where(DD==np.min(finite_distances))

            # Find segment straight length
            # "Create empty image with only two points" - this is the MATLAB comment, but it seems they only create one point - SamK
            skeletoneucl = np.zeros_like(skeleton_chopped)
            skeletoneucl[B_loc[k2,0], B_loc[k2,1]] = 1
            euclD = ndimage.distance_transform_edt(skeletoneucl == 0) # for each pixel, euclidean dist to nearest non-zero value 
            disttopt = euclD[ptlocy, ptlocx]
            STRAIGHTDIST[i,k] = disttopt
            # Save index of branch point for junction angle assignment
            if np.isscalar(ptlocx) and np.isscalar(ptlocy):
                # If they're already scalars
                ptloc_linear = np.ravel_multi_index((ptlocy, ptlocx), DD.shape)
            else:
                # If they're arrays
                ptloc_linear = np.ravel_multi_index((ptlocy[0], ptlocx[0]), DD.shape)
            IDXBRANCH[i,k] = ptloc_linear

            # Create normal vectors to each segment and store their coordinates
            x1, y1, x2, y2, x3, y3, BWrem = normal_coord_test(DD, distanceToEndPt, x[k2], y[k2], limit, skeleton_chopped, B_loctemp)
            
            if col < IDXSEG.shape[1]:
                    IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
            else:
                new_values = [x1, y1, x2, y2, x3, y3]
                IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                IDXSEG[i, -6:] = new_values

            col += 6

            # Prepare to remove the segments order i
            # Dmask[DD < distanceToEndPt] = True
            Dmask=BWrem

        # Find connected segment (detect closest branch point)
        else:
            # Store segment real length into strahler table
            STRAHLER[i,k] = distanceToBranchPt

            ptlocy, ptlocx = np.where(DD==distanceToBranchPt)
            ptloc = np.where(DD==distanceToBranchPt)
            ptloc = ptloc[0]
            ptlocx = ptlocx[0]
            ptlocy = ptlocy[0]

            # Find segment straight length
            # "Create empty image with only two points" - this is the MATLAB comment, but it seems they only create one point - SamK
            skeletoneucl = np.zeros_like(skeleton_chopped)
            skeletoneucl[B_loc[k2,0], B_loc[k2,1]] = 1
            euclD = ndimage.distance_transform_edt(skeletoneucl == 0) # for each pixel, euclidean dist to nearest non-zero value 
            disttopt = euclD[ptlocy, ptlocx]
            STRAIGHTDIST[i,k] = disttopt
            # Save index of branch point for junction angle assignment
            if np.isscalar(ptlocx) and np.isscalar(ptlocy):
                # If they're already scalars
                ptloc_linear = np.ravel_multi_index((ptlocy, ptlocx), DD.shape)
            else:
                # If they're arrays
                ptloc_linear = np.ravel_multi_index((ptlocy[0], ptlocx[0]), DD.shape)
            IDXBRANCH[i,k] = ptloc_linear

            # Create normal vectors to each segment and store their coordinates
            x1, y1, x2, y2, x3, y3, BWrem = normal_coord_test(DD, distanceToBranchPt, x[k2], y[k2], limit, skeleton_chopped, B_loctemp)
            
            if col < IDXSEG.shape[1]:
                    IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
            else:
                new_values = [x1, y1, x2, y2, x3, y3]
                IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                IDXSEG[i, -6:] = new_values

            col += 6

            # Prepare to remove the segments order i
            # Dmask[DD < distanceToBranchPt] = True
            Dmask=BWrem


        # Remove order i segments
        # skelD = skeleton_chopped - Dmask # subtract matrices
        skelD = np.logical_xor(skeleton_chopped, Dmask) # or use skelD = skeleton_chopped ^ Dmask
        skeleton_chopped = skelD.astype(bool) # convert to boolean/logical, could also do skelchopped = skelD > 0 -SamK

        # Prepare the creek order plot
        creekordermask = bwmorph_diag(Dmask)
        # Assign creek orders
        creekordersing[creekordermask != 0] = creekordermask[creekordermask != 0] * i
        # Thicken the mask i times
        # creekordermask = bwmorph_thicken(creekordermask, i)
        creekordermask = dilation(creekordermask, disk(1))
        # Assign creek orders to thickened mask
        creekorder[creekordermask != 0] = creekordermask[creekordermask != 0] * i
        # Clear the mask
        del creekordermask

        # # Remove order i segments
        # skelD = skeleton_chopped - Dmask # subtract matrices
        # skelD = np.logical_xor(skeleton_chopped, Dmask) # or use skelD = skeleton_chopped ^ Dmask
        # skeleton_chopped = skelD.astype(bool) # convert to boolean/logical, could also do skelchopped = skelD > 0 -SamK

        # Redo skeleton
        skeleton_chopped = morphology.skeletonize(skeleton_chopped)

        # # Find the end and branch points of the order i+1 network configuration
        # i += 1 # Store order number and move on to the next creek order until skeleton contains only 0
        # col = 0
        skeleton_chopped = bwmorph_clean(skeleton_chopped)
        B, E, PTS = analyze_skeleton(skeleton_chopped)
        B = PTS

        i_check += 1
        if i_check >= max_iterations:
            print("Max iterations reached, breaking out of the loop.")
            break

    # while loop 4 out of 4 - loop processing - but commented out while loop in AnnaM's MATLAB code
    # so maybe we don't use a while loop, but let's see if we do -SamK
    while np.any(skeleton_chopped) and i_check<= max_iterations:
        # Process remaining loops with no nodes
        # Prepare the creek order plot
        creekordermask = bwmorph_diag(Dmask)
        # Assign creek orders
        creekordersing[creekordermask != 0] = creekordermask[creekordermask != 0] * i
        # Thicken the mask i times
        # creekordermask = bwmorph_thicken(creekordermask, i)
        creekordermask = dilation(creekordermask, disk(1))
        # Assign creek orders to thickened mask
        creekorder[creekordermask != 0] = creekordermask[creekordermask != 0] * i
        # Clear the mask
        del creekordermask
        # Get connected components
        skelcomp = bwconncomp(skeleton_chopped)
        # Get number of objects
        idxnum = skelcomp['n_objects']
        print('idxnum = ', idxnum)

        if idxnum == 0:
            # No more components to process
            break


        # Initialize temp arrays to store masks and distances
        temp_masks = []
        max_distances = []
            
        # Go through each remaining loop with no nodes
        for iii in range(idxnum):
            k = k + 1

            # Get pixel indices for current component
            idxlist = skelcomp['pixel_idx'][iii]
            if not idxlist[0].size or not idxlist[1].size:
                continue
                
            # Create mask for current component
            loopmask = np.zeros_like(skeleton_chopped, dtype=bool)
            loopmask[idxlist] = True

            # Store total pixels in component 
            sinuouslength = np.sum(loopmask, dtype=float)
            
            # Safely expand STRAHLER array if needed
            if i >= STRAHLER.shape[0] or k >= STRAHLER.shape[1]:
                new_rows = max(i + 1, STRAHLER.shape[0])
                new_cols = max(k + 1, STRAHLER.shape[1])
                STRAHLER_temp = np.zeros((new_rows, new_cols), dtype=float)  # Explicitly set dtype
                STRAHLER_temp[:STRAHLER.shape[0], :STRAHLER.shape[1]] = STRAHLER
                STRAHLER = STRAHLER_temp
                
            # Store length
            STRAHLER[i,k] = sinuouslength

            # Get coordinates of first point
            y_first = idxlist[0][0]
            x_first = idxlist[1][0]
            
            try:
                # Calculate distances
                DD = calculate_chessboard_distance(loopmask, y_first, x_first)
                
                # Find point furthest from first point
                valid_distances = DD[np.isfinite(DD)]
                if len(valid_distances) > 0:
                    max_dist = np.max(valid_distances)
                    y_far, x_far = np.where(DD == max_dist)
                    if len(y_far) > 0 and len(x_far) > 0:
                        y_far = y_far[0]
                        x_far = x_far[0]
                        
                        # Calculate normal vectors
                        try:
                            x1, y1, x2, y2, x3, y3, BWrem = normal_coord_test(
                                DD, sinuouslength, x_first, y_first, 
                                limit, skeleton_chopped, (y_far, x_far)
                            )

                            if col < IDXSEG.shape[1]:
                                IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
                            else:
                                new_values = [x1, y1, x2, y2, x3, y3]
                                IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                                IDXSEG[i, -6:] = new_values

                            col += 6

                            # # Store coordinates if successful
                            # if i < IDXSEG.shape[0]:
                            #     IDXSEG[i, :] = [x1, y1, x2, y2, x3, y3]
                        except Exception as e:
                            print(f"Error calculating normal vectors: {str(e)}")
                            continue
                            
                        # Remove processed component from skeleton
                        skeleton_chopped[loopmask] = False
                        skeleton_chopped = morphology.skeletonize(skeleton_chopped)

            except Exception as e:
                print(f"Error processing component {iii}: {str(e)}")
                continue


        i_check += 1
        if i_check >= max_iterations:
            print("Max iterations reached, breaking out of the loop.")
            break
    
    # Convert lists to numpy arrays
    STRAHLER = np.array(STRAHLER)
    STRAIGHTDIST = np.array(STRAIGHTDIST)
    IDXSEG = np.array(IDXSEG)
    IDXBRANCH = np.array(IDXBRANCH)
    
    return STRAHLER, STRAIGHTDIST, IDXSEG, IDXBRANCH, idxbreach, xbreach, ybreach, skeleton_breached, creekorder, creekordersing, all_pts, ID


def process_creek_ordering_diagnostic(ordermax, Z, skeleton, outletdetection, nbbreaches): # for debugging
    
    # Automatically detect one or several outlets (deepest end points in the skeleton)
    if outletdetection == 1:
        idxbreach, xbreach, ybreach, skeleton_breached = process_outlet_detection(Z, skeleton, nbbreaches)
    else:
        idxbreach, xbreach, ybreach, skeleton_breached = [], [], [], []

    # resave skeleton as breached skeleton to ensure outlet point is connected to skeleton
    skeleton = skeleton_breached # added to debug -SamK

    # Initialize variables
    B, E, PTS = analyze_skeleton(skeleton)
    STRAHLER = np.zeros_like(E, dtype=float)
    STRAIGHTDIST = np.zeros_like(E, dtype=float) # contains straight length
    IDXSEG = np.zeros((skeleton.shape[0], 6)) # contains coordinates of the normal vector (end and mid points)
    IDXBRANCH = np.zeros_like(skeleton, dtype=int) # contains indices of all branch points
    # IDXBRANCH = np.zeros_like(skeleton, dtype=[('row', int), ('col', int)])
    creekorder = np.zeros_like(skeleton, dtype=int)
    creekordersing = np.zeros_like(skeleton, dtype=int)
    skeleton_chopped = skeleton # we keep skeleton intact and remove segments from skeleton_chopped
    ID = []
    print('initial ID = ', ID)
    i = 1
    limit = 3

    # Get the Strahler order and segment lengths in a loop
    i_check = 1
    max_iterations = 1000  # Failsafe to prevent infinite loop
    skelprev = np.zeros_like(skeleton_chopped, dtype=bool)
    
    print('Starting while loop 1:')
    # while loop 1 out of 4
    while np.any(E) and not np.array_equal(skelprev, skeleton_chopped) and i_check<= max_iterations: # and i < ordermax - this last condition was commented out in AnnaM's MATLAB code -SamK
        print(f"while 1/4, Iteration {i_check}: {np.sum(E)} endpoints remaining")
        print('order i = ', i)

        # # Find entry channels, write them as branch points
        # # E.ravel()[idxbreach] = False # ravel flattens a 2D array
        # # B.ravel()[idxbreach] = True
        # Add breach point as branch point
        B[ybreach, xbreach] = True
        E[ybreach, xbreach] = False
       
        # start segment selection from end points
        # Find the locations of all branch and end points in the order i configuration
        B_loc = np.argwhere(B) # np.argwhere returns a 2D array of coordinates
        E_loc = np.argwhere(E)
        Dmask = np.zeros_like(skeleton_chopped, dtype=bool)
        col = 0
        print('col = ', col)
        coords = np.argwhere(E)
        y, x = coords[:, 0], coords[:, 1]
        
        for k in range(len(x)):
            print('k = ', k)
            print('E location y, x = ', (y[k], x[k]))
            figure_creek_skeleton_diagnostic(skeleton_chopped, x[k], y[k])
            # remove the selected kth end point from E_loc
            E_loctemp = np.delete(E_loc, k, axis=0) # assume E_loc is 2D array
            
            # Get distances from each pixel to kth end point
            DD = calculate_geodesic_distance(skeleton_chopped, y[k], x[k])
            
            # Retrieve distance from kth point to nearest branch point, using DD values
            branch_distances = DD[B_loc[:, 0], B_loc[:, 1]]
            finite_branch_distances = branch_distances[np.isfinite(branch_distances)]
            # have a check to make sure there are actual values
            try:
                distanceToBranchPt = np.min(finite_branch_distances).astype(float)
            except ValueError:
                distanceToBranchPt = np.inf  # or another appropriate value

            # Retrieve distance from kth point to nearest end point other than itself, using DD values
            end_distances = DD[E_loctemp[:, 0], E_loctemp[:, 1]]
            finite_end_distances = end_distances[np.isfinite(end_distances)]
            # have a check to make sure there are actual values
            try:
                distanceToEndPt = np.min(finite_end_distances).astype(float)
            except ValueError:
                distanceToEndPt = np.inf  # or another appropriate value
            
            # Three possibilities here: the end point we have detected can be:
            # 1) an isolated point (no creek segment to detect)
            # 2) an isolated creek segment delimited by two end points
            # 3) a creek segment delimited by a branch point and an end point

            # Remove isolated points (no creek segment to detect)
            if np.isinf(distanceToEndPt) and np.isinf(distanceToBranchPt):
                # Remove the point from skeleton_chopped
                skeleton_chopped[y[k], x[k]] = False
                # Remove point from endpoint array E
                E[y[k], x[k]] = False
                print(f"Removed isolated point at y={y[k]}, x={x[k]}")
                
                # Continue to next point in the original coords list
                continue
            
            # Find isolated segment (end point closer than branch point) - I guess like a vernal pool? -SamK
            elif distanceToEndPt < distanceToBranchPt:
                # Store segment real length into strahler table
                STRAHLER[i,k] = distanceToEndPt

                # Find coordinates of the closest endpoint
                ptlocy, ptlocx = np.where(DD==distanceToEndPt)
                ptloc = np.where(DD==distanceToEndPt)
                ptloc = ptloc[0]
                ptlocx = ptlocx[0]
                ptlocy = ptlocy[0]
                # # alternate:
                # E_loc_distances = DD[E_loctemp[:, 0], E_loctemp[:, 1]]
                # finite_distances = E_loc_distances[np.isfinite(end_distances)]
                # # Add check for empty array
                # if finite_distances.size == 0:
                #     # No valid distances found, skip this point
                #     continue
                # ptlocy, ptlocx = np.where(DD==np.min(finite_distances))

                # Find segment straight length
                # "Create empty image with only two points" - this is the MATLAB comment, but it seems they only create one point - SamK
                skeletoneucl = np.zeros_like(skeleton_chopped)
                skeletoneucl[E_loc[k,0], E_loc[k,1]] = 1
                euclD = ndimage.distance_transform_edt(skeletoneucl == 0) # for each pixel, euclidean dist to nearest non-zero value 
                disttopt = euclD[ptlocy, ptlocx]
                STRAIGHTDIST[i,k] = disttopt
                # Save index of branch point for junction angle assignment
                if np.isscalar(ptlocx) and np.isscalar(ptlocy):
                    # If they're already scalars
                    ptloc_linear = np.ravel_multi_index((ptlocy, ptlocx), DD.shape)
                else:
                    # If they're arrays
                    ptloc_linear = np.ravel_multi_index((ptlocy[0], ptlocx[0]), DD.shape)
                IDXBRANCH[i,k] = ptloc_linear

                # Create normal vectors to each segment and store their coordinates
                # should it be y[k], x[k] or x[k], y[k]? -SamK
                x1, y1, x2, y2, x3, y3, _, _ = normal_coord(DD, distanceToEndPt, y[k], x[k], limit)
                
                if col < IDXSEG.shape[1]:
                    IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
                else:
                    new_values = [x1, y1, x2, y2, x3, y3]
                    IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                    IDXSEG[i, -6:] = new_values

                col += 6
                print('col = ', col)
                
                # Prepare to remove the segments order i
                Dmask[DD < distanceToEndPt] = True

            # Find isolated segment (branch point doesn't exist)
            # seems exact same as if distanceToEndPt < distanceToBranchPt, except 
            # uses normal_coord function on x[k], y[k] rather than y[k], x[k] as above -SamK
            elif np.isinf(distanceToBranchPt) and np.isfinite(distanceToEndPt):
                # Store segment real length into strahler table
                STRAHLER[i,k] = distanceToEndPt

                # Find coordinates of the closest endpoint
                ptlocy, ptlocx = np.where(DD==distanceToEndPt)
                ptloc = np.where(DD==distanceToEndPt)
                ptloc = ptloc[0]
                ptlocx = ptlocx[0]
                ptlocy = ptlocy[0]
                # # alternate:
                # E_loc_distances = DD[E_loctemp[:, 0], E_loctemp[:, 1]]
                # finite_distances = E_loc_distances[np.isfinite(end_distances)]
                # # Add check for empty array
                # if finite_distances.size == 0:
                #     # No valid distances found, skip this point
                #     continue
                # ptlocy, ptlocx = np.where(DD==np.min(finite_distances))

                # Find segment straight length
                # "Create empty image with only two points" - this is the MATLAB comment, but it seems they only create one point - SamK
                skeletoneucl = np.zeros_like(skeleton_chopped)
                skeletoneucl[E_loc[k,0], E_loc[k,1]] = 1
                euclD = ndimage.distance_transform_edt(skeletoneucl == 0) # for each pixel, euclidean dist to nearest non-zero value 
                disttopt = euclD[ptlocy, ptlocx]
                STRAIGHTDIST[i,k] = disttopt
                # Save index of branch point for junction angle assignment
                if np.isscalar(ptlocx) and np.isscalar(ptlocy):
                    # If they're already scalars
                    ptloc_linear = np.ravel_multi_index((ptlocy, ptlocx), DD.shape)
                else:
                    # If they're arrays
                    ptloc_linear = np.ravel_multi_index((ptlocy[0], ptlocx[0]), DD.shape)
                IDXBRANCH[i,k] = ptloc_linear

                # Create normal vectors to each segment and store their coordinates
                x1, y1, x2, y2, x3, y3, _, _ = normal_coord(DD, distanceToEndPt, x[k], y[k], limit)
                
                if col < IDXSEG.shape[1]:
                    IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
                else:
                    new_values = [x1, y1, x2, y2, x3, y3]
                    IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                    IDXSEG[i, -6:] = new_values

                col += 6
                print('col = ', col)
                
                # Prepare to remove the segments order i
                Dmask[DD < distanceToEndPt] = True

            # Find connected segment (detect closest branch point)
            else:
                # Store segment real length into strahler table
                STRAHLER[i,k] = distanceToBranchPt

                ptlocy, ptlocx = np.where(DD==distanceToBranchPt)
                ptloc = np.where(DD==distanceToBranchPt)
                ptloc = ptloc[0]
                ptlocx = ptlocx[0]
                ptlocy = ptlocy[0]

                # Find segment straight length
                # "Create empty image with only two points" - this is the MATLAB comment, but it seems they only create one point - SamK
                skeletoneucl = np.zeros_like(skeleton_chopped)
                skeletoneucl[E_loc[k,0], E_loc[k,1]] = 1
                euclD = ndimage.distance_transform_edt(skeletoneucl == 0) # for each pixel, euclidean dist to nearest non-zero value 
                disttopt = euclD[ptlocy, ptlocx]
                STRAIGHTDIST[i,k] = disttopt
                # Save index of branch point for junction angle assignment
                if np.isscalar(ptlocx) and np.isscalar(ptlocy):
                    # If they're already scalars
                    ptloc_linear = np.ravel_multi_index((ptlocy, ptlocx), DD.shape)
                else:
                    # If they're arrays
                    ptloc_linear = np.ravel_multi_index((ptlocy[0], ptlocx[0]), DD.shape)
                IDXBRANCH[i,k] = ptloc_linear

                # Create normal vectors to each segment and store their coordinates
                # should it be y[k], x[k] or x[k], y[k]? -SamK
                x1, y1, x2, y2, x3, y3, _, _ = normal_coord(DD, distanceToBranchPt, x[k], y[k], limit)
                
                if col < IDXSEG.shape[1]:
                    IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
                else:
                    new_values = [x1, y1, x2, y2, x3, y3]
                    IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                    IDXSEG[i, -6:] = new_values

                col += 6
                print('col = ', col)
                
                # Prepare to remove the segments order i
                Dmask[DD < distanceToBranchPt] = True


            # # Plot the coordinates of the normal vector for each segment.
            # # CAUTION: uncommenting this results in 50+ figures
            # plt.figure()
            # plt.imshow(skeleton, cmap='binary')  # Use binary colormap for black/white skeleton
            # plt.plot(y1, x1, 'or', label='Center point')  # Red circle
            # plt.plot(y2, x2, '+g', label='Normal vector 1')  # Green plus
            # plt.plot(y3, x3, '+g', label='Normal vector 2')  # Green plus
            # plt.legend()
            # plt.axis('image')  # Keep aspect ratio 1:1
            # plt.show()

        # Prepare the creek order plot
        creekordermask = bwmorph_diag(Dmask)
        # Assign creek orders
        creekordersing[creekordermask != 0] = creekordermask[creekordermask != 0] * i
        # Thicken the mask i times
        # creekordermask = bwmorph_thicken(creekordermask, i)
        creekordermask = dilation(creekordermask, disk(1))
        # Assign creek orders to thickened mask
        creekorder[creekordermask != 0] = creekordermask[creekordermask != 0] * i
        # Clear the mask
        del creekordermask

        # Remove order i segments
        # skelD = skeleton_chopped - Dmask # subtract matrices
        skelD = np.logical_xor(skeleton_chopped, Dmask) # or use skelD = skeleton_chopped ^ Dmask
        skeleton_chopped = skelD.astype(bool) # convert to boolean/logical, could also do skelchopped = skelD > 0 -SamK

        # Redo skeleton
        skeleton_chopped = morphology.skeletonize(skeleton_chopped)
        ID = np.append(ID, i) # update order tracking
        print('redo skeleton ID = ', ID)

        # Find the end and branch points of the order i+1 network configuration
        i += 1 # Store order number and move on to the next creek order until skeleton contains only 0
        col = 0
        B, E, PTS = analyze_skeleton(skeleton_chopped)
        # Update breach points
        E.ravel()[idxbreach] = False
        B.ravel()[idxbreach] = True

        i_check += 1
        if i_check >= max_iterations:
            print("Max iterations reached, breaking out of the loop.")
            break

    if i == ordermax:
        ID = np.append(ID, i)
        print('i == ordermax ID = ', ID)
    else:
        i = ordermax
        ID = np.arange(1, ordermax + 1)  # Create array from 1 to ordermax
        print('i != ordermax ID = ', ID)

    # Segment assignation has occured until no more end points are detected. 
    # We do a second one that also detects the outlets.
    # Segment assignation for outlets
    skelprev = np.zeros_like(skeleton, dtype=bool)
    B, E, PTS = analyze_skeleton(skeleton_chopped)
    ESIZE = np.any(E) # doesn't seem to be used later -SamK
    test = 1

    i_check = 1

    print('Starting while loop 2:')
    # while loop 2 out of 4 - outlets
    # ensure there are endpoints and that skelprev and skeleton_chopped are not identical
    while np.any(E) and not np.array_equal(skelprev, skeleton_chopped) and i_check<= max_iterations:
        print(f"while 2/4, Iteration {i_check}: {np.sum(E)} endpoints remaining")
        print('order i = ', i)
        test = test + 1

        # Find the locations of all branch and end points in the order i configuration
        B_loc = np.argwhere(B) # np.argwhere returns a 2D array of coordinates
        E_loc = np.argwhere(E)
        Dmask = np.zeros_like(skeleton_chopped, dtype=bool)
        # col = 0
        # print('col = ', col)
        coords = np.argwhere(E)
        y, x = coords[:, 0], coords[:, 1]
        
        # Loop through each end point in order i configuration
        for k2 in range(len(x)):
            k = k + 1

            # print figure of current study point in current skeleton
            print('k = ', k)
            print('k2 = ', k2)
            print('E location y[k2], x[k2] = ', (y[k2], x[k2]))
            figure_creek_skeleton_diagnostic(skeleton_chopped, x[k2], y[k2])

            # Remove selected end point from E_loc
            E_loctemp = np.delete(E_loc, k2, axis=0) # assume E_loc is 2D array

            # get quasi-euclidean distance from selected end point to the rest of the creek network and find closest branch or end point
            # Get distances to kth end point
            DD = calculate_geodesic_distance(skeleton_chopped, y[k2], x[k2])
            
            # Retrieve distance from kth point to nearest branch point, using DD values
            branch_distances = DD[B_loc[:, 0], B_loc[:, 1]]
            finite_branch_distances = branch_distances[np.isfinite(branch_distances)]
            # have a check to make sure there are actual values
            try:
                distanceToBranchPt = np.min(finite_branch_distances).astype(float)
            except ValueError:
                distanceToBranchPt = np.inf  # or another appropriate value

            # Retrieve distance from kth point to nearest end point other than itself, using DD values
            end_distances = DD[E_loctemp[:, 0], E_loctemp[:, 1]]
            finite_end_distances = end_distances[np.isfinite(end_distances)]
            # have a check to make sure there are actual values
            try:
                distanceToEndPt = np.min(finite_end_distances).astype(float)
            except ValueError:
                distanceToEndPt = np.inf  # or another appropriate value
            
            # Three possibilities here: the end point we have detected can be:
            # 1) an isolated point (no creek segment to detect)
            # 2) an isolated creek segment delimited by two end points
            # 3) a creek segment delimited by a branch point and an end point

            # Remove isolated points (no creek segment to detect)
            if np.isinf(distanceToEndPt) and np.isinf(distanceToBranchPt):
                # Remove the point from skeleton_chopped
                skeleton_chopped[y[k2], x[k2]] = False
                # Remove point from endpoint array E
                E[y[k2], x[k2]] = False
                print(f"Removed isolated point at y={y[k2]}, x={x[k2]}")
                
                # Continue to next point in the original coords list
                continue
            
            # Find isolated segment (end point closer than branch point) - I guess like a vernal pool? -SamK
            elif distanceToEndPt < distanceToBranchPt:
                # Store segment real length into strahler table
                STRAHLER[i,k] = distanceToEndPt

                # Find coordinates of the closest endpoint
                ptlocy, ptlocx = np.where(DD==distanceToEndPt)
                ptloc = np.where(DD==distanceToEndPt)
                ptloc = ptloc[0]
                ptlocx = ptlocx[0]
                ptlocy = ptlocy[0]
                # # alternate:
                # E_loc_distances = DD[E_loctemp[:, 0], E_loctemp[:, 1]]
                # finite_distances = E_loc_distances[np.isfinite(end_distances)]
                # # Add check for empty array
                # if finite_distances.size == 0:
                #     # No valid distances found, skip this point
                #     continue
                # ptlocy, ptlocx = np.where(DD==np.min(finite_distances))

                # Find segment straight length
                # "Create empty image with only two points" - this is the MATLAB comment, but it seems they only create one point - SamK
                skeletoneucl = np.zeros_like(skeleton_chopped)
                skeletoneucl[E_loc[k2,0], E_loc[k2,1]] = 1
                euclD = ndimage.distance_transform_edt(skeletoneucl == 0) # for each pixel, euclidean dist to nearest non-zero value 
                disttopt = euclD[ptlocy, ptlocx]
                STRAIGHTDIST[i,k] = disttopt
                # Save index of branch point for junction angle assignment
                if np.isscalar(ptlocx) and np.isscalar(ptlocy):
                    # If they're already scalars
                    ptloc_linear = np.ravel_multi_index((ptlocy, ptlocx), DD.shape)
                else:
                    # If they're arrays
                    ptloc_linear = np.ravel_multi_index((ptlocy[0], ptlocx[0]), DD.shape)
                IDXBRANCH[i,k] = ptloc_linear

                # Create normal vectors to each segment and store their coordinates
                # should it be y[k2], x[k2] or x[k2], y[k2]? -SamK
                x1, y1, x2, y2, x3, y3, _, _ = normal_coord(DD, distanceToEndPt, y[k2], x[k2], limit)
                
                if col < IDXSEG.shape[1]:
                    IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
                else:
                    new_values = [x1, y1, x2, y2, x3, y3]
                    IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                    IDXSEG[i, -6:] = new_values

                col += 6
                print('col = ', col)

                # Prepare to remove the segments order i
                Dmask[DD < distanceToEndPt] = True

            # Find isolated segment (branch point doesn't exist)
            # seems exact same as if distanceToEndPt < distanceToBranchPt, except 
            # uses normal_coord function on x[k], y[k] rather than y[k], x[k] as above -SamK
            elif np.isinf(distanceToBranchPt) and np.isfinite(distanceToEndPt):
                # Store segment real length into strahler table
                STRAHLER[i,k] = distanceToEndPt

                # Find coordinates of the closest endpoint
                ptlocy, ptlocx = np.where(DD==distanceToEndPt)
                ptloc = np.where(DD==distanceToEndPt)
                ptloc = ptloc[0]
                ptlocx = ptlocx[0]
                ptlocy = ptlocy[0]
                # # alternate:
                # E_loc_distances = DD[E_loctemp[:, 0], E_loctemp[:, 1]]
                # finite_distances = E_loc_distances[np.isfinite(end_distances)]
                # # Add check for empty array
                # if finite_distances.size == 0:
                #     # No valid distances found, skip this point
                #     continue
                # ptlocy, ptlocx = np.where(DD==np.min(finite_distances))

                # Find segment straight length
                # "Create empty image with only two points" - this is the MATLAB comment, but it seems they only create one point - SamK
                skeletoneucl = np.zeros_like(skeleton_chopped)
                skeletoneucl[E_loc[k2,0], E_loc[k2,1]] = 1
                euclD = ndimage.distance_transform_edt(skeletoneucl == 0) # for each pixel, euclidean dist to nearest non-zero value 
                disttopt = euclD[ptlocy, ptlocx]
                STRAIGHTDIST[i,k] = disttopt
                # Save index of branch point for junction angle assignment
                if np.isscalar(ptlocx) and np.isscalar(ptlocy):
                    # If they're already scalars
                    ptloc_linear = np.ravel_multi_index((ptlocy, ptlocx), DD.shape)
                else:
                    # If they're arrays
                    ptloc_linear = np.ravel_multi_index((ptlocy[0], ptlocx[0]), DD.shape)
                IDXBRANCH[i,k] = ptloc_linear

                # Create normal vectors to each segment and store their coordinates
                x1, y1, x2, y2, x3, y3, _, _ = normal_coord(DD, distanceToEndPt, x[k2], y[k2], limit)
               
                if col < IDXSEG.shape[1]:
                    IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
                else:
                    new_values = [x1, y1, x2, y2, x3, y3]
                    IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                    IDXSEG[i, -6:] = new_values

                col += 6
                print('col = ', col)

                # Prepare to remove the segments order i
                Dmask[DD < distanceToEndPt] = True

            # Find connected segment (detect closest branch point)
            else:
                # Store segment real length into strahler table
                STRAHLER[i,k] = distanceToBranchPt

                ptlocy, ptlocx = np.where(DD==distanceToBranchPt)
                ptloc = np.where(DD==distanceToBranchPt)
                ptloc = ptloc[0]
                ptlocx = ptlocx[0]
                ptlocy = ptlocy[0]

                # Find segment straight length
                # "Create empty image with only two points" - this is the MATLAB comment, but it seems they only create one point - SamK
                skeletoneucl = np.zeros_like(skeleton_chopped)
                skeletoneucl[E_loc[k2,0], E_loc[k2,1]] = 1
                euclD = ndimage.distance_transform_edt(skeletoneucl == 0) # for each pixel, euclidean dist to nearest non-zero value 
                disttopt = euclD[ptlocy, ptlocx]
                STRAIGHTDIST[i,k] = disttopt
                # Save index of branch point for junction angle assignment
                if np.isscalar(ptlocx) and np.isscalar(ptlocy):
                    # If they're already scalars
                    ptloc_linear = np.ravel_multi_index((ptlocy, ptlocx), DD.shape)
                else:
                    # If they're arrays
                    ptloc_linear = np.ravel_multi_index((ptlocy[0], ptlocx[0]), DD.shape)
                IDXBRANCH[i,k] = ptloc_linear

                # Create normal vectors to each segment and store their coordinates
                # should it be y[k], x[k] or x[k], y[k]? -SamK
                x1, y1, x2, y2, x3, y3, _, _ = normal_coord(DD, distanceToBranchPt, x[k2], y[k2], limit)
                
                if col < IDXSEG.shape[1]:
                    IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
                else:
                    new_values = [x1, y1, x2, y2, x3, y3]
                    IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                    IDXSEG[i, -6:] = new_values

                col += 6
                print('col = ', col)
                
                # Prepare to remove the segments order i
                Dmask[DD < distanceToBranchPt] = True


            # # Plot the coordinates of the normal vector for each segment.
            # # CAUTION: uncommenting this results in 50+ figures
            # plt.figure()
            # plt.imshow(skeleton, cmap='binary')  # Use binary colormap for black/white skeleton
            # plt.plot(y1, x1, 'or', label='Center point')  # Red circle
            # plt.plot(y2, x2, '+g', label='Normal vector 1')  # Green plus
            # plt.plot(y3, x3, '+g', label='Normal vector 2')  # Green plus
            # plt.legend()
            # plt.axis('image')  # Keep aspect ratio 1:1
            # plt.show()

        # Prepare the creek order plot
        creekordermask = bwmorph_diag(Dmask)
        # Assign creek orders
        creekordersing[creekordermask != 0] = creekordermask[creekordermask != 0] * i
        # Thicken the mask i times
        # creekordermask = bwmorph_thicken(creekordermask, i)
        creekordermask = dilation(creekordermask, disk(1))
        # Assign creek orders to thickened mask
        creekorder[creekordermask != 0] = creekordermask[creekordermask != 0] * i
        # Clear the mask
        del creekordermask

        # Remove order i segments
        # skelD = skeleton_chopped - Dmask # subtract matrices
        skelD = np.logical_xor(skeleton_chopped, Dmask) # or use skelD = skeleton_chopped ^ Dmask
        skeleton_chopped = skelD.astype(bool) # convert to boolean/logical, could also do skelchopped = skelD > 0 -SamK

        # Redo skeleton
        skeleton_chopped = morphology.skeletonize(skeleton_chopped)
        # ID = np.append(ID, i) # update order tracking

        # Find the end and branch points of the order i+1 network configuration
        # i += 1 # Store order number and move on to the next creek order until skeleton contains only 0
        # col = 0
        B, E, PTS = analyze_skeleton(skeleton_chopped)

        i_check += 1
        if i_check >= max_iterations:
            print("Max iterations reached, breaking out of the loop.")
            break
    
    # Outlet Segment assignation has occured until no more end points are detected. 
    # We do a THIRD one that detects all branch points and gets interconnected sections.

    skelprev = np.zeros_like(skeleton, dtype=bool)
    B, E, PTS = analyze_skeleton(skeleton_chopped)
    B[E != 0] = True # could also do: B[np.where(E)] = True -SamK
    # # Update breach points
    # E.ravel()[idxbreach] = False
    # B.ravel()[idxbreach] = True

    # Figure - not translated as of now, commented out in AnnaM's MATLAB code -SamK

    i_check = 1

    print('Starting while loop 3:')
    # while loop 3 out of 4 - branch points / interconnected segments
    while np.any(B) and i_check<= max_iterations:
        print(f"while 3/4, Iteration {i_check}: {np.sum(B)} endpoints remaining")
        print('order i = ', i)
        # Select 1 node
        k2 = 1

        # Find the locations of all branch and end points in the order i configuration
        B_loc = np.argwhere(B) # np.argwhere returns a 2D array of coordinates
        # E_loc = np.argwhere(E)
        Dmask = np.zeros_like(skeleton_chopped, dtype=bool)
        # col = 0
        # print('col = ', col)
        coords = np.argwhere(B) # different than being based on E in first two while loops -SamK
        y, x = coords[:, 0], coords[:, 1]

        # "Plot the order i configuration" and "Check for interconnections" commented out in AnnaM's MATLAB code -SamK
        
        # for k2 in range(len(x)): # commented out in AnnaM's MATLAB code -SamK
        k = k+1

        # print figure of current study point in current skeleton
        print('k = ', k)
        print('k2 = ', k2)
        print('E location y[k2], x[k2] = ', (y[k2], x[k2]))
        figure_creek_skeleton_diagnostic(skeleton_chopped, x[k2], y[k2])

        # remove the selected kth end point from E_loc
        E_loctemp = np.delete(E_loc, k2, axis=0) # assume E_loc is 2D array
        B_loctemp = np.delete(B_loc, k2, axis=0) # assume B_loc is 2D array

        y2 = x
        x = y
        y = y2

        # get quasi-euclidean distance from selected end point to the 
        # rest of the creek network and find closest branch or end point
        # Get distances
        DD = calculate_geodesic_distance(skeleton_chopped, y[k2], x[k2]) 
        # commented out figure -SamK
            
        # Retrieve distance from kth point to nearest branch point, using DD values
        branch_distances = DD[B_loctemp[:, 0], B_loctemp[:, 1]]
        finite_branch_distances = branch_distances[np.isfinite(branch_distances)]
        # have a check to make sure there are actual values
        try:
            distanceToBranchPt = np.min(finite_branch_distances).astype(float)
        except ValueError:
            distanceToBranchPt = np.inf  # or another appropriate value

        # Retrieve distance from kth point to nearest end point other than itself, using DD values
        end_distances = DD[E_loctemp[:, 0], E_loctemp[:, 1]]
        finite_end_distances = end_distances[np.isfinite(end_distances)]
        # have a check to make sure there are actual values
        try:
            distanceToEndPt = np.min(finite_end_distances).astype(float)
        except ValueError:
            distanceToEndPt = np.inf  # or another appropriate value
            
        # Three possibilities here: the end point we have detected can be:
        # 1) an isolated point (no creek segment to detect)
        # 2) an isolated creek segment delimited by two end points
        # 3) a creek segment delimited by a branch point and an end point

        # Remove isolated points (no creek segment to detect)
        if np.isinf(distanceToEndPt) and np.isinf(distanceToBranchPt):
            # Remove the point from skeleton_chopped
            skeleton_chopped[y[k2], x[k2]] = False
            # Remove point from endpoint array E
            E[y[k2], x[k2]] = False
            print(f"Removed isolated point at y={y[k2]}, x={x[k2]}")
            
            # Continue to next point in the original coords list
            continue
        
        # Find isolated segment (end point closer than branch point) - I guess like a vernal pool? -SamK
        elif distanceToEndPt < distanceToBranchPt:
            # Store segment real length into strahler table
            STRAHLER[i,k] = distanceToEndPt

            # Find coordinates of the closest endpoint
            ptlocy, ptlocx = np.where(DD==distanceToEndPt)
            ptloc = np.where(DD==distanceToEndPt)
            ptloc = ptloc[0]
            ptlocx = ptlocx[0]
            ptlocy = ptlocy[0]
            # # alternate:
            # E_loc_distances = DD[E_loctemp[:, 0], E_loctemp[:, 1]]
            # finite_distances = E_loc_distances[np.isfinite(end_distances)]
            # # Add check for empty array
            # if finite_distances.size == 0:
            #     # No valid distances found, skip this point
            #     continue
            # ptlocy, ptlocx = np.where(DD==np.min(finite_distances))

            # Find segment straight length
            # "Create empty image with only two points" - this is the MATLAB comment, but it seems they only create one point - SamK
            skeletoneucl = np.zeros_like(skeleton_chopped)
            skeletoneucl[B_loc[k2,0], B_loc[k2,1]] = 1
            euclD = ndimage.distance_transform_edt(skeletoneucl == 0) # for each pixel, euclidean dist to nearest non-zero value 
            disttopt = euclD[ptlocy, ptlocx]
            STRAIGHTDIST[i,k] = disttopt
            # Save index of branch point for junction angle assignment
            if np.isscalar(ptlocx) and np.isscalar(ptlocy):
                # If they're already scalars
                ptloc_linear = np.ravel_multi_index((ptlocy, ptlocx), DD.shape)
            else:
                # If they're arrays
                ptloc_linear = np.ravel_multi_index((ptlocy[0], ptlocx[0]), DD.shape)
            IDXBRANCH[i,k] = ptloc_linear

            # Create normal vectors to each segment and store their coordinates
            print('while loop 3 if part 1: normal_coord_test inputs:')
            print('   ptlocy, ptlocx = y[k2], x[k2] = ', (y[k2], x[k2]))
            print('   B_loc = B_loctemp = ', B_loctemp)
            x1, y1, x2, y2, x3, y3, BWrem = normal_coord_test(DD, distanceToEndPt, x[k2], y[k2], limit, skeleton_chopped, B_loctemp)
            
            if col < IDXSEG.shape[1]:
                IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
            else:
                new_values = [x1, y1, x2, y2, x3, y3]
                IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                IDXSEG[i, -6:] = new_values

            col += 6
            print('col = ', col)

            # Prepare to remove the segments order i
            # Dmask[DD < distanceToEndPt] = True
            Dmask=BWrem

        # Find isolated segment (branch point doesn't exist)
        # seems exact same as if distanceToEndPt < distanceToBranchPt, except 
        # uses normal_coord function on x[k], y[k] rather than y[k], x[k] as above -SamK
        elif np.isinf(distanceToBranchPt) and np.isfinite(distanceToEndPt):
            # Store segment real length into strahler table
            STRAHLER[i,k] = distanceToEndPt

            # Find coordinates of the closest endpoint
            ptlocy, ptlocx = np.where(DD==distanceToEndPt)
            ptloc = np.where(DD==distanceToEndPt)
            ptloc = ptloc[0]
            ptlocx = ptlocx[0]
            ptlocy = ptlocy[0]
            # # alternate:
            # E_loc_distances = DD[E_loctemp[:, 0], E_loctemp[:, 1]]
            # finite_distances = E_loc_distances[np.isfinite(end_distances)]
            # # Add check for empty array
            # if finite_distances.size == 0:
            #     # No valid distances found, skip this point
            #     continue
            # ptlocy, ptlocx = np.where(DD==np.min(finite_distances))

            # Find segment straight length
            # "Create empty image with only two points" - this is the MATLAB comment, but it seems they only create one point - SamK
            skeletoneucl = np.zeros_like(skeleton_chopped)
            skeletoneucl[B_loc[k2,0], B_loc[k2,1]] = 1
            euclD = ndimage.distance_transform_edt(skeletoneucl == 0) # for each pixel, euclidean dist to nearest non-zero value 
            disttopt = euclD[ptlocy, ptlocx]
            STRAIGHTDIST[i,k] = disttopt
            # Save index of branch point for junction angle assignment
            if np.isscalar(ptlocx) and np.isscalar(ptlocy):
                # If they're already scalars
                ptloc_linear = np.ravel_multi_index((ptlocy, ptlocx), DD.shape)
            else:
                # If they're arrays
                ptloc_linear = np.ravel_multi_index((ptlocy[0], ptlocx[0]), DD.shape)
            IDXBRANCH[i,k] = ptloc_linear

            # Create normal vectors to each segment and store their coordinates
            print('while loop 3 if part 2: normal_coord_test inputs:')
            print('   ptlocy, ptlocx = y[k2], x[k2] = ', (y[k2], x[k2]))
            print('   B_loc = B_loctemp = ', B_loctemp)
            x1, y1, x2, y2, x3, y3, BWrem = normal_coord_test(DD, distanceToEndPt, x[k2], y[k2], limit, skeleton_chopped, B_loctemp)
            
            if col < IDXSEG.shape[1]:
                IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
            else:
                new_values = [x1, y1, x2, y2, x3, y3]
                IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                IDXSEG[i, -6:] = new_values

            col += 6
            print('col = ', col)

            # Prepare to remove the segments order i
            # Dmask[DD < distanceToEndPt] = True
            Dmask=BWrem

        # Find connected segment (detect closest branch point)
        else:
            # Store segment real length into strahler table
            STRAHLER[i,k] = distanceToBranchPt

            ptlocy, ptlocx = np.where(DD==distanceToBranchPt)
            ptloc = np.where(DD==distanceToBranchPt)
            ptloc = ptloc[0]
            ptlocx = ptlocx[0]
            ptlocy = ptlocy[0]

            # Find segment straight length
            # "Create empty image with only two points" - this is the MATLAB comment, but it seems they only create one point - SamK
            skeletoneucl = np.zeros_like(skeleton_chopped)
            skeletoneucl[B_loc[k2,0], B_loc[k2,1]] = 1
            euclD = ndimage.distance_transform_edt(skeletoneucl == 0) # for each pixel, euclidean dist to nearest non-zero value 
            disttopt = euclD[ptlocy, ptlocx]
            STRAIGHTDIST[i,k] = disttopt
            # Save index of branch point for junction angle assignment
            if np.isscalar(ptlocx) and np.isscalar(ptlocy):
                # If they're already scalars
                ptloc_linear = np.ravel_multi_index((ptlocy, ptlocx), DD.shape)
            else:
                # If they're arrays
                ptloc_linear = np.ravel_multi_index((ptlocy[0], ptlocx[0]), DD.shape)
            IDXBRANCH[i,k] = ptloc_linear

            # Create normal vectors to each segment and store their coordinates
            print('while loop 3 if part 3: normal_coord_test inputs:')
            print('   ptlocy, ptlocx = y[k2], x[k2] = ', (y[k2], x[k2]))
            print('   B_loc = B_loctemp = ', B_loctemp)
            x1, y1, x2, y2, x3, y3, BWrem = normal_coord_test(DD, distanceToBranchPt, x[k2], y[k2], limit, skeleton_chopped, B_loctemp)
            
            if col < IDXSEG.shape[1]:
                IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
            else:
                new_values = [x1, y1, x2, y2, x3, y3]
                IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                IDXSEG[i, -6:] = new_values

            col += 6
            print('col = ', col)

            # Prepare to remove the segments order i
            # Dmask[DD < distanceToBranchPt] = True
            Dmask=BWrem


            # # Plot the coordinates of the normal vector for each segment.
            # # CAUTION: uncommenting this results in 50+ figures
            # plt.figure()
            # plt.imshow(skeleton, cmap='binary')  # Use binary colormap for black/white skeleton
            # plt.plot(y1, x1, 'or', label='Center point')  # Red circle
            # plt.plot(y2, x2, '+g', label='Normal vector 1')  # Green plus
            # plt.plot(y3, x3, '+g', label='Normal vector 2')  # Green plus
            # plt.legend()
            # plt.axis('image')  # Keep aspect ratio 1:1
            # plt.show()

        # Remove order i segments
        # skelD = skeleton_chopped - Dmask # subtract matrices
        skelD = np.logical_xor(skeleton_chopped, Dmask) # or use skelD = skeleton_chopped ^ Dmask
        skeleton_chopped = skelD.astype(bool) # convert to boolean/logical, could also do skelchopped = skelD > 0 -SamK

        # Prepare the creek order plot
        creekordermask = bwmorph_diag(Dmask)
        # Assign creek orders
        creekordersing[creekordermask != 0] = creekordermask[creekordermask != 0] * i
        # Thicken the mask i times
        # creekordermask = bwmorph_thicken(creekordermask, i)
        creekordermask = dilation(creekordermask, disk(1))
        # Assign creek orders to thickened mask
        creekorder[creekordermask != 0] = creekordermask[creekordermask != 0] * i
        # Clear the mask
        del creekordermask

        # # Remove order i segments
        # skelD = skeleton_chopped - Dmask # subtract matrices
        # skelD = np.logical_xor(skeleton_chopped, Dmask) # or use skelD = skeleton_chopped ^ Dmask
        # skeleton_chopped = skelD.astype(bool) # convert to boolean/logical, could also do skelchopped = skelD > 0 -SamK

        # Redo skeleton
        skeleton_chopped = morphology.skeletonize(skeleton_chopped)

        # # Find the end and branch points of the order i+1 network configuration
        # i += 1 # Store order number and move on to the next creek order until skeleton contains only 0
        # col = 0
        skeleton_chopped = bwmorph_clean(skeleton_chopped)
        B, E, PTS = analyze_skeleton(skeleton_chopped)
        B = PTS

        i_check += 1
        if i_check >= max_iterations:
            print("Max iterations reached, breaking out of the loop.")
            break

    print('Starting while loop 4:')
    # while loop 4 out of 4 - loop processing - but commented out while loop in AnnaM's MATLAB code
    # so maybe we don't use a while loop, but let's see if we do -SamK
    while np.any(skeleton_chopped) and i_check<= max_iterations:
        print(f"while 4/4, Iteration {i_check}: {np.sum(skeleton_chopped)} skeleton_chopped points remaining")
        print('order i = ', i)
        # Process remaining loops with no nodes
        # Prepare the creek order plot
        creekordermask = bwmorph_diag(Dmask)
        # Assign creek orders
        creekordersing[creekordermask != 0] = creekordermask[creekordermask != 0] * i
        # Thicken the mask i times
        # creekordermask = bwmorph_thicken(creekordermask, i)
        creekordermask = dilation(creekordermask, disk(1))
        # Assign creek orders to thickened mask
        creekorder[creekordermask != 0] = creekordermask[creekordermask != 0] * i
        # Clear the mask
        del creekordermask
        # Get connected components
        skelcomp = bwconncomp(skeleton_chopped)
        # Get number of objects
        idxnum = skelcomp['n_objects']
        print('idxnum = ', idxnum)

        if idxnum == 0:
            # No more components to process
            break

        # # Go through each remaining loop with no nodes
        # for iii in range(0, idxnum + 1): # MATLAB goes 1:idxnum -SamK
        #     print('iii = ', iii)
        #     k = k + 1

        #     # Define 2 nodes
        #     idxlist = skelcomp['pixel_idx'][iii]
        #     # if idxlist is empty, pass on this iii
        #     if not idxlist[0].size or not idxlist[1].size:
        #         continue
        #     loopmask = np.zeros_like(skeleton_chopped, dtype=bool)
        #     loopmask[idxlist] = True
        #     print('idxlist = ', idxlist)

        #     sinuouslength = np.sum(loopmask) # or use sinuouslength = np.nansum(loopmask.ravel()) if loopmask not boolean
        #     STRAHLER[i,k] = sinuouslength # store segment length into strahler table
        #     halflg = round(sinuouslength/2) # doesn't seem to be used later -SamK
        #     # Defined first point
        #     ptmask = np.zeros_like(skeleton_chopped, dtype=bool)
        #     ptmask[idxlist[0][0], idxlist[1][0]] = True
        #     ptlocy, ptlocx = np.where(ptmask)
        #     ptloc = np.where(ptmask)
        #     ptloc = ptloc[0]
        #     ptlocx = ptlocx[0]
        #     ptlocy = ptlocy[0]
        #     # Define second point halfway across the loop
        #     DD = calculate_chessboard_distance(loopmask, ptlocy, ptlocx) # I *think* should be input as ptlocy, ptlocx? -SamK
        #     midpt = round(np.nanmean(DD)) # 1. Find value equal to rounded mean
        #     linear_indices = np.where(DD.ravel() == midpt)[0] # 2. Get linear indices where Dtemp equals this value
        #     mid_idx = linear_indices[len(linear_indices)//2]  # 3. Take middle element of these indices
        #     row_midpt, col_midpt = np.unravel_index(mid_idx, DD.shape) # 4. Convert back to 2D coordinates

        #     # print figure of current study point in current skeleton
        #     print('k = ', k)
        #     figure_creek_skeleton_diagnostic(skeleton_chopped, ptlocx, ptlocy)
            
        #     print(f"mid_idx: {mid_idx}")
        #     print(f"DD.shape: {DD.shape}")
        #     print(f"row_midpt, col_midpt: {row_midpt}, {col_midpt}")
            
        #     # Find segment straight length
        #     # "Create empty image with only two points" - this is the MATLAB comment, but it seems they only create one point - SamK
        #     skeletoneucl = np.zeros_like(skeleton_chopped)
        #     # row_mid, col_mid = np.where(DD == midpt)  # get coordinates where value appears
        #     # Set that location to 1
        #     # skeletoneucl[row_mid, col_mid] = 1
        #     skeletoneucl[row_midpt, col_midpt] = 1
        #     euclD = ndimage.distance_transform_edt(skeletoneucl == 0) # for each pixel, euclidean dist to nearest non-zero value 
        #     disttopt = euclD[ptlocy, ptlocx]

        #     STRAIGHTDIST[i,k] = np.nan # No sinuosity for loops

        #     # # Save index of branch point for junction angle assignment
        #     # IDXBRANCH[i,k] = ptloc

        #     # Create normal vectors to each segment and store their coordinates
        #     print('while loop 4: normal_coord_test inputs:')
        #     print('   ptlocy, ptlocx = ptlocy, ptlocx = ', (ptlocy, ptlocx))
        #     print('   B_loc = (row_midpt, col_midpt) = ', (row_midpt, col_midpt))
        #     x1, y1, x2, y2, x3, y3, BWrem = normal_coord_test(DD, sinuouslength, ptlocx, ptlocy, limit, skeleton_chopped, [row_midpt, col_midpt])
            # if col < IDXSEG.shape[1]:
            #     IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
            # else:
            #     new_values = [x1, y1, x2, y2, x3, y3]
            #     IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
            #     IDXSEG[i, -6:] = new_values

            # col += 6
            # print('col = ', col)
            # print('col = ', col)


        # Initialize temp arrays to store masks and distances
        temp_masks = []
        max_distances = []
            
        # Go through each remaining loop with no nodes
        for iii in range(idxnum):
            print('iii = ', iii)
            k = k + 1

            # Get pixel indices for current component
            idxlist = skelcomp['pixel_idx'][iii]
            if not idxlist[0].size or not idxlist[1].size:
                continue
                
            # Create mask for current component
            loopmask = np.zeros_like(skeleton_chopped, dtype=bool)
            loopmask[idxlist] = True
            print('idxlist = ', idxlist)

            # Store total pixels in component 
            sinuouslength = np.sum(loopmask, dtype=float)
            
            # Safely expand STRAHLER array if needed
            if i >= STRAHLER.shape[0] or k >= STRAHLER.shape[1]:
                new_rows = max(i + 1, STRAHLER.shape[0])
                new_cols = max(k + 1, STRAHLER.shape[1])
                STRAHLER_temp = np.zeros((new_rows, new_cols), dtype=float)  # Explicitly set dtype
                STRAHLER_temp[:STRAHLER.shape[0], :STRAHLER.shape[1]] = STRAHLER
                STRAHLER = STRAHLER_temp
                
            # Store length
            STRAHLER[i,k] = sinuouslength

            # Get coordinates of first point
            y_first = idxlist[0][0]
            x_first = idxlist[1][0]
            
            try:
                # Calculate distances
                DD = calculate_chessboard_distance(loopmask, y_first, x_first)
                
                # Find point furthest from first point
                valid_distances = DD[np.isfinite(DD)]
                if len(valid_distances) > 0:
                    max_dist = np.max(valid_distances)
                    y_far, x_far = np.where(DD == max_dist)
                    if len(y_far) > 0 and len(x_far) > 0:
                        y_far = y_far[0]
                        x_far = x_far[0]
                        
                        # Calculate normal vectors
                        try:
                            x1, y1, x2, y2, x3, y3, BWrem = normal_coord_test(
                                DD, sinuouslength, x_first, y_first, 
                                limit, skeleton_chopped, (y_far, x_far)
                            )

                            if col < IDXSEG.shape[1]:
                                IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
                            else:
                                new_values = [x1, y1, x2, y2, x3, y3]
                                IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                                IDXSEG[i, -6:] = new_values

                            col += 6
                            print('col = ', col)

                            # # Store coordinates if successful
                            # if i < IDXSEG.shape[0]:
                            #     IDXSEG[i, :] = [x1, y1, x2, y2, x3, y3]
                        except Exception as e:
                            print(f"Error calculating normal vectors: {str(e)}")
                            continue
                            
                        # Remove processed component from skeleton
                        skeleton_chopped[loopmask] = False
                        skeleton_chopped = morphology.skeletonize(skeleton_chopped)

            except Exception as e:
                print(f"Error processing component {iii}: {str(e)}")
                continue


        i_check += 1
        if i_check >= max_iterations:
            print("Max iterations reached, breaking out of the loop.")
            break
    
    # Convert lists to numpy arrays
    STRAHLER = np.array(STRAHLER)
    STRAIGHTDIST = np.array(STRAIGHTDIST)
    IDXSEG = np.array(IDXSEG)
    IDXBRANCH = np.array(IDXBRANCH)
    
    return STRAHLER, STRAIGHTDIST, IDXSEG, IDXBRANCH, idxbreach, xbreach, ybreach, skeleton_breached, creekorder, creekordersing, PTS, ID

def process_outlet_detection(Z, skeleton, nbbreaches):
    # uses slightly different process than MATLAB, but that's okay because the important output is the breach location -SamK
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

    # Create new border mask 4 pixels inward
    # 4 pixels comes from MATLAB code: contourmask = bwmorph(contourmask, 'thicken', 4); -SamK
    border_mask_inward = (distance_from_border >= 4) & (distance_from_border <= 5)
    border_mask_inward = border_mask_inward & landmask

    # Create a distance map from the contracted border
    distance_from_edge = ndimage.distance_transform_edt(border_mask_inward)
    
    # Consider points close to the edge (within 5 pixels)
    # 5 pixels comes from MATLAB code: PTSclusters = bwmorph(PTSperim, 'thicken', 5); -SamK
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
        skeleton_breached = skeleton
        
        # Add the new border point to our list
        new_ybreach.append(new_y)
        new_xbreach.append(new_x)
    
    # Convert lists back to arrays
    new_ybreach = np.array(new_ybreach)
    new_xbreach = np.array(new_xbreach)
    
    # Create the breach indices
    idxbreach = np.ravel_multi_index((new_ybreach, new_xbreach), Z.shape)
    # print('new_xbreach = ', new_xbreach)
    # print('new_ybreach = ', new_ybreach)
    # print('idxbreach = ', idxbreach)
    
    # # Visualization
    # plt.figure(figsize=(12, 12))
    # plt.imshow(landmask, cmap='gray')
    # plt.imshow(skeleton, cmap='Blues', alpha=0.5)
    # plt.plot(new_xbreach, new_ybreach, 'r+', markersize=20, markeredgewidth=2, label='New Outlets')
    # plt.plot(xbreach, ybreach, 'g+', markersize=15, markeredgewidth=2, label='Original Outlets')
    # plt.legend()
    # plt.title('Outlet Detection with Border Correction')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(12, 12))
    # plt.imshow(landmask, cmap='gray')
    # plt.imshow(skeleton, cmap='Blues', alpha=0.5)
    # plt.title('Outlet Detection Final Skeleton')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()
    
    return idxbreach, new_xbreach, new_ybreach, skeleton_breached

def normal_coord(D, distanceToPt, x, y, limit):
    """
    Find normal vectors to creek segments and their coordinates.
    
    Parameters:
    -----------
    D : ndarray
        A matrix or nested array of raster distance values.
    x, y : int
        Coordinates of given kth endpoint, starting coordinates.
    limit : float
        Defines the threshold between short and long segments.
        
    Returns:
    --------
    x1, y1, x2, y2, x3, y3 : float
        Coordinates of segment points and their normal vectors
    Dtemp, Etemp: ndarray
        Not sure if these are actually used, and Etemp is a placeholder -SamK
    """

    # Find the indices of each segment
    Dmasktemp = np.zeros_like(D, dtype=bool)
    Dmasktemp[D < distanceToPt] = True  # Remove creek segments - make everthing farther away than point False
    # Calculate geodesic distance using graph approach for better accuracy
    Dtemp = calculate_chessboard_distance(Dmasktemp, y, x)
    maxDtemp = np.nanmax(Dtemp)
    
    if maxDtemp <= limit: # Short segments handling
        xm, ym = x, y
        
        # Find the other endpoint
        indices = np.where(Dtemp == maxDtemp)  # Get the indices where Dtemp equals maxDtemp
        # Get the last occurrence
        last_row = indices[0][-1]  # Last row index
        last_col = indices[1][-1]  # Last column index
        endpt = [last_row, last_col]
        
        y1, x1 = endpt  # unpack coordinate pair

        # translation matrix along the x-axis by xm and along the y-axis by ym
        T1 = np.array([
            [1, 0, 0, xm],
            [0, 1, 0, ym],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Rotation matrices (+90 and -90 degrees) along z-axis
        R90 = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        Rm90 = Rotation.from_euler('z', -90, degrees=True).as_matrix()
        T_rot1 = np.eye(4)
        T_rot1[:3, :3] = R90
        T_rot2 = np.eye(4)
        T_rot2[:3, :3] = Rm90
        
        # Apply transformations
        point = np.array([x1, y1, 0, 1]) # homogeneous coordinates of the point

        Vnew1 = T1 @ T_rot1 @ np.linalg.inv(T1) @ point # perform transformation
        x1new, y1new = Vnew1[0], Vnew1[1]

        Vnew2 = T1 @ T_rot2 @ np.linalg.inv(T1) @ point # perform transformation
        x2new, y2new = Vnew2[0], Vnew2[1]
        
    else: # Long segments handling
        Dtemp_nonan = Dtemp[~np.isnan(Dtemp)]

        midpt = round(np.mean(Dtemp_nonan)) # 1. Find value equal to rounded mean
        linear_indices = np.where(Dtemp.ravel() == midpt)[0] # 2. Get linear indices where Dtemp equals this value
        mid_idx = linear_indices[len(linear_indices)//2]  # 3. Take middle element of these indices
        row_midpt, col_midpt = np.unravel_index(mid_idx, Dtemp.shape) # 4. Convert back to 2D coordinates
        ym, xm = row_midpt, col_midpt # instead of ind2sub
        
        # Find the two points at a distance of 0.2*length of midpoint of long segment
        dist = round(0.2 * maxDtemp)
        mid_distances = calculate_chessboard_distance(Dmasktemp, ym, xm) # find distances from midpoint coordinates
        ALLIDX = np.where(mid_distances == dist)
        y1, x1 = ALLIDX[0][0], ALLIDX[1][0]  # first point
        y2, x2 = ALLIDX[0][1], ALLIDX[1][1]  # second point
                
        # Create transformation matrices
        T1 = np.array([
            [1, 0, 0, xm],
            [0, 1, 0, ym],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Rotation matrices (+90 and -90 degrees) along z-axis
        R90 = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        
        T_rot = np.eye(4)
        T_rot[:3, :3] = R90
        
        # Create points as column vectors
        point_1 = np.array([x1, y1, 0, 1], dtype=float).reshape((4, 1))
        point_2 = np.array([x2, y2, 0, 1], dtype=float).reshape((4, 1))

        # Apply transformations
        Vnew1 = T1 @ T_rot @ np.linalg.inv(T1) @ point_1
        Vnew2 = T1 @ T_rot @ np.linalg.inv(T1) @ point_2

        # Extract transformed coordinates
        x1new = Vnew1[0, 0]
        y1new = Vnew1[1, 0]
        x2new = Vnew2[0, 0]
        y2new = Vnew2[1, 0]
    
    # Final coordinate assignments
    x2, y2 = x1new, y1new
    x3, y3 = x2new, y2new
    x1, y1 = xm, ym
    
    Etemp = np.zeros_like(D, dtype=bool)  # placeholder for compatibility - not sure if this or Dtemp are needed -SamK
    
    return x1, y1, x2, y2, x3, y3, Dtemp, Etemp

def normal_coord_test(D, distanceToPt, x, y, limit, skeleton_chopped, B_loc):
    """
    Test version of normal_coord that handles branch points differently
    
    Parameters:
    -----------
    D : ndarray
        Distance matrix or array from raster.
    distanceToPt : float
        Distance to target point.
    x, y : int
        Starting coordinates.
    limit : float
        Threshold for short/long segment classification.
    skeleton_chopped : ndarray
        Skeleton binary image, currently pruned.
    B_loc : array
        Branch point locations, should be given as return of np.argwhere, 
        with a nested array of coordinate pairs.
        
    Returns:
    --------
    x1, y1, x2, y2, x3, y3 : float
        Coordinates of segment points and their normal vectors.
    BWrem : ndarray
        Modified skeleton mask.
    """
    print('normal_coord_test: ')
    print('   y, x = ', (y, x))
    # Create branch point mask
    Bmask = np.zeros_like(D, dtype=bool)
    Bmask[B_loc] = True 
    # Bmask.ravel()[B_loc] = True
    yB, xB = np.where(Bmask)

    # Find the indices of each segment
    Dmasktemp = np.zeros_like(D, dtype=bool)
    Dmasktemp[D < distanceToPt] = True  # Remove creek segments - make everthing farther away than point False
    # Calculate geodesic distance using graph approach for better accuracy
    # Dtemp = calculate_chessboard_distance(Dmasktemp, y, x)
    Dtemp = calculate_chessboard_distance(Dmasktemp, y, x)
    # maxDtemp = np.nanmax(Dtemp)

    # ADDITIONAL STEP: remove the longer segments of Dmasktemp not connected to a point in B_loc
    Dmasktemp[B_loc] = True
    Dmasktempconn = bwconncomp(Dmasktemp) # could also do labeled_array, num_features = ndimage.label(Dmasktemp)

    XYmask = np.zeros_like(D, dtype=bool)
    XYmask[y, x] = True
    XYind = np.where(XYmask)

    idxnum = Dmasktempconn['n_objects'] # Get number of objects
    # Find which components contain pixels from XYind
    idx = [i for i, component in enumerate(Dmasktempconn['pixel_idx']) 
        if any(np.in1d(component, XYind))]
    # Remove all other elements
    Dmasktemp = np.zeros_like(D, dtype=bool)
    # Set pixels using row,col coordinates
    if isinstance(idx, list): # Use first element if idx is a list:
        idx = idx[0]
    # Handle pixel_idx which contains coordinate tuples
    pixel_coords = Dmasktempconn['pixel_idx'][idx]
    # print('Dmasktempconn[\'pixel_idx\'][idx] = ', Dmasktempconn['pixel_idx'][idx])
    row_indices = pixel_coords[0]  # first array has row indices
    col_indices = pixel_coords[1]  # second array has column indices

    # print('Dmasktemp[row_indices, col_indices] = True, Dmasktemp = ', Dmasktemp)
    Dmasktemp[row_indices, col_indices] = True
    # Dmasktemp[pixel_coords[1], pixel_coords[0]] = True

    ptmask1 = np.zeros_like(Dmasktemp, dtype=bool)
    # ptmask1[B_loc] = True
    # ptmask1[B_loc[:,0], B_loc[:,1]] = True
    ptmask1[B_loc[0], B_loc[1]] = True
    print('   B_loc = ', B_loc)
    ptmask = ptmask1 & Dmasktemp
    print("   Dmasktemp shape and sum:", Dmasktemp.shape, np.sum(Dmasktemp))
    print("   ptmask1 shape and sum:", ptmask1.shape, np.sum(ptmask1))
    print("   Final ptmask shape and sum:", ptmask.shape, np.sum(ptmask))
    yptend, xptend = np.where(ptmask)
    ptmaskind = np.where(ptmask)
    print('   D1:')
    D1 = calculate_geodesic_distance(skeleton_chopped, y, x)
    print('   D2:')
    D2 = calculate_geodesic_distance(skeleton_chopped, yptend, xptend)
    Dnew = D1 + D2
    Dnew = np.round(Dnew * 8) / 8
    Dnew[np.isnan(Dnew)] = np.inf
    paths = imregionalmin(Dnew)
    solution_path = morphology.thin(paths)
    
    Dmasktemp = solution_path.copy()
    Dmasktemp[ptmaskind] = True
    BWrem = Dmasktemp.copy()
    BWrem[ptmaskind] = False
    
    maxDtemp = np.nansum(Dmasktemp)
    
    if maxDtemp <= limit: # Short segments handling
        # Handle short segments
        xm, ym = x, y
        # Find the other end point of the short segment
        x1, y1 = xptend[0], yptend[0]
        
        # translation matrix along the x-axis by xm and along the y-axis by ym
        T1 = np.array([
            [1, 0, 0, xm],
            [0, 1, 0, ym],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Rotation matrices (+90 and -90 degrees) along z-axis
        R90 = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        Rm90 = Rotation.from_euler('z', -90, degrees=True).as_matrix()
        T_rot1 = np.eye(4)
        T_rot1[:3, :3] = R90
        T_rot2 = np.eye(4)
        T_rot2[:3, :3] = Rm90
        
        # Apply transformations
        point = np.array([x1, y1, 0, 1]) # homogeneous coordinates of the point

        Vnew1 = T1 @ T_rot1 @ np.linalg.inv(T1) @ point # perform transformation
        x1new, y1new = Vnew1[0], Vnew1[1]

        Vnew2 = T1 @ T_rot2 @ np.linalg.inv(T1) @ point # perform transformation
        x2new, y2new = Vnew2[0], Vnew2[1]
        
    else: # Long segments handling
        # Find the middle point of the segment for long segments
        Dtemp = calculate_geodesic_distance(Dmasktemp, y, x)
        # # Dtemp_nonan = Dtemp[~np.isnan(Dtemp)]
        # mid_value = round(np.nanmean(Dtemp))
        # row_midpts, col_midpts = np.where(Dtemp == mid_value)
        # midpts = Dtemp[row_midpts, col_midpts]
        # midpt = midpts[len(midpts)//2] 
        # # midpt = np.median(midpts.flatten())
        midpt = round(np.nanmean(Dtemp)) # 1. Find value equal to rounded mean
        linear_indices = np.where(Dtemp.ravel() == midpt)[0] # 2. Get linear indices where Dtemp equals this value
        mid_idx = linear_indices[len(linear_indices)//2]  # 3. Take middle element of these indices
        row_midpt, col_midpt = np.unravel_index(mid_idx, Dtemp.shape) # 4. Convert back to 2D coordinates
        
        MIDPT = np.zeros_like(Dtemp, dtype=bool)
        # row_mid, col_mid = np.where(Dtemp == midpt)  # get coordinates where value appears
        # Set that location to True
        # MIDPT[row_mid, col_mid] = True
        MIDPT[row_midpt, col_midpt] = True
        ym, xm = np.where(MIDPT) # instead of find(), ind2sub() is commented out by AnnaM in MATLAB -SamK
        
        # Find the two points at a distance of 0.2*length of midpoint of long segment
        dist = round(maxDtemp/4)
        mid_distances = calculate_chessboard_distance(Dmasktemp, ym, xm) # find distances from midpoint coordinates
        ALLIDX = np.where(mid_distances == dist)
        if ALLIDX[0].size == 0:  # if ALLIDX is from np.where()
            x1, y1 = x[0], y[0]
            x2, y2 = xptend[0], yptend[0]
        else:
            XY1 = np.zeros_like(Dtemp, dtype=bool)
            XY1[ALLIDX[0][0], ALLIDX[1][0]] = True
            y1, x1 = np.where(XY1)
            XY2 = np.zeros_like(Dtemp, dtype=bool)
            XY2[ALLIDX[0][1], ALLIDX[1][1]] = True
            y2, x2 = np.where(XY2)
            x1, y1 = x1[0], y1[0]
            x2, y2 = x2[0], y2[0]

        # translation matrix along the x-axis by xm and along the y-axis by ym
        T1 = np.array([
            [1, 0, 0, xm],
            [0, 1, 0, ym],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Rotation matrices (+90 and -90 degrees) along z-axis
        R90 = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        
        T_rot = np.eye(4)
        T_rot[:3, :3] = R90

        # Create points as column vectors
        point_1 = np.array([x1, y1, 0, 1], dtype=float).reshape((4, 1))
        point_2 = np.array([x2, y2, 0, 1], dtype=float).reshape((4, 1))

        # Apply transformations
        Vnew1 = T1 @ T_rot @ np.linalg.inv(T1) @ point_1
        Vnew2 = T1 @ T_rot @ np.linalg.inv(T1) @ point_2

        # Extract transformed coordinates
        x1new = Vnew1[0, 0]
        y1new = Vnew1[1, 0]
        x2new = Vnew2[0, 0]
        y2new = Vnew2[1, 0]

    # Final coordinate assignments
    x2, y2 = x1new, y1new
    x3, y3 = x2new, y2new
    x1, y1 = xm, ym
    
    return x1, y1, x2, y2, x3, y3, BWrem

def bwconncomp(binary_image):
    """
    Mimics MATLAB's bwconncomp.
    Identifies connected components in a binary image.

    Parameters:
        binary_image (np.ndarray): A binary 2D numpy array.

    Returns:
        dict: A dictionary with keys:
            - 'connectivity': 8 for 2D connectivity
            - 'image_size': Size of the image
            - 'n_objects': Number of connected components
            - 'pixel_idx': List of tuples (arrays) with indices for each component
    """
    # Define 8-connectivity structure
    structure = np.ones((3, 3))
    
    # Label connected components
    labeled, n_objects = ndimage.label(binary_image, structure=structure)
    
    # Extract pixel indices for each connected component
    pixel_idx = [np.where(labeled == i) for i in range(1, n_objects + 1)]
    
    return {
        'connectivity': 8,
        'image_size': binary_image.shape,
        'n_objects': n_objects,
        'pixel_idx': pixel_idx
    }

def skeleton_to_graph(skeleton):
    """Convert skeleton image to networkx graph"""
    # global nx  # Add this line to explicitly use the global nx import
    import networkx as nx # type: ignore

    # Get coordinates of skeleton pixels
    points = np.transpose(np.nonzero(skeleton))
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for i, (y, x) in enumerate(points):
        G.add_node(i, pos=(y, x))
    
    # Add edges between 8-connected neighbors
    for i, (y1, x1) in enumerate(points):
        for j, (y2, x2) in enumerate(points[i+1:], i+1):
            if abs(y1-y2) <= 1 and abs(x1-x2) <= 1:
                dist = np.sqrt((y1-y2)**2 + (x1-x2)**2)
                G.add_edge(i, j, weight=dist)
    
    return G, points

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

def imregionalmin(image):
    """
    Find regional minima in an image (equivalent to MATLAB's imregionalmin)
    
    Parameters:
    image : ndarray
        Input image
    
    Returns:
    binary mask : ndarray
        Boolean array where True indicates local minima
    """
    # Get minimum filter using 3x3 square structuring element
    min_filtered = ndimage.minimum_filter(image, size=(3,3))
    
    # Regional minimum is where pixel equals filtered value
    # and is not equal to neighbors (to exclude plateaus)
    return (image == min_filtered) & (image != np.inf)

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

def bwmorph_diag(image):
    """Eliminates 8-connectivity of the background by adding diagonal fills
    Equivalent to MATLAB's bwmorph(image,'diag')
    
    Example:
    0 1 0     0 1 0
    1 0 0 ->  1 1 0
    0 0 0     0 0 0
    """
    # Create kernels to detect background diagonal patterns
    diag1 = np.array([[0, 1, 0],
                      [1, 0, 0],
                      [0, 0, 0]], dtype=bool)
    
    diag2 = np.array([[0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]], dtype=bool)
    
    diag3 = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0]], dtype=bool)
    
    diag4 = np.array([[0, 0, 0],
                      [0, 0, 1],
                      [0, 1, 0]], dtype=bool)
    
    # Find all diagonal patterns
    hits = []
    for kernel in [diag1, diag2, diag3, diag4]:
        hits.append(ndimage.binary_hit_or_miss(image, kernel))
    
    # Fill in diagonals by adding pixels
    result = image.copy()
    for hit in hits:
        result = result | hit
        
    return result

def bwmorph_thicken(image, iterations):
    """Thicken the image by n iterations"""
    result = image.copy()
    for _ in range(iterations):
        result = dilation(result, disk(1))
    return result

# def bwmorph_thicken_matlab(image, n_iter=np.inf):
#     """
#     Implement MATLAB-style morphological thickening.
#     Uses hit-or-miss transformations to add border pixels while preserving topology.
    
#     Parameters:
#     -----------
#     image : ndarray
#         Binary input image
#     n_iter : int or np.inf
#         Number of iterations to perform
        
#     Returns:
#     --------
#     ndarray : Thickened binary image
#     """
#     # Convert to boolean if not already
#     img = image.astype(bool)
    
#     # Define the hit-or-miss kernels for thickening
#     # These are the standard MATLAB kernels for 8-connectivity
#     hm_kernels = [
#         np.array([[0, 0, 0],
#                  [0, 0, 1],
#                  [0, 1, 0]], dtype=bool),
#         np.array([[0, 0, 0],
#                  [1, 0, 0],
#                  [0, 1, 0]], dtype=bool),
#         np.array([[0, 1, 0],
#                  [0, 0, 1],
#                  [0, 0, 0]], dtype=bool),
#         np.array([[0, 1, 0],
#                  [1, 0, 0],
#                  [0, 0, 0]], dtype=bool)
#     ]
    
#     # Add rotated versions of the kernels
#     all_kernels = []
#     for kernel in hm_kernels:
#         all_kernels.extend([
#             kernel,
#             np.rot90(kernel, 1),
#             np.rot90(kernel, 2),
#             np.rot90(kernel, 3)
#         ])
    
#     # Initialize iteration counter and previous result
#     n = 0
#     prev = None
    
#     while n < n_iter:
#         result = img.copy()
        
#         # Apply each hit-or-miss kernel
#         for kernel in all_kernels:
#             # Create the inverse kernel for background matching
#             k_inv = np.logical_not(kernel)
            
#             # Perform hit-or-miss transform
#             hm = ndimage.binary_hit_or_miss(img, structure1=kernel, structure2=k_inv)
            
#             # Add matched pixels to result
#             result = np.logical_or(result, hm)
        
#         # Check if image has changed
#         if prev is not None and np.array_equal(prev, result):
#             break
            
#         img = result
#         prev = result
#         n += 1
        
#     return img

# def bwmorph_clean(image):
#     """Remove isolated pixels"""
#     return ndimage.binary_opening(image)

def bwmorph_clean(image):
    """
    Remove isolated pixels (1's surrounded by 0's)
    """
    # Create 3x3 structuring element
    struct = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
    # Count neighbors
    neighbors = ndimage.convolve(image.astype(int), struct, mode='constant')
    # Remove pixels with no neighbors
    return image & (neighbors > 1)

# def check_imports():
#     """Check if all required packages are imported correctly"""
#     try:
#         import networkx as nx
#         print(f"NetworkX version: {nx.__version__}")
        
#         import numpy as np
#         print(f"NumPy version: {np.__version__}")
        
#         import skimage
#         print(f"scikit-image version: {skimage.__version__}")
        
#         import scipy
#         print(f"SciPy version: {scipy.__version__}")
        
#         print("All imports successful!")
#         return True
        
#     except ImportError as e:
#         print(f"Import error: {str(e)}")
#         return False

# def test_imports():
#     """Test if all required functionality is available"""
#     try:
#         # Test networkx
#         import networkx as nx
#         G = nx.Graph()
#         G.add_node(1)
        
#         # Test numpy
#         import numpy as np
#         test_array = np.zeros((10, 10))
        
#         # Test skimage morphology
#         from skimage import morphology
#         test_skel = morphology.skeletonize(test_array)
        
#         # Test scipy
#         from scipy import ndimage
#         test_dist = ndimage.distance_transform_edt(test_array)
        
#         print("All functionality tests passed!")
#         return True
        
#     except Exception as e:
#         print(f"Functionality test failed: {str(e)}")
#         return False