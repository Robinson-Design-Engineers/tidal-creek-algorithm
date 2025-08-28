import numpy as np # type: ignore
from skimage import morphology, measure, feature, draw # type: ignore
import matplotlib.pyplot as plt # type: ignore
from scipy import ndimage # type: ignore
from scipy.ndimage import distance_transform_edt # type: ignore
from skimage.morphology import dilation, disk # type: ignore
import networkx as nx # type: ignore
from scipy.spatial.transform import Rotation # type: ignore
from skimage.graph import route_through_array
from scipy.ndimage import binary_dilation

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

def figure_creek_skeleton(skeleton,Z):
    # Thicken the skeleton for display
    skeletonpic = morphology.dilation(skeleton, morphology.square(3))
    skeletonpic2 = np.logical_not(skeletonpic)

    B, E, _ = analyze_skeleton(skeleton,Z)

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

def figure_creek_skeleton_diagnostic(skeleton, x_poi, y_poi, Z):
    # Thicken the skeleton for display
    skeletonpic = morphology.dilation(skeleton, morphology.square(3))
    skeletonpic2 = np.logical_not(skeletonpic)

    B, E, _ = analyze_skeleton(skeleton, Z)
    
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


def process_creek_ordering(ordermax, Z, skeleton, outletdetection, nbbreaches): # for debugging
    return STRAHLER, STRAIGHTDIST, IDXSEG, IDXBRANCH, idxbreach, xbreach, ybreach, skeleton_breached, creekorder, creekordersing, PTS, ID



def process_creek_ordering_diagnostic(ordermax, Z, skeleton, outletdetection, nbbreaches): # for debugging
    
    # Automatically detect one or several outlets (deepest end points in the skeleton)
    if outletdetection == 1:
        idxbreach, xbreach, ybreach, skeleton_breached = process_outlet_detection(Z, skeleton, nbbreaches)
    else:
        idxbreach, xbreach, ybreach, skeleton_breached = [], [], [], []

    # resave skeleton as breached skeleton to ensure outlet point is connected to skeleton
    skeleton = skeleton_breached.copy() # added to debug -SamK

    # Initialize variables
    B, E, PTS = analyze_skeleton(skeleton, Z)
    STRAHLER = np.zeros_like(E, dtype=float)
    STRAIGHTDIST = np.zeros_like(E, dtype=float) # contains straight length
    IDXSEG = np.zeros((skeleton.shape[0], 6)) # contains coordinates of the normal vector (end and mid points)
    IDXBRANCH = np.zeros_like(skeleton, dtype=int) # contains indices of all branch points
    # IDXBRANCH = np.zeros_like(skeleton, dtype=[('row', int), ('col', int)])
    creekorder = np.zeros_like(skeleton, dtype=int)
    creekordersing = np.zeros_like(skeleton, dtype=int)
    skeleton_chopped = skeleton.copy()  # we keep skeleton intact and remove segments from skeleton_chopped
    ID = []
    print('initial ID = ', ID)
    i = 1
    limit = 3

    # Get the Strahler order and segment lengths in a loop
    i_check = 1
    max_iterations = 1000  # Failsafe to prevent infinite loop
    skelprev = np.zeros_like(skeleton_chopped, dtype=bool)
    
    print('Starting while loop 1 (while there are any endpoints left):')
    # while loop 1 out of 4 - while there are any endpoints left
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
        # print('col = ', col)
        coords = np.argwhere(E)
        y, x = coords[:, 0], coords[:, 1]
        
        for k in range(len(x)):
            print('k = ', k, ' of ', len(x))
            # print('E location y, x = ', (y[k], x[k]))
            if k < 1:
                figure_creek_skeleton_diagnostic(skeleton_chopped, x[k], y[k], Z)
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

            # print('distanceToEndPt = ', distanceToEndPt)
            # print('distanceToBranchPt = ', distanceToBranchPt)
            
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
            elif np.isfinite(distanceToEndPt) and np.isfinite(distanceToBranchPt) and distanceToEndPt < distanceToBranchPt:
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
                skeletoneucl[E_loc[k,0], E_loc[k,1]] = 1 # make point of interest's value = 1 or True
                euclD = ndimage.distance_transform_edt(skeletoneucl == 0) # for each pixel, euclidean dist to nearest non-zero value 
                disttopt = euclD[ptlocy, ptlocx] # straight distance to point of interest, measured from ptloc, the nearest endpoint
                STRAIGHTDIST[i,k] = disttopt
                # Save index of branch point / nearest point for junction angle assignment
                if np.isscalar(ptlocx) and np.isscalar(ptlocy):
                    # If they're already scalars
                    ptloc_linear = np.ravel_multi_index((ptlocy, ptlocx), DD.shape)
                else:
                    # If they're arrays
                    ptloc_linear = np.ravel_multi_index((ptlocy[0], ptlocx[0]), DD.shape)
                IDXBRANCH[i,k] = ptloc_linear

                # Create normal vectors to each segment and store their coordinates
                # should it be y[k], x[k] or x[k], y[k]? Currently following MATLAB code, where y[k], x[k] are set as the x,y inptus in the function -SamK
                x1, y1, x2, y2, x3, y3, _, _ = normal_coord(DD, distanceToEndPt, y[k], x[k], limit)
                
                if col < IDXSEG.shape[1]:
                    IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
                else:
                    new_values = [x1, y1, x2, y2, x3, y3]
                    IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                    IDXSEG[i, -6:] = new_values

                # print('STRAHLER = ', STRAHLER)
                # print('STRAIGHTDIST = ', STRAIGHTDIST)
                # print('IDXSEG = ', IDXSEG)
                # print('IDXBRANCH = ', IDXBRANCH)

                col += 6
                # print('col = ', col)
                
                # Prepare to remove the segments order i
                Dmask[DD < distanceToEndPt] = True # Dmask starts as zeros like skeleton_chopped - this makes the current segment True or 1

            # Find isolated segment (branch point doesn't exist)
            # seems exact same as if distanceToEndPt < distanceToBranchPt, except 
            # uses normal_coord function on x[k], y[k] rather than y[k], x[k] as above -SamK
            elif np.isinf(distanceToBranchPt) and np.isfinite(distanceToEndPt):
                # Store segment real length into strahler table
                STRAHLER[i,k] = distanceToEndPt # k is index of this segment within the given strahler order

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
                skeletoneucl[E_loc[k,0], E_loc[k,1]] = 1 # make the point of interest True or 1
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

                # print('STRAHLER = ', STRAHLER)
                # print('STRAIGHTDIST = ', STRAIGHTDIST)
                # print('IDXSEG = ', IDXSEG)
                # print('IDXBRANCH = ', IDXBRANCH)

                col += 6
                # print('col = ', col)
                
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

                # print('STRAHLER = ', STRAHLER)
                # print('STRAIGHTDIST = ', STRAIGHTDIST)
                # print('IDXSEG = ', IDXSEG)
                # print('IDXBRANCH = ', IDXBRANCH)

                col += 6
                # print('col = ', col)
                
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

            # print('')

        # Prepare the creek order plot
        creekordermask = bwmorph_diag(Dmask) 
        # ^takes away 8-connectivity, makes the creek order skeleton (Dmask) into 4-connectivity - every pixel connects on square connections
        # Assign creek orders:
        creekordersing[creekordermask != 0] = creekordermask[creekordermask != 0] * i
        # Thicken the mask i times:
        # creekordermask = bwmorph_thicken(creekordermask, i)
        creekordermask = dilation(creekordermask, disk(1)) # thickens creekordermask for visibility of "creekorder" in plots of skeleton
        # Assign creek orders to thickened mask
        creekorder[creekordermask != 0] = creekordermask[creekordermask != 0] * i
        # Clear the mask
        del creekordermask

        # Remove order i segments
        # skelD = skeleton_chopped - Dmask # subtract matrices
        # skelD = np.logical_xor(skeleton_chopped, Dmask) # or use skelD = skeleton_chopped ^ Dmask - take away current segment from skeleton
        skelD = skeleton_chopped & ~Dmask  # Remove Dmask pixels from skeleton
        skeleton_chopped = skelD.astype(bool) # convert to boolean/logical, could also do skelchopped = skelD > 0 -SamK

        # Redo skeleton
        skeleton_chopped = morphology.skeletonize(skeleton_chopped)
        np.savetxt(fname='/Users/sam/Desktop/skeleton_chopped.csv', X=skeleton_chopped , delimiter=',')
        ID = np.append(ID, i) # update order tracking
        print('redo skeleton ID = ', ID)

        # Find the end and branch points of the order i+1 network configuration
        i += 1 # Store order number and move on to the next creek order until skeleton contains only 0
        col = 0
        B, E, PTS = analyze_skeleton(skeleton_chopped, Z)
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
    B, E, PTS = analyze_skeleton(skeleton_chopped, Z)
    ESIZE = np.any(E) # doesn't seem to be used later -SamK
    test = 1

    i_check = 1
    print('')

    print('Starting while loop 2 (outlets):')
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
            print('k2 = ', k2, ' of ', len(x))
            print('E location y[k2], x[k2] = ', (y[k2], x[k2]))
            if np.abs(len(x) - k2) < 3 or k2 < 1:
                figure_creek_skeleton_diagnostic(skeleton_chopped, x[k2], y[k2], Z)

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

            print('distanceToEndPt = ', distanceToEndPt)
            print('distanceToBranchPt = ', distanceToBranchPt)
            
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
            elif np.isfinite(distanceToEndPt) and np.isfinite(distanceToBranchPt) and distanceToEndPt < distanceToBranchPt:
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

                # print('STRAHLER = ', STRAHLER)
                # print('STRAIGHTDIST = ', STRAIGHTDIST)
                # print('IDXSEG = ', IDXSEG)
                # print('IDXBRANCH = ', IDXBRANCH)

                col += 6
                # print('col = ', col)

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

                # print('STRAHLER = ', STRAHLER)
                # print('STRAIGHTDIST = ', STRAIGHTDIST)
                # print('IDXSEG = ', IDXSEG)
                # print('IDXBRANCH = ', IDXBRANCH)

                col += 6
                # print('col = ', col)

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

                # print('STRAHLER = ', STRAHLER)
                # print('STRAIGHTDIST = ', STRAIGHTDIST)
                # print('IDXSEG = ', IDXSEG)
                # print('IDXBRANCH = ', IDXBRANCH)

                col += 6
                # print('col = ', col)
                
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
            print('')

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
        # skelD = np.logical_xor(skeleton_chopped, Dmask) # or use skelD = skeleton_chopped ^ Dmask
        skelD = skeleton_chopped & ~Dmask  # Remove Dmask pixels from skeleton
        skeleton_chopped = skelD.astype(bool) # convert to boolean/logical, could also do skelchopped = skelD > 0 -SamK

        # Redo skeleton
        skeleton_chopped = morphology.skeletonize(skeleton_chopped)
        # ID = np.append(ID, i) # update order tracking

        # Find the end and branch points of the order i+1 network configuration
        # i += 1 # Store order number and move on to the next creek order until skeleton contains only 0
        # col = 0
        B, E, PTS = analyze_skeleton(skeleton_chopped, Z)

        i_check += 1
        if i_check >= max_iterations:
            print("Max iterations reached, breaking out of the loop.")
            break
    
        print('')

    # Outlet Segment assignation has occured until no more end points are detected. 
    # We do a THIRD one that detects all branch points and gets interconnected sections.
    skelprev = np.zeros_like(skeleton, dtype=bool)
    B, E, PTS = analyze_skeleton(skeleton_chopped, Z)
    B[E != 0] = True # could also do: B[np.where(E)] = True -SamK
    # # Update breach points
    # E.ravel()[idxbreach] = False
    # B.ravel()[idxbreach] = True

    # Figure - not translated as of now, commented out in AnnaM's MATLAB code -SamK

    i_check = 1

    print('Starting while loop 3 (branch points / interconnected segments):')
    # while loop 3 out of 4 - branch points / interconnected segments
    while np.any(B) and i_check<= max_iterations:
        print(f"while 3/4, Iteration {i_check}: {np.sum(B)} endpoints remaining")
        print('order i = ', i)
        # Select 1 node
        k2 = 1

        print(f"Total skeleton pixels: {np.sum(skeleton_chopped)}")
        print(f"Number of branch points found: {np.sum(B)}")
        print(f"Number of PTS points: {np.sum(PTS)}")

        # Find the locations of all branch and end points in the order i configuration
        B_loc = np.argwhere(B) # np.argwhere returns a 2D array of coordinates
        print(f"B_loc shape: {B_loc.shape}")
        print(f"First few B_loc coordinates: {B_loc[:5] if len(B_loc) > 0 else 'None'}")
        # E_loc = np.argwhere(E)
        Dmask = np.zeros_like(skeleton_chopped, dtype=bool)
        # col = 0
        # print('col = ', col)
        coords = np.argwhere(B) # different than being based on E in first two while loops -SamK
        y, x = coords[:, 0], coords[:, 1]

        # "Plot the order i configuration" and "Check for interconnections" commented out in AnnaM's MATLAB code -SamK
        
        for k2 in range(len(x)): # commented out in AnnaM's MATLAB code, let's try using -SamK
            k = k+1

            # print figure of current study point in current skeleton
            print('len(x) = ', len(x))
            print('k = ', k)
            print('k2 = ', k2, ' of ', len(x))
            print('E location y[k2], x[k2] = ', (y[k2], x[k2]))
            if np.abs(len(x) - k2) < 3 or k2 < 1:
                figure_creek_skeleton_diagnostic(skeleton_chopped, x[k2], y[k2], Z)

            # remove the selected kth end point from B_loc
            B_loctemp = np.delete(B_loc, k2, axis=0) # assume B_loc is 2D array
            E_loctemp = E_loc # there should be no end points left -SamK

            # not sure why this is here, was in MATLAB, will comment out for now -SamK
            # y2 = x
            # x = y
            # y = y2

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

            print('distanceToEndPt = ', distanceToEndPt)
            print('distanceToBranchPt = ', distanceToBranchPt)
                
            # Three possibilities here: the end point we have detected can be:
            # 1) an isolated point (no creek segment to detect)
            # 2) an isolated creek segment delimited by two end points
            # 3) a creek segment delimited by a branch point and an end point

            # Remove isolated points (no creek segment to detect)
            if np.isinf(distanceToEndPt) and np.isinf(distanceToBranchPt):
                # Remove the point from skeleton_chopped
                skeleton_chopped[y[k2], x[k2]] = False
                # Remove point from endpoint array B
                B[y[k2], x[k2]] = False
                print(f"Removed isolated point at y={y[k2]}, x={x[k2]}")
                
                # Continue to next point in the original coords list
                continue
            
            # Find isolated segment (end point closer than branch point) - I guess like a vernal pool? -SamK
            elif np.isfinite(distanceToEndPt) and np.isfinite(distanceToBranchPt) and distanceToEndPt < distanceToBranchPt:
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
                # print('while loop 3 if part 1: normal_coord_test inputs:')
                # print('   ptlocy, ptlocx = y[k2], x[k2] = ', (y[k2], x[k2]))
                # print('   B_loc = B_loctemp = ', B_loctemp)
                try:
                    x1, y1, x2, y2, x3, y3, BWrem, BWrem_withEndPts, BWrem_withEndPtKnobs = normal_coord_test_diagnostic(DD, distanceToBranchPt, x[k2], y[k2], limit, skeleton_chopped, B_loctemp, (len(x)-k2))
                    
                    if col < IDXSEG.shape[1]:
                        IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
                    else:
                        new_values = [x1, y1, x2, y2, x3, y3]
                        IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                        IDXSEG[i, -6:] = new_values
                    # print('STRAHLER = ', STRAHLER)
                    # print('STRAIGHTDIST = ', STRAIGHTDIST)
                    # print('IDXSEG = ', IDXSEG)
                    # print('IDXBRANCH = ', IDXBRANCH)
                
                    col += 6
                    # print('col = ', col)

                    # Prepare to remove the segments order i
                    # Dmask[DD < distanceToEndPt] = True
                    # Dmask=BWrem # BWrem is the segment without branch or endpoints # this keeps 
                    # if len(x) == 2 and k2 == 0: # processing the last of two remaining point
                    #     print('len(x) == 2 and k2 == 0: breaking for loop now')
                    #     # include both of the last branch/endpoints to be removed, as well as adjacent pts to end points to ensure breaks in any possible loops
                    #     Dmask |= BWrem_withEndPtKnobs  # Add to existing Dmask
                    #     break # Exit the for loop
                    # else:
                    #     Dmask |= BWrem  # Add to existing Dmask
                    Dmask |= BWrem
                    
                except ValueError as e:
                    print(f"Removing problematic branch point at ({y[k2]}, {x[k2]}) from B - {str(e)}")
                    B[y[k2], x[k2]] = False  # Remove from branch point mask
                    continue  # Skip to next k2 iteration

                

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
                # print('while loop 3 if part 2: normal_coord_test inputs:')
                # print('   ptlocy, ptlocx = y[k2], x[k2] = ', (y[k2], x[k2]))
                # print('   B_loc = B_loctemp = ', B_loctemp)
                try:
                    x1, y1, x2, y2, x3, y3, BWrem, BWrem_withEndPts, BWrem_withEndPtKnobs = normal_coord_test_diagnostic(DD, distanceToBranchPt, x[k2], y[k2], limit, skeleton_chopped, B_loctemp, (len(x)-k2))
                    
                    if col < IDXSEG.shape[1]:
                        IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
                    else:
                        new_values = [x1, y1, x2, y2, x3, y3]
                        IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                        IDXSEG[i, -6:] = new_values
                    # print('STRAHLER = ', STRAHLER)
                    # print('STRAIGHTDIST = ', STRAIGHTDIST)
                    # print('IDXSEG = ', IDXSEG)
                    # print('IDXBRANCH = ', IDXBRANCH)
                
                    col += 6
                    # print('col = ', col)

                    # Prepare to remove the segments order i
                    # Dmask[DD < distanceToEndPt] = True
                    # Dmask=BWrem # BWrem is the segment without branch or endpoints # this keeps 
                    # if len(x) == 2 and k2 == 0: # processing the last of two remaining point
                    #     print('len(x) == 2 and k2 == 0: breaking for loop now')
                    #     # include both of the last branch/endpoints to be removed, as well as adjacent pts to end points to ensure breaks in any possible loops
                    #     Dmask |= BWrem_withEndPtKnobs  # Add to existing Dmask
                    #     break # Exit the for loop
                    # else:
                    #     Dmask |= BWrem  # Add to existing Dmask
                    Dmask |= BWrem
                    
                except ValueError as e:
                    print(f"Removing problematic branch point at ({y[k2]}, {x[k2]}) from B - {str(e)}")
                    B[y[k2], x[k2]] = False  # Remove from branch point mask
                    continue  # Skip to next k2 iteration


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
                # print('while loop 3 if part 3: normal_coord_test inputs:')
                # print('   ptlocy, ptlocx = y[k2], x[k2] = ', (y[k2], x[k2]))
                # print('   B_loc = B_loctemp = ', B_loctemp)

                try:
                    x1, y1, x2, y2, x3, y3, BWrem, BWrem_withEndPts, BWrem_withEndPtKnobs = normal_coord_test_diagnostic(DD, distanceToBranchPt, x[k2], y[k2], limit, skeleton_chopped, B_loctemp, (len(x)-k2))
                    
                    if col < IDXSEG.shape[1]:
                        IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
                    else:
                        new_values = [x1, y1, x2, y2, x3, y3]
                        IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                        IDXSEG[i, -6:] = new_values
                    # print('STRAHLER = ', STRAHLER)
                    # print('STRAIGHTDIST = ', STRAIGHTDIST)
                    # print('IDXSEG = ', IDXSEG)
                    # print('IDXBRANCH = ', IDXBRANCH)
                
                    col += 6
                    # print('col = ', col)

                    # Prepare to remove the segments order i
                    # Dmask[DD < distanceToEndPt] = True
                    # Dmask=BWrem # BWrem is the segment without branch or endpoints # this keeps 
                    # if len(x) == 2 and k2 == 0: # processing the last of two remaining point
                    #     print('len(x) == 2 and k2 == 0: breaking for loop now')
                    #     # include both of the last branch/endpoints to be removed, as well as adjacent pts to end points to ensure breaks in any possible loops
                    #     Dmask |= BWrem_withEndPtKnobs  # Add to existing Dmask
                    #     break # Exit the for loop
                    # else:
                    #     Dmask |= BWrem  # Add to existing Dmask
                    Dmask |= BWrem
                    
                except ValueError as e:
                    print(f"Removing problematic branch point at ({y[k2]}, {x[k2]}) from B - {str(e)}")
                    B[y[k2], x[k2]] = False  # Remove from branch point mask
                    continue  # Skip to next k2 iteration



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

            print('')

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

        # earlier had this chunk above the creekordermask chunk, but not sure why, was not this way in first two while loops -SamK
        # Remove order i segments
        # skelD = skeleton_chopped - Dmask # subtract matrices
        # skelD = np.logical_xor(skeleton_chopped, Dmask) # or use skelD = skeleton_chopped ^ Dmask
        skelD = skeleton_chopped & ~Dmask  # Remove Dmask pixels from skeleton
        skeleton_chopped = skelD.astype(bool) # convert to boolean/logical, could also do skelchopped = skelD > 0 -SamK

        # # Redo skeleton - not sure why this is here - not in first two while loops -SamK
        # skeleton_chopped = morphology.skeletonize(skeleton_chopped)

        # # Find the end and branch points of the order i+1 network configuration
        # i += 1 # Store order number and move on to the next creek order until skeleton contains only 0
        # col = 0
        # skeleton_chopped = bwmorph_clean(skeleton_chopped) # not sure what the point of this is
        B, E, PTS = analyze_skeleton(skeleton_chopped, Z)
        B = PTS
        print(f"Total skeleton pixels: {np.sum(skeleton_chopped)}")
        print(f"Number of branch points found: {np.sum(B)}")
        print(f"B_loctemp coordinates: {B_loctemp}")

        i_check += 1
        if i_check >= max_iterations:
            print("Max iterations reached, breaking out of the loop.")
            break

        print('')

    print('Starting while loop 4 (loops):')
    # while loop 4 out of 4 - loop processing - but commented out while loop in AnnaM's MATLAB code
    # so maybe we don't use a while loop, but let's see if we do -SamK
    # While loop 4 - handle remaining connected components
    while np.any(skeleton_chopped) and i_check <= max_iterations:
        print(f"while 4/4, Iteration {i_check}: {np.sum(skeleton_chopped)} skeleton_chopped points remaining")
        print('order i = ', i)

        skelcomp = bwconncomp(skeleton_chopped)
        idxnum = skelcomp['n_objects']

        # Visualize skelcomp
        # Assuming `skeleton_chopped` is a 2D binary array
        labeled_components = skelcomp['labels']  # Get the labeled components
        plt.figure(figsize=(10, 10))
        plt.imshow(skeleton_chopped, cmap='gray', alpha=0.5)  # Plot the original skeleton as the background
        plt.imshow(labeled_components, cmap='nipy_spectral', alpha=0.7)  # Overlay the labeled components
        plt.colorbar(label="Component Labels")
        plt.title("Connected Components in Skeleton")
        plt.axis('off')  # Hide axes for better visualization
        plt.show()
        
        if idxnum == 0:  # no more components to process
            break
            
        for iii in range(idxnum):
            print('iii = ', iii, ' of ', idxnum)
            k = k + 1
            print('k = ', k)
            
            # Get current component
            pixel_coords = skelcomp['pixel_idx'][iii]
            component_mask = np.zeros_like(skeleton_chopped, dtype=bool)
            component_mask[pixel_coords] = True
            
            # Check if it's a closed loop (no branch/end points)
            B_comp, E_comp, _ = analyze_skeleton(component_mask, Z)
            
            # Store total pixels in component 
            sinuous_length = np.sum(component_mask, dtype=float)
            
            # Safely expand STRAHLER array if needed
            if i >= STRAHLER.shape[0] or k >= STRAHLER.shape[1]:
                new_rows = max(i + 1, STRAHLER.shape[0])
                new_cols = max(k + 1, STRAHLER.shape[1])
                STRAHLER_temp = np.zeros((new_rows, new_cols), dtype=float)
                STRAHLER_temp[:STRAHLER.shape[0], :STRAHLER.shape[1]] = STRAHLER
                STRAHLER = STRAHLER_temp
                
            # Store length
            STRAHLER[i,k] = sinuous_length
            
            # Get coordinates of first point for processing
            ptlocy, ptlocx = pixel_coords[0][0], pixel_coords[1][0]
            
            # Calculate distances from first point
            DD = calculate_chessboard_distance(component_mask, ptlocy, ptlocx)
            
            if np.sum(E_comp) == 0 and np.sum(B_comp) == 0:
                # True closed loop - process as single segment with artificial midpoint
                print(f"Processing closed loop component {iii}")
                
                # Find midpoint using max distance approach
                midpt = round(np.nanmax(DD)) # 1. Find value equal to rounded max
                linear_indices = np.where(DD.ravel() == midpt)[0] # 2. Get linear indices where Dtemp equals this value
                mid_idx = linear_indices[len(linear_indices)//2] # 3. Take middle element of these indices
                row_midpt, col_midpt = np.unravel_index(mid_idx, DD.shape) # 4. Convert back to 2D coordinates

                # print figure of current study point in current skeleton
                figure_creek_skeleton_diagnostic(skeleton_chopped, ptlocx, ptlocy, Z)
                print(f"row_midpt, col_midpt: {row_midpt}, {col_midpt}")
                
                # Use normal_coord_test_diagnostic with artificial endpoint, with exception handling
                try:
                    x1, y1, x2, y2, x3, y3, BWrem, BWrem_withEndPts, BWrem_withEndPtKnobs = normal_coord_test_diagnostic(
                        DD, sinuous_length, ptlocx, ptlocy, limit, skeleton_chopped, [row_midpt, col_midpt])
                    
                    # Store coordinates
                    if col < IDXSEG.shape[1]:
                        IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
                    else:
                        new_values = [x1, y1, x2, y2, x3, y3]
                        IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                        IDXSEG[i, -6:] = new_values
                    
                    col += 6
                    
                except ValueError as e:
                    print(f"Skipping closed loop component {iii} - could not process normal vectors: {str(e)}")
                    print("Using fallback coordinates: center and midpoint along skeleton path")
                    
                    # Calculate center coordinates of the component
                    center_y = np.mean(pixel_coords[0])
                    center_x = np.mean(pixel_coords[1])
                    
                    # Find midpoint along the skeleton path between start and end
                    # Create a simple path from start to endpoint using component_mask
                    start_point = (ptlocy, ptlocx)
                    end_point = (row_midpt, col_midpt)
                    
                    # Get all skeleton pixels in this component
                    skeleton_pixels = list(zip(pixel_coords[0], pixel_coords[1]))
                    
                    # Find midpoint pixel along the skeleton path
                    # Calculate distances from start point to all pixels in component
                    distances_from_start = {}
                    for pix_y, pix_x in skeleton_pixels:
                        dist = calculate_chessboard_distance(component_mask, ptlocy, ptlocx)
                        if not np.isnan(dist[pix_y, pix_x]):
                            distances_from_start[(pix_y, pix_x)] = dist[pix_y, pix_x]
                    
                    # Find pixel that's roughly halfway along the path
                    if distances_from_start:
                        max_dist = max(distances_from_start.values())
                        target_dist = max_dist / 2
                        # Find pixel closest to target distance
                        closest_pixel = min(distances_from_start.items(), 
                                        key=lambda x: abs(x[1] - target_dist))
                        midpoint_y, midpoint_x = closest_pixel[0]
                    else:
                        # Fallback to calculated midpoint
                        midpoint_y, midpoint_x = row_midpt, col_midpt
                    
                    # Create normal vectors
                    x1, y1 = center_x, center_y  # Center point
                    x2, y2 = midpoint_x, midpoint_y  # Midpoint along skeleton path
                    x3, y3 = center_x, center_y  # Second normal vector point
                    
                    # Store the fallback coordinates
                    if col < IDXSEG.shape[1]:
                        IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
                    else:
                        new_values = [x1, y1, x2, y2, x3, y3]
                        IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                        IDXSEG[i, -6:] = new_values
                    
                    col += 6
                    
                # Set straight distance as NaN for loops
                STRAIGHTDIST[i,k] = np.nan
                
            else:
                # Has branch/end points - process like while loop 3
                print(f"Processing component {iii} with branch/end points like while loop 3")
                
                # Initialize Dmask for this component
                Dmask_comp = np.zeros_like(skeleton_chopped, dtype=bool)
                
                # Get branch and endpoint coordinates for this component
                B_comp_loc = np.argwhere(B_comp)
                E_comp_loc = np.argwhere(E_comp)
                
                # Process each endpoint in this component
                for k_comp in range(len(E_comp_loc)):
                    y_comp, x_comp = E_comp_loc[k_comp, 0], E_comp_loc[k_comp, 1]
                    
                    # Remove the selected endpoint from E_comp_loc
                    E_comp_temp = np.delete(E_comp_loc, k_comp, axis=0)
                    
                    # Get distances from current endpoint to rest of component
                    DD_comp = calculate_geodesic_distance(component_mask, y_comp, x_comp)
                    
                    # Find distances to branch points
                    if len(B_comp_loc) > 0:
                        branch_distances_comp = DD_comp[B_comp_loc[:, 0], B_comp_loc[:, 1]]
                        finite_branch_distances_comp = branch_distances_comp[np.isfinite(branch_distances_comp)]
                        try:
                            distanceToBranchPt_comp = np.min(finite_branch_distances_comp).astype(float)
                        except ValueError:
                            distanceToBranchPt_comp = np.inf
                    else:
                        distanceToBranchPt_comp = np.inf
                    
                    # Find distances to other endpoints
                    if len(E_comp_temp) > 0:
                        end_distances_comp = DD_comp[E_comp_temp[:, 0], E_comp_temp[:, 1]]
                        finite_end_distances_comp = end_distances_comp[np.isfinite(end_distances_comp)]
                        try:
                            distanceToEndPt_comp = np.min(finite_end_distances_comp).astype(float)
                        except ValueError:
                            distanceToEndPt_comp = np.inf
                    else:
                        distanceToEndPt_comp = np.inf
                    
                    # Process based on nearest connection
                    if np.isinf(distanceToEndPt_comp) and np.isinf(distanceToBranchPt_comp):
                        # Isolated point - skip
                        continue
                    elif np.isfinite(distanceToEndPt_comp) and (np.isinf(distanceToBranchPt_comp) or distanceToEndPt_comp < distanceToBranchPt_comp):
                        # Connect to nearest endpoint
                        target_distance = distanceToEndPt_comp
                        ptlocy_comp, ptlocx_comp = np.where(DD_comp == distanceToEndPt_comp)
                        ptlocy_comp, ptlocx_comp = ptlocy_comp[0], ptlocx_comp[0]
                    else:
                        # Connect to nearest branch point
                        target_distance = distanceToBranchPt_comp
                        ptlocy_comp, ptlocx_comp = np.where(DD_comp == distanceToBranchPt_comp)
                        ptlocy_comp, ptlocx_comp = ptlocy_comp[0], ptlocx_comp[0]
                    
                    # Store segment data
                    STRAHLER[i, k] = target_distance
                    
                    # Calculate straight distance
                    skeletoneucl_comp = np.zeros_like(component_mask)
                    skeletoneucl_comp[y_comp, x_comp] = 1
                    euclD_comp = ndimage.distance_transform_edt(skeletoneucl_comp == 0)
                    disttopt_comp = euclD_comp[ptlocy_comp, ptlocx_comp]
                    STRAIGHTDIST[i, k] = disttopt_comp
                    
                    # Save index of branch point for junction angle assignment
                    ptloc_linear_comp = np.ravel_multi_index((ptlocy_comp, ptlocx_comp), DD_comp.shape)
                    IDXBRANCH[i, k] = ptloc_linear_comp
                    
                    # Create normal vectors using try-catch for error handling
                    try:
                        x1_comp, y1_comp, x2_comp, y2_comp, x3_comp, y3_comp, BWrem_comp, _, _ = normal_coord_test_diagnostic(
                            DD_comp, target_distance, x_comp, y_comp, limit, component_mask, [ptlocy_comp, ptlocx_comp])
                        
                        # Store coordinates
                        if col < IDXSEG.shape[1]:
                            IDXSEG[i, col:col+6] = [x1_comp, y1_comp, x2_comp, y2_comp, x3_comp, y3_comp]
                        else:
                            new_values = [x1_comp, y1_comp, x2_comp, y2_comp, x3_comp, y3_comp]
                            IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                            IDXSEG[i, -6:] = new_values
                        
                        col += 6
                        
                        # Add segment to removal mask
                        Dmask_comp |= BWrem_comp
                        
                    except ValueError as e:
                        print(f"Error processing segment in component {iii}: {str(e)}")
                        # Store basic coordinates as fallback
                        if col < IDXSEG.shape[1]:
                            IDXSEG[i, col:col+6] = [x_comp, y_comp, ptlocx_comp, ptlocy_comp, x_comp, y_comp]
                        else:
                            new_values = [x_comp, y_comp, ptlocx_comp, ptlocy_comp, x_comp, y_comp]
                            IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
                            IDXSEG[i, -6:] = new_values
                        col += 6
                        continue
                
                # Note: We don't remove segments here since we're processing the entire component
                # The component will be removed at the end of the loop iteration
            
            # Remove processed component directly
            skeleton_chopped[pixel_coords] = False

        # Re-analyze skeleton to detect any newly exposed branch/end points
        # This is important because removing components might expose new endpoints
        # at connection points that were previously internal to components
        B, E, PTS = analyze_skeleton(skeleton_chopped, Z)
        B = PTS
        
        i_check += 1
        if i_check >= max_iterations:
            print("Max iterations reached, breaking out of the loop.")
            break

        print('')

    # original try, true to MATLAB loop:
    # while np.any(skeleton_chopped) and i_check<= max_iterations:
    #     print(f"while 4/4, Iteration {i_check}: {np.sum(skeleton_chopped)} skeleton_chopped points remaining")
    #     print('order i = ', i)
    #     # Process remaining loops with no nodes
    #     # # Prepare the creek order plot
    #     # creekordermask = bwmorph_diag(Dmask)
    #     # # Assign creek orders
    #     # creekordersing[creekordermask != 0] = creekordermask[creekordermask != 0] * i
    #     # # Thicken the mask i times
    #     # # creekordermask = bwmorph_thicken(creekordermask, i)
    #     # creekordermask = dilation(creekordermask, disk(1))
    #     # # Assign creek orders to thickened mask
    #     # creekorder[creekordermask != 0] = creekordermask[creekordermask != 0] * i
    #     # # Clear the mask
    #     # del creekordermask
    #     # Get connected components
    #     skelcomp = bwconncomp(skeleton_chopped)
    #     # print('skelcomp:')
    #     # print(skelcomp)
    #     # Get number of objects
    #     idxnum = skelcomp['n_objects']
    #     print('idxnum = ', idxnum)

    #     # Visualize skelcomp
    #     # Assuming `skeleton_chopped` is a 2D binary array
    #     labeled_components = skelcomp['labels']  # Get the labeled components
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(skeleton_chopped, cmap='gray', alpha=0.5)  # Plot the original skeleton as the background
    #     plt.imshow(labeled_components, cmap='nipy_spectral', alpha=0.7)  # Overlay the labeled components
    #     plt.colorbar(label="Component Labels")
    #     plt.title("Connected Components in Skeleton")
    #     plt.axis('off')  # Hide axes for better visualization
    #     plt.show()

    #     if idxnum == 0:
    #         # No more components to process
    #         break

        # # loop 4/4 from line-for-line MATLAB, commented out for now -SamK
        # # Go through each remaining loop with no nodes
        # for iii in range(idxnum): # MATLAB goes 1:idxnum, but idxnum is a number of components -SamK
        #     print('iii = ', iii)
        #     print('iii = ', iii, ' of ', idxnum)
        #     k = k + 1
        #     print('k = ', k)

            # # Get pixel indices for current component
            # # print('skelcomp[pixel_idx] = \n', skelcomp['pixel_idx'])
            # idxlist = skelcomp['pixel_idx'][iii]
            # if not idxlist[0].size or not idxlist[1].size:
            #     continue
            # # Create mask for current component
            # loopmask = np.zeros_like(skeleton_chopped, dtype=bool)
            # loopmask[idxlist] = True
            # print('idxlist = ', idxlist)

            # # Store total pixels in component 
            # sinuouslength = np.sum(loopmask, dtype=float)

            # # Safely expand STRAHLER array if needed
            # if i >= STRAHLER.shape[0] or k >= STRAHLER.shape[1]:
            #     new_rows = max(i + 1, STRAHLER.shape[0])
            #     new_cols = max(k + 1, STRAHLER.shape[1])
            #     STRAHLER_temp = np.zeros((new_rows, new_cols), dtype=float)  # Explicitly set dtype
            #     STRAHLER_temp[:STRAHLER.shape[0], :STRAHLER.shape[1]] = STRAHLER
            #     STRAHLER = STRAHLER_temp
                
            # # Store length
            # STRAHLER[i,k] = sinuouslength

            # halflg = round(sinuouslength/2) # doesn't seem to be used later -SamK
            # # Defined first point
            # ptmask = np.zeros_like(skeleton_chopped, dtype=bool)
            # ptmask[idxlist[0][0], idxlist[1][0]] = True # add points at first point
            # ptlocy, ptlocx = np.where(ptmask)
            # ptloc = np.where(ptmask)
            # ptloc = ptloc[0]
            # ptlocx = ptlocx[0]
            # ptlocy = ptlocy[0]

            # # print figure of current study point in current skeleton
            # plot_matrix(skeleton_chopped, "Initial Skeleton (Binary)")

            # # Define second point halfway across the loop
            # DD = calculate_chessboard_distance(loopmask, ptlocy, ptlocx) # I THINK should be input as ptlocy, ptlocx? -SamK
            # # midpt = round(np.nanmean(DD)) # 1. Find value equal to rounded mean
            # # linear_indices = np.where(DD.ravel() == midpt)[0] # 2. Get linear indices where Dtemp equals this value
            # # mid_idx = linear_indices[len(linear_indices)//2]  # 3. Take middle element of these indices
            # # row_midpt, col_midpt = np.unravel_index(mid_idx, DD.shape) # 4. Convert back to 2D coordinates
            # # MATLAB used the mean of DD, but if the loop is truly a closed loop, the halfway point will be the max: -SamK
            # # also, if the loop is closed and this rule holds true, the while loop will become endless by just knocking off half of the component each time -SamK
            # midpt = round(np.nanmax(DD)) # 1. Find value equal to rounded max
            # linear_indices = np.where(DD.ravel() == midpt)[0] # 2. Get linear indices where Dtemp equals this value
            # mid_idx = linear_indices[len(linear_indices)//2]  # 3. Take middle element of these indices
            # row_midpt, col_midpt = np.unravel_index(mid_idx, DD.shape) # 4. Convert back to 2D coordinates

            # # print figure of current study point in current skeleton
            # print('k = ', k)
            # figure_creek_skeleton_diagnostic(skeleton_chopped, ptlocx, ptlocy, Z)

            # # print(f"mid_idx: {mid_idx}")
            # # print(f"DD.shape: {DD.shape}")
            # print(f"row_midpt, col_midpt: {row_midpt}, {col_midpt}")
            
            # # Find segment straight length - redundant for loops -SamK
            # # "Create empty image with only two points" - this is the MATLAB comment, but it seems they only create one point - SamK
            # skeletoneucl = np.zeros_like(skeleton_chopped)
            # # row_mid, col_mid = np.where(DD == midpt)  # get coordinates where value appears
            # # Set that location to 1
            # skeletoneucl[row_midpt, col_midpt] = 1
            # euclD = ndimage.distance_transform_edt(skeletoneucl == 0) # for each pixel, euclidean dist to nearest non-zero value, the midpt
            # disttopt = euclD[ptlocy, ptlocx] # distance to midpt matrix from point of interest, poi = ptloc

            # STRAIGHTDIST[i,k] = np.nan # No sinuosity for loops

            # # Save index of branch point for junction angle assignment
            # IDXBRANCH[i,k] = ptloc

            # Create normal vectors to each segment and store their coordinates
            # print('while loop 4: normal_coord_test inputs:')
            # print('   ptlocy, ptlocx = ptlocy, ptlocx = ', (ptlocy, ptlocx))
            # print('   B_loc = (row_midpt, col_midpt) = ', (row_midpt, col_midpt))
            # x1, y1, x2, y2, x3, y3, BWrem, BWrem_withEndPts, BWrem_withEndPtKnobs = normal_coord_test_diagnostic(DD, sinuouslength, ptlocx, ptlocy, limit, skeleton_chopped, [row_midpt, col_midpt], (idxnum - iii))
            
            # Dmask |= BWrem
            
            # if col < IDXSEG.shape[1]:
            #     IDXSEG[i, col:col+6] = [x1, y1, x2, y2, x3, y3]
            # else:
            #     new_values = [x1, y1, x2, y2, x3, y3]
            #     IDXSEG = np.concatenate((IDXSEG, np.zeros((IDXSEG.shape[0], 6))), axis=1)
            #     IDXSEG[i, -6:] = new_values

            # col += 6
            # # print('col = ', col)

            # # originally this was before the for loop, let's try it here where it is in the other 3 while loops: -SamK
            # # Prepare the creek order plot
            # creekordermask = bwmorph_diag(Dmask)
            # # Assign creek orders
            # creekordersing[creekordermask != 0] = creekordermask[creekordermask != 0] * i
            # # Thicken the mask i times
            # # creekordermask = bwmorph_thicken(creekordermask, i)
            # creekordermask = dilation(creekordermask, disk(1))
            # # Assign creek orders to thickened mask
            # creekorder[creekordermask != 0] = creekordermask[creekordermask != 0] * i
            # # Clear the mask
            # del creekordermask
            
            # # skeleton_chopped[idxlist[0], idxlist[1]] = False # not sure exactly how this line works, going to try what was used earlier:
            # # Prepare to remove the segments order i
            # skelD = skeleton_chopped & ~Dmask  # Remove Dmask pixels from skeleton
            # skeleton_chopped = skelD.astype(bool) # convert to boolean/logical, could also do skelchopped = skelD > 0 -SamK


            # print('')

        # i_check += 1
        # if i_check >= max_iterations:
        #     print("Max iterations reached, breaking out of the loop.")
        #     break

        # print('')
    
    print('done with all while loops')
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

    # #debugging prints:
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

def plot_matrix(matrix, title, point_coords=None, cmap='gray', alpha=1.0, crosshair_color='cyan'):
    """Helper function to plot matrices with optional crosshairs."""
    plt.figure(figsize=(10, 10))
    matrix = np.array(matrix, copy=True)  # Ensure writable
    matrix = np.transpose(matrix)
    # masked_matrix = np.nan_to_num(matrix, nan=np.nanmin(matrix)*2)
    # masked_matrix = np.nan_to_num(matrix, nan=0)
    masked_matrix = np.ma.masked_invalid(matrix) # mask NaN values
    if cmap=='gray':
        plt.imshow(masked_matrix, cmap=cmap, alpha=alpha)
    else:
        plt.imshow(masked_matrix, cmap=cmap, alpha=alpha, 
                   vmin=np.nanmin(matrix), vmax=np.nanmax(matrix), interpolation='nearest')
        plt.colorbar()
    if point_coords:
        for y, x in point_coords:
            plt.plot(y, x, 'o', markersize=10, markeredgecolor=crosshair_color, markerfacecolor='none', alpha=0.6)
    plt.title(title)
    # plt.axis('off')
    plt.show()

def normal_coord_test_diagnostic(D, distanceToPt, x, y, limit, skeleton_chopped, B_loc, k_check=None):
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
    k_check : int
        iteration value to determine if to plot
        
    Returns:
    --------
    x1, y1, x2, y2, x3, y3 : float
        Coordinates of segment points and their normal vectors.
    BWrem : ndarray
        Modified skeleton mask.
    """
    # normal_coord_test(D,  distanceToPt,    x,     y,     limit, skeleton_chopped, B_loc) # inputs -SamK
    # normal_coord_test(DD, distanceToEndPt, x[k2], y[k2], limit, skeleton_chopped, B_loctemp) # ex. 1 -SamK
    
    # Ensure B_loc is a 2D NumPy array
    B_loc = np.array(B_loc)
    if B_loc.ndim == 1:
        B_loc = B_loc.reshape(1, -1)

    # print('normal_coord_test: ')
    # print('   y, x = ', (y, x))
    # print('   B_loc = ', B_loc)
    # print('   B_loc[:, 0] = ', B_loc[:, 0])
    # print('   B_loc[:, 1] = ', B_loc[:, 1])
    # Create branch point mask
    Bmask = np.zeros_like(D, dtype=bool)
    Bmask[B_loc[:, 0], B_loc[:, 1]] = True 
    # rows, cols = np.unravel_index(B_loc, Bmask.shape)
    # Bmask[rows, cols] = True
    # Bmask.ravel()[B_loc] = True
    yB, xB = np.where(Bmask) # coords of branch points

    # BOOKMARK - start of problem I think

    # Find the indices of each segment
    Dmasktemp = np.zeros_like(D, dtype=bool)
    Dmasktemp[D < distanceToPt] = True  # Remove creek segments - make everthing farther away than point False -SamK
    # plot_matrix(Dmasktemp, "Initial Dmasktemp (Binary)", point_coords=[(y, x)], cmap='gray')
    # Calculate geodesic distance using graph approach for better accuracy
    Dtemp = calculate_chessboard_distance(Dmasktemp, y, x)
    # print("Dtemp min:", np.nanmin(Dtemp), "Dtemp max:", np.nanmax(Dtemp))
    # print("Number of finite distances:", np.sum(~np.isnan(Dtemp)))
    # does not weight diagonal differently than straight connections, compared to geodesic_distance func -SamK
    # plot_matrix(Dtemp, "Dtemp (Distance Matrix)", point_coords=[(y, x)], cmap='hot')
    # maxDtemp = np.nanmax(Dtemp)

    # ADDITIONAL STEP: remove the longer segments of Dmasktemp not connected to a point in B_loc
    # rows, cols = np.unravel_index(B_loc, Dmasktemp.shape)
    # Dmasktemp[rows, cols] = True
    Dmasktemp[B_loc[:, 0], B_loc[:, 1]] = True
    # plot_matrix(Dmasktemp, "Updated Dmasktemp with Branch Points", cmap='gray')
    Dmasktempconn = bwconncomp(Dmasktemp) # could also do labeled_array, num_features = ndimage.label(Dmasktemp)
    # gives different, discrete, connected objects in Dmasktemp -SamK
    
    # Visualize
    # Create an empty image the same size as `skeleton_chopped`
    label_image = np.zeros_like(skeleton_chopped)
    labeled_components = Dmasktempconn['pixel_idx']  # Get the labeled components
    # Assume labeled_components is a list of (y, x) coordinates
    for i_component, (y_component, x_component) in enumerate(labeled_components):
        label_image[x_component, y_component] = i_component + 1  # Assign a label (avoid 0 for visibility)
    # # Now plot the corrected version
    # plt.figure(figsize=(10, 10))
    # skeleton_chopped_plt = np.transpose(skeleton_chopped)
    # # plt.imshow(skeleton_chopped_plt, cmap='gray', alpha=0.5)
    # plt.imshow(label_image, cmap='nipy_spectral', alpha=1.0)
    # # plt.colorbar(label="Component Labels")
    # plt.title("Connected Components in Skeleton")
    # plt.show()

    XYmask = np.zeros_like(D, dtype=bool)
    XYmask[y, x] = True
    # plot_matrix(XYmask, f"XYmask (Current Branch Point, y, x = {y, x})", point_coords=[(y, x)], cmap='gray')
    XYind = np.where(XYmask)
    # XY is just the point of interest, ie the current branch point in the while loop 3/4

    idxnum = Dmasktempconn['n_objects'] # Get number of objects
    # Find which components contain pixels from XYind
    idx = [i_conn for i_conn, component in enumerate(Dmasktempconn['pixel_idx'])
        if any(np.in1d(component, XYind))]
    # Remove all other elements
    Dmasktemp = np.zeros_like(D, dtype=bool) # reset Dmasktemp -SamK
    # Set pixels using row,col coordinates
    if isinstance(idx, list): # Use first element if idx is a list:
        idx = idx[0]
    # Handle pixel_idx which contains coordinate tuples
    pixel_coords = Dmasktempconn['pixel_idx'][idx] # gives coordinates of connected pixels -SamK
    # print('Dmasktempconn[\'pixel_idx\'][idx] = ', Dmasktempconn['pixel_idx'][idx])
    row_indices = pixel_coords[0]  # first array has row indices
    col_indices = pixel_coords[1]  # second array has column indices

    # print('Dmasktemp[row_indices, col_indices] = True, Dmasktemp = ', Dmasktemp)
    Dmasktemp[row_indices, col_indices] = True # makes path along Dmasktemp of connected pixels -SamK
    # plot_matrix(Dmasktemp, "Final Dmasktemp (Connected Segment)", point_coords=[(y, x)], cmap='gray')
    # Dmasktemp[pixel_coords[1], pixel_coords[0]] = True

    ptmask1 = np.zeros_like(Dmasktemp, dtype=bool) # Dmasktemp is zeros_like D -SamK
    # ptmask1[B_loc] = True
    ptmask1[B_loc[:,0], B_loc[:,1]] = True
    # ptmask1[B_loc[0], B_loc[1]] = True # adds branch points to ptmask1 -SamK
    # ptmask1[B_loc[0][0], B_loc[1][1]] = True # adds branch points to ptmask1 -SamK
    # plot_matrix(ptmask1, "ptmask1 (Branch Points)", point_coords=[(y, x)], cmap='gray')
    # print('   B_loc = ', B_loc)
    ptmask = ptmask1 & Dmasktemp # where do branch points overlap the connected segment - eg, the other end of this segment -SamK
    # plot_matrix(ptmask, "ptmask (Branch Points Overlap)", point_coords=[(y, x)], cmap='gray')
    # print("   Dmasktemp shape and sum:", Dmasktemp.shape, np.sum(Dmasktemp))
    # print("   ptmask1 shape and sum:", ptmask1.shape, np.sum(ptmask1))
    # print("   Final ptmask shape and sum:", ptmask.shape, np.sum(ptmask))
    yptend, xptend = np.where(ptmask) # coords of branch points of connected segment - eg, other end of this segement -SamK
    print("   yptend, xptend = ", yptend, xptend)
    ptmaskind = np.where(ptmask)
    D1 = calculate_geodesic_distance(skeleton_chopped, y, x) # distance along skeleton from current k2th branch point of interest -SamK
    # print('   D1:')
    # print(D1)
    D2 = calculate_geodesic_distance(skeleton_chopped, yptend, xptend) # distance along skeleton from other end point of segment -SamK
    # plot_matrix(D1, "D1 (Geodesic Distance from Start Point)", point_coords=[(y, x)], cmap='hot')
    # plot_matrix(D2, "D2 (Geodesic Distance from End Point)", point_coords=[(y, x)], cmap='hot')
    # print('   D2:')
    # print(D2)
    Dnew = D1 + D2 # add distance rasters together -SamK
    Dnew_check1 = Dnew
    # print('   D1[199:202,199:202] = \n', D1[199:202,199:202])
    # print('   D2[199:202,199:202] = \n', D2[199:202,199:202])
    # print('   Dnew[199:202,199:202] = \n', Dnew[199:202,199:202])
    # np.savetxt('/Users/sam/Desktop/temp_check/D1.csv', D1, delimiter=',')
    # np.savetxt('/Users/sam/Desktop/temp_check/D2.csv', D2, delimiter=',')
    # np.savetxt('/Users/sam/Desktop/temp_check/Dnew1.csv', Dnew, delimiter=',')
    Dnew = np.round(Dnew * 8) / 8
    # print('   Dnew[199:202,199:202] = \n', Dnew[199:202,199:202])
    # np.savetxt('/Users/sam/Desktop/temp_check/Dnew2.csv', Dnew, delimiter=',')
    # print('max(Dnew-Dnew) after rounding = \n', max(Dnew-Dnew_check1))
    Dnew[np.isnan(Dnew)] = np.inf
    # print('   Dnew[199:202,199:202] = \n', Dnew[199:202,199:202])
    # np.savetxt('/Users/sam/Desktop/temp_check/Dnew3.csv', Dnew, delimiter=',')
    # plot_matrix(Dnew, "Dnew (Sum of D1 and D2)", point_coords=[(y, x)], cmap='hot')
    # paths = imregionalmin(Dnew) # of the geodesic distances to each end point of the segment, which pixels are the least -SamK
    # plot_matrix(paths, "Paths (Regional Minima of Dnew)", point_coords=[(y, x)], cmap='gray')
    # solution_path = morphology.thin(paths)
    # plot_matrix(solution_path, "Solution Path (Thinned)", point_coords=[(y, x)], cmap='gray')

    # Define cost array — restrict movement to skeleton only
    cost = np.where(skeleton_chopped, 1, np.inf)
    # Define start and end points — assuming only one branch point at each
    start = (y, x)
    end = (yptend[0], xptend[0])  # yptend and xptend from np.where(ptmask)
    print(f"Start point {start} in skeleton: {skeleton_chopped[start]}")
    print(f"End point {end} in skeleton: {skeleton_chopped[end]}")
    print(f"Cost at start: {cost[start]}")
    print(f"Cost at end: {cost[end]}")
    # Get the path
    indices, weight = route_through_array(cost, start, end, fully_connected=True)
    # Create the solution path mask
    solution_path = np.zeros_like(skeleton_chopped, dtype=bool)
    for r, c in indices:
        solution_path[r, c] = True
    # plot_matrix(solution_path, "Solution Path (Using route_through_array)", point_coords=[(y, x)], cmap='gray')
    
    Dmasktemp = solution_path.copy() # reset Dmasktemp to path between end points -SamK
    Dmasktemp[ptmaskind] = True # add end points -SamK
    plot_matrix(Dmasktemp, "Final Dmasktemp with Path and Endpoints", point_coords=[(y, x)], cmap='gray') # plot always
    # if k_check:
    #     if k_check < 3:
    #         plot_matrix(Dmasktemp, "Final Dmasktemp with Path and Endpoints", point_coords=[(y, x)], cmap='gray')
    BWrem_withEndPts = Dmasktemp.copy()
    BWrem_withEndPts[y, x] = True # add starting point - point of interest - SamK
    BWrem_withEndPts[yptend[0], xptend[0]] = True  # add ending point - nearest branch/endpoint - SamK

    # BWrem with endpoints, plus any adjacent points to end points - to ensure a break in loops later
    # Create masks for the endpoints
    endpoint_mask = np.zeros_like(skeleton_chopped, dtype=bool)
    endpoint_mask[y, x] = True
    endpoint_mask[yptend[0], xptend[0]] = True
    # Dilate the endpoints by 1 pixel and keep only skeleton points
    adjacent_mask = binary_dilation(endpoint_mask, structure=np.ones((3,3))) & skeleton_chopped
    # Remove the original endpoints to get only adjacent points
    adjacent_only = adjacent_mask & ~endpoint_mask
    # Add to your existing mask
    BWrem_withEndPtKnobs = Dmasktemp.copy()
    BWrem_withEndPtKnobs[y, x] = True
    BWrem_withEndPtKnobs[yptend[0], xptend[0]] = True
    BWrem_withEndPtKnobs[adjacent_only] = True

    BWrem = Dmasktemp.copy()
    BWrem[ptmaskind] = False # path between end points without endpoints -SamK
    # plot_matrix(BWrem, "BWrem (Path Without Endpoints)", point_coords=[(y, x)], cmap='gray')

    # BOOKMARK - end of problem I think
    
    maxDtemp = np.nansum(Dmasktemp) # sum of how many pixels in path -SamK
    
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
        Dtemp = calculate_geodesic_distance(Dmasktemp, y, x) # distance raster just along path and endpts, to k2th branch point -SamK
        # # Dtemp_nonan = Dtemp[~np.isnan(Dtemp)]
        # mid_value = round(np.nanmean(Dtemp))
        # row_midpts, col_midpts = np.where(Dtemp == mid_value)
        # midpts = Dtemp[row_midpts, col_midpts]
        # midpt = midpts[len(midpts)//2] 
        # # midpt = np.median(midpts.flatten())
        midpt = round(np.nanmean(Dtemp)) # 1. Find value equal to rounded mean
        # print('midpt = ', midpt)
        # Get the linear indices where Dtemp is within certain error of midpt distance
        linear_indices = np.where(np.abs(Dtemp.ravel() - midpt) <= 0.9)[0] # 2. Get linear indices where Dtemp equals this value
        # linear_indices = np.where(Dtemp.ravel() == midpt)[0] # 2. Get linear indices where Dtemp equals this value
        # print('linear_indices = ', linear_indices)
        mid_idx = linear_indices[len(linear_indices)//2]  # 3. Take middle element of these indices
        # print('mid_idx = ', mid_idx)
        row_midpt, col_midpt = np.unravel_index(mid_idx, Dtemp.shape) # 4. Convert back to 2D coordinates
        
        MIDPT = np.zeros_like(Dtemp, dtype=bool)
        # row_mid, col_mid = np.where(Dtemp == midpt)  # get coordinates where value appears
        # Set that location to True
        # MIDPT[row_mid, col_mid] = True
        MIDPT[row_midpt, col_midpt] = True
        ym, xm = np.where(MIDPT) # instead of find(), ind2sub() is commented out by AnnaM in MATLAB -SamK
        ym, xm = ym[0], xm[0] # ensure they are integer datatypes
        
        # Find the two points at a distance of 0.2*length of midpoint of long segment
        dist = round(maxDtemp/4)
        mid_distances = calculate_chessboard_distance(Dmasktemp, ym, xm) # find distances from midpoint coordinates
        ALLIDX = np.where(mid_distances == dist)
        if ALLIDX[0].size < 2:
            # print('ALLIDX[0].size < 2, = ', ALLIDX[0].size)
            x1, y1 = x, y
            x2, y2 = xptend[0], yptend[0]
        # if ALLIDX[0].size == 0:  # if ALLIDX is from np.where()
        #     x1, y1 = x[0], y[0]
        #     x2, y2 = xptend[0], yptend[0]
        else:
            # print('ALLIDX[0]:', ALLIDX[0])
            # print('ALLIDX[1]:', ALLIDX[1])
            # print('len(ALLIDX[0]):', len(ALLIDX[0]))
            # print('len(ALLIDX[1]):', len(ALLIDX[1]))

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
    
    return x1, y1, x2, y2, x3, y3, BWrem, BWrem_withEndPts, BWrem_withEndPtKnobs

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
            - 'labels': 2D array of labeled components (like MATLAB's bwlabel)
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
        'pixel_idx': pixel_idx,
        'labels': labeled
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

def calculate_geodesic_distance(mask, target_ys, target_xs):
    """
    Calculate geodesic distance from all points in a binary mask to the nearest
    target points using a graph approach.

    Parameters:
        mask (np.ndarray): Binary array representing the skeleton (True/1 for valid points).
        target_ys (int or np.ndarray): Single y-coordinate or array of y-coordinates of the target points.
        target_xs (int or np.ndarray): Single x-coordinate or array of x-coordinates of the target points.

    Returns:
        np.ndarray: Array of geodesic distances with the same shape as mask.
    """
    import networkx as nx  # type: ignore

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
                    weight = np.sqrt(dy**2 + dx**2)  # √2 for diagonal, 1 for horizontal/vertical
                    G.add_edge((y, x), (num_y, num_x), weight=weight)

    # Initialize distance array
    distances = np.full_like(mask, np.nan, dtype=float)

    # Normalize inputs to be iterable
    if isinstance(target_ys, (int, np.integer)) and isinstance(target_xs, (int, np.integer)):
        target_points = [(target_ys, target_xs)]
    else:
        target_points = list(zip(target_ys, target_xs))

    # Filter valid target points
    valid_targets = [point for point in target_points if point in G.nodes()]

    if valid_targets:
        # Add a virtual source node connected to all valid targets with zero weight
        source_node = 'virtual_source'
        for target in valid_targets:
            G.add_edge(source_node, target, weight=0)

        # Compute shortest path lengths from the virtual source
        path_lengths = nx.single_source_dijkstra_path_length(G, source_node, weight='weight')

        # Fill distances array
        for node, dist in path_lengths.items():
            if isinstance(node, tuple) and len(node) == 2:  # Ensure it's a valid coordinate (y, x)
                distances[node[0], node[1]] = dist

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
    # print('mask.shape: ', mask.shape)
    # print("Mask sum:", np.sum(mask))  # Should be >1 for multiple valid points
    # print("Start point in mask:", mask[start_y, start_x])

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
                if G.has_node((newy, newx)):
                    G.add_edge((y, x), (newy, newx), weight=1)
    
    # print(f"Graph has {len(G.nodes())} nodes and {len(G.edges())} edges")
    # print(list(G.edges)[:10])
    
    # Calculate distances
    distances = np.full_like(mask, np.nan, dtype=float)
    # distances = np.full(mask.shape, np.inf, dtype=float)
    try:
        if (start_y, start_x) in G.nodes():
            path_lengths = nx.single_source_dijkstra_path_length(
                G, (start_y, start_x), weight='weight')
            # print(f"Number of computed distances: {len(path_lengths)}")
            for (y, x), dist in path_lengths.items():
                distances[y, x] = dist
    except Exception as e:
        print(f"Error in distance calculation: {str(e)}")
        print(f"Start point: ({start_y}, {start_x})")
        print(f"Graph nodes: {len(G.nodes())}")

    # print('distances.shape: ', mask.shape)
    # print("distances sum:", np.sum(mask))  # Should be >1 for multiple valid points
    
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

# def bwmorph_clean(image):
#     """
#     Remove isolated pixels (1's surrounded by 0's)
#     """
#     # Create 3x3 structuring element
#     struct = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
#     # Count neighbors
#     neighbors = ndimage.convolve(image.astype(int), struct, mode='constant')
#     # Remove pixels with no neighbors
#     return image & (neighbors > 1)

def bwmorph_clean(image):
    """Remove truly isolated pixels only (no neighbors at all)"""
    struct = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
    neighbors = ndimage.convolve(image.astype(int), struct, mode='constant')
    return image & (neighbors > 0)  # Only remove completely isolated pixels

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