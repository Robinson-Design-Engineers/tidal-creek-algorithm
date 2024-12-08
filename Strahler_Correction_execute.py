import numpy as np # type: ignore
import json
from skimage.morphology import skeletonize # type: ignore
from Strahler_Correction_class import CreekNetworkAnalyzer  # type: ignore # This is from the class created for the algorithm step 6

# Load the variables from the JSON file
with open("variables_Strahler_Correction.json", "r") as f:
    variables = json.load(f)

skeleton = variables["skeleton"]
X = variables["X"]
Y = variables["Y"]
creek_order = variables["creek_order"]
creek_order_single = variables["creek_order_single"]
pts = variables["pts"]
order_max = variables["order_max"]

def create_meshgrid(x, y):
    """Create proper 2D meshgrid from 1D coordinate arrays"""
    # Ensure inputs are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Create meshgrid
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X, Y

def print_data_info(name, data):
    """Helper function to safely print information about data structure"""
    print(f"\n{name}:")
    print(f"Type: {type(data)}")
    if isinstance(data, np.ndarray):
        print(f"Shape: {data.shape}")
        print(f"Sample: {data.flatten()[0] if data.size > 0 else None}")
    elif isinstance(data, list):
        print(f"Length: {len(data)}")
        print(f"Sample: {data[0] if data else None}")
        if data and isinstance(data[0], list):
            print(f"Inner length: {len(data[0])}")
    else:
        print(f"Value: {data}")

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

def main(skeleton, X, Y, creek_order, creek_order_single, pts, order_max):
    """Execute creek network analysis with proper data validation and conversion"""
    # print("\nInput data information:")
    # print_data_info("skeleton", skeleton)
    # print_data_info("X", X)
    # print_data_info("Y", Y)
    # print_data_info("creek_order", creek_order)
    # print_data_info("pts", pts)
    # print_data_info("order_max", order_max)

    # print("\nConverting inputs to numpy arrays...")
    try:
        # Convert to numpy arrays first
        skeleton = convert_to_numpy(skeleton, dtype=bool)
        X_1d = convert_to_numpy(X, dtype=float)
        Y_1d = convert_to_numpy(Y, dtype=float)
        creek_order = convert_to_numpy(creek_order, dtype=float)
        pts = convert_to_numpy(pts, dtype=bool)
        order_max = int(order_max)

        # Create proper meshgrids for X and Y
        # print("\nCreating coordinate meshgrids...")
        X, Y = create_meshgrid(X_1d, Y_1d)
        
        # Ensure all arrays have the same shape
        target_shape = skeleton.shape
        if X.shape != target_shape:
            X = X[:target_shape[0], :target_shape[1]]
        if Y.shape != target_shape:
            Y = Y[:target_shape[0], :target_shape[1]]
        if creek_order.shape != target_shape:
            creek_order = creek_order[:target_shape[0], :target_shape[1]]
        if pts.shape != target_shape:
            pts = pts[:target_shape[0], :target_shape[1]]

        # print("\nArray shapes after conversion:")
        # print(f"skeleton: {skeleton.shape}")
        # print(f"X: {X.shape}")
        # print(f"Y: {Y.shape}")
        # print(f"creek_order: {creek_order.shape}")
        # print(f"pts: {pts.shape}")

    except Exception as e:
        print(f"Error during data conversion: {e}")
        return

    try:
        # print("\nInitializing Creek Network Analyzer...")
        analyzer = CreekNetworkAnalyzer(skeleton, X, Y, creek_order, creek_order_single, pts, order_max)
        
        # print("Processing creek orders...")
        analyzer.swap_creek_orders()
        
        # print("Launching GUI...")
        print("GUI Instructions:")
        print("1. Click 'Correct Creek Segment' to start segment correction")
        print("2. Click two points on the right plot to select a segment")
        print("3. Choose the order number from the dropdown")
        print("4. Repeat for other segments as needed")
        print("5. Click 'Finish Correction' when done")
        
        analyzer.create_correction_gui()
        
        print("Processing corrected segments...")
        analyzer.process_corrected_segments()
        
        print("Analysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    try:
        # # Load your data
        # creek_order = np.load('creek_order.npy', allow_pickle=True).tolist()  # If it's saved as a list
        # skeleton = np.load('skeleton.npy')
        # X = np.load('X.npy')
        # Y = np.load('Y.npy')
        # pts = np.load('pts.npy')
        # order_max = 6

        # # Print initial data structure
        # print("\nInitial data loaded:")
        # print_data_info("creek_order", creek_order)

        # if isinstance(creek_order, list):
        #     print("Sample of creek_order list:")
        #     print("First element:", creek_order[0])
        #     if isinstance(creek_order[0], list):
        #         print("First inner element:", creek_order[0][0])
        
        main(skeleton, X, Y, creek_order, creek_order_single, pts, order_max)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise