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

STRAHLER = variables["STRAHLER"]
STRAIGHTDIST = variables["STRAIGHTDIST"]
IDXBRANCH = variables["IDXBRANCH"]
IDXSEG = variables["IDXSEG"]

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

def save_processed_values(analyzer, skeleton, X, Y, pts, order_max):
    """
    Save the processed values from the CreekNetworkAnalyzer to a JSON file.
    
    Parameters:
    analyzer (CreekNetworkAnalyzer): The analyzer instance after processing
    skeleton (np.ndarray): The skeleton array
    X (np.ndarray): X coordinates
    Y (np.ndarray): Y coordinates
    pts (np.ndarray): Points array
    order_max (int): Maximum order value
    """
    # Create dictionary with all variables
    variables_correction = {
        "skeleton": skeleton,
        "X": X,
        "Y": Y,
        "creek_order": analyzer.creek_order,
        "creek_order_single": analyzer.creek_order_single,
        "creek_order_swapped": analyzer.creek_order_swapped,
        "creek_order_single_swapped": analyzer.creek_order_single_swapped,
        "PTS": pts,
        "order_max": order_max,
        "STRAHLER": analyzer.STRAHLER,
        "STRAIGHTDIST": analyzer.STRAIGHTDIST,
        "IDXBRANCH": analyzer.IDXBRANCH,
        "IDXSEG": analyzer.IDXSEG
    }

    # Convert all numpy arrays to lists
    variables_correction = {
        key: value.tolist() if isinstance(value, np.ndarray) else value 
        for key, value in variables_correction.items()
    }

    # Save to a JSON file
    with open("processed_variables_Strahler_Correction.json", "w") as f:
        json.dump(variables_correction, f)
    
    print("Processed values saved to 'processed_variables_Strahler_Correction.json'")


def main(skeleton, X, Y, creek_order, creek_order_single, pts, order_max):
    """Execute creek network analysis with proper data validation and conversion"""

    try:
        # Convert to numpy arrays first
        skeleton = convert_to_numpy(skeleton, dtype=bool)
        X_1d = convert_to_numpy(X, dtype=float)
        Y_1d = convert_to_numpy(Y, dtype=float)
        creek_order = convert_to_numpy(creek_order, dtype=float)
        pts = convert_to_numpy(pts, dtype=bool)
        order_max = int(order_max)

        # Create proper meshgrids for X and Y
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

    except Exception as e:
        print(f"Error during data conversion: {e}")
        return

    try:
        # Initialize analyzer
        analyzer = CreekNetworkAnalyzer(skeleton, X, Y, creek_order, creek_order_single, pts, order_max, STRAHLER, STRAIGHTDIST, IDXBRANCH, IDXSEG)
        
        # Process creek orders
        analyzer.swap_creek_orders()
        
        # Launch GUI
        print("GUI Instructions:")
        print("1. Click 'Correct Creek Segment' to start segment correction")
        print("2. Choose the order number from the dropdown")
        print("3. Click two points on the right plot to select a segment")
        print("4. Repeat for other segments as needed")
        print("5. Click 'Finish Correction' when done")
        
        analyzer.create_correction_gui()
        
        # Process corrected segments
        print("Processing corrected segments...")
        analyzer.process_corrected_segments()

        # Save processed values
        save_processed_values(analyzer, skeleton, X, Y, pts, order_max)
        
        print("Analysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    try:
        main(skeleton, X, Y, creek_order, creek_order_single, pts, order_max)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise