import numpy as np
import json
from skimage.morphology import skeletonize
from Strahler_Correction_class import CreekNetworkAnalyzer  # type: ignore # This is from the class created for the algorithm step 6

# Load the variables from the JSON file
with open("variables_Strahler_Correction.json", "r") as f:
    variables = json.load(f)

skeleton = variables["skeleton"]
X = variables["X"]
Y = variables["Y"]
creek_order = variables["creek_order"]
pts = variables["pts"]
order_max = variables["order_max"]

def main(skeleton, X, Y, creek_order, pts, order_max):
    # Initialize the analyzer
    print("Initializing Creek Network Analyzer...")
    analyzer = CreekNetworkAnalyzer(skeleton, X, Y, creek_order, pts, order_max)
    
    # Swap creek orders
    print("Processing creek orders...")
    analyzer.swap_creek_orders()
    
    # Launch the GUI
    print("Launching GUI...")
    print("Instructions:")
    print("1. Click 'Correct Creek Segment' to start segment correction")
    print("2. Click two points on the right plot to select a segment")
    print("3. Choose the order number from the dropdown")
    print("4. Repeat for other segments as needed")
    print("5. Click 'Finish Correction' when done")
    
    analyzer.create_correction_gui()
    
    # After GUI closes, process the corrected segments
    print("Processing corrected segments...")
    analyzer.process_corrected_segments()
    
    print("Analysis complete!")

if __name__ == "__main__":
    main(skeleton, X, Y, creek_order, pts, order_max)