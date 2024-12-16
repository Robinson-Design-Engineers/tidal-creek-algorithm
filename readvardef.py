# generated with Claude AI and ChatGPT, reviewed by Sam Kraus 2024/09/13
import numpy as np
import h5py
from tkinter import filedialog, Tk

def readvardef(filename_UserVar):
# def readvardef(filename_UserVar): # manual input option
    # Initialize variables
    name = shortname = elev = elevmeta = slope = slopemeta = resamplestep = threshold = detrendyn = outletdetection = None
    HAT = MHWS = MHWN = MLWS = MLWN = Cth = Ctharea = LZth = LZtharea = HZth = HZtharea = None
    nbbreaches = noisethreshold = reconnect = reconnectiondist = ordermax = None
    filtersmall1 = filterlarge1 = filtersmall2 = filterlarge2 = smoothing = None
    connectivity = holesizeinfill = None
    
    # Open file dialog
    # # GUI option
    # root = Tk()
    # root.withdraw()  # Hide the main window
    # filename = filedialog.askopenfilename(initialdir="INPUTS", 
    #                                       title="Select variable definitions file",
    #                                       filetypes=(("Text files", "*.txt"), ("all files", "*.*")))
    filename = filename_UserVar # manual input option
    if not filename:
        return None  # User aborted
    
    # Read and evaluate the file content
    with open(filename, 'r') as file:
        content = file.read()
        
    # # WARNING: Using eval can be dangerous. Make sure the input file is trusted.
    # try:
    #     exec(content)
    # except Exception as e:
    #     print(f"Error in settings file: {str(e)}")
    #     return None

    # WARNING: Using exec can be dangerous. Make sure the input file is trusted.
    try:
        # Use a separate dictionary for exec
        exec_globals = {}
        exec(content, exec_globals)
    except Exception as e:
        print(f"Error in settings file: {str(e)}")
        return None
    
    # Now assign variables from exec_globals
    name = exec_globals.get("name")
    shortname = exec_globals.get("shortname")
    elev = exec_globals.get("elev")
    elevmeta = exec_globals.get("elevmeta")
    slope = exec_globals.get("slope")
    slopemeta = exec_globals.get("slopemeta")
    resamplestep = exec_globals.get("resamplestep")
    threshold = exec_globals.get("threshold")
    detrendyn = exec_globals.get("detrendyn")
    outletdetection = exec_globals.get("outletdetection")
    
    HAT = exec_globals.get("HAT")
    MHWS = exec_globals.get("MHWS")
    MHWN = exec_globals.get("MHWN")
    MLWS = exec_globals.get("MLWS")
    MLWN = exec_globals.get("MLWN")
    
    Cth = exec_globals.get("Cth")
    Ctharea = exec_globals.get("Ctharea")
    LZth = exec_globals.get("LZth")
    LZtharea = exec_globals.get("LZtharea")
    HZth = exec_globals.get("HZth")
    HZtharea = exec_globals.get("HZtharea")
    
    nbbreaches = exec_globals.get("nbbreaches")
    noisethreshold = exec_globals.get("noisethreshold")
    reconnect = exec_globals.get("reconnect")
    reconnectiondist = exec_globals.get("reconnectiondist")
    ordermax = exec_globals.get("ordermax")
    
    filtersmall1 = exec_globals.get("filtersmall1")
    filterlarge1 = exec_globals.get("filterlarge1")
    smoothing = exec_globals.get("smoothing")
    connectivity = exec_globals.get("connectivity")
    holesizeinfill = exec_globals.get("holesizeinfill")
    filtersmall2 = exec_globals.get("filtersmall2")
    filterlarge2 = exec_globals.get("filterlarge2")

    
    # Create the dictionary structures
    FILENAMES = {"name": name, "shortname": shortname, "elev": elev, "elevmeta": elevmeta, "slope": slope, "slopemeta": slopemeta}
    PROCESSING = {"resamplestep": resamplestep, "threshold": threshold, "detrendyn": detrendyn, "outletdetection": outletdetection}
    TIDE = {"HAT": HAT, "MHWS": MHWS, "MLWS": MLWS, "MHWN": MHWN, "MLWN": MLWN}
    THRESHOLDS = {"Cth": Cth, "Ctharea": Ctharea, "LZth": LZth, "LZtharea": LZtharea, "HZth": HZth, "HZtharea": HZtharea}
    RECONNECTION = {"nbbreaches": nbbreaches, "noisethreshold": noisethreshold, "reconnect": reconnect, 
                    "reconnectiondist": reconnectiondist, "ordermax": ordermax, "filtersmall1": filtersmall1, 
                    "filterlarge1": filterlarge1, "smoothing": smoothing, "connectivity": connectivity, 
                    "holesizeinfill": holesizeinfill, "filtersmall2": filtersmall2, "filterlarge2": filterlarge2}
    
    return FILENAMES, PROCESSING, TIDE, THRESHOLDS, RECONNECTION