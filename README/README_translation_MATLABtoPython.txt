README for MATLB translation to Python of CHIROL_CREEK_EXTRACTION algorithm

Notes: 
- Anna Logan McClendon (ALM, mcclend, Anna) worked on the MATLAB script for RDE project Ashleyville Marsh Summer 2024
- Sam Kraus (SamK, Sam) worked on translating the MATLAB script to Python following Anna's work September 2024
- Anna had to modify some of Chirol's original MATLAB code - this modified code is what Sam worked to translate

Loose Timeline:
LOADINFILES.py was an attempt to translate LOADINFILES.m, but it seems Anna did not use this.
Indeed, LOADINFILES.py doesn't work anyway due to issues with using a GUI to select input files. This seems unnecessary anyway.
Anna used input_formatting.m, which doesn't use GUI but instead requires manual entry of elevation and slope ascii filepaths - perfectly acceptable.
input_formatting.py translates input_formatting.m and makes it into a python function requiring the argument of the input ascii filepaths.

After LOADINFILES, Sam only worked on using Anna's modified MATLAB code, disregarding the original Chirol Dataset code.
Chirol's Dataset and original code is available for download here: http://dx.doi.org/10.5258/SOTON/D1114

CHIROL_CREEK_EXTRACTION_2024.m:

Step 1:
- had to change out GUI file selection for manual filepath option again - Python GUI should work, Sam just doesn't have experience or see need to debug at this step.
- filename for UserVar .txt file is passed as one argument from readvardef.py file's readvardef function
- had some issues when trying to use simple exec() to load variable names from UserVar .txt file, ChatGPT came up with fix by setting .txt file variables as global variables
- had to change all '%' MATLAB comment symbols to '#' Python comment symbols in UserVar .txt file
- changed '\' backward slash in MATLAB to '/' forward slash for filepaths - backward slash does not seem as universally compatible depending on operating system
- readvardef function is in .py file, rest of step 1 is in .ipynb jupyter notebook cell

Step 2:
- changed X2 = X[::resamplestep, ::resamplestep] to X2 = X[::resamplestep] -> X2 was 2D in Matlab, maybe not in Python, gave error when trying to treat it as such. It makes sense this 1D indexing in Python.

Step 3:
- GUI used to select thresholds for hypsometry and slope curves to detect creek extents
- GUI in python needs to be run from a .py file in terminal - instructions are in Jupyter notebook .ipynb file
- hex star markers in hypsometry curve were changed to X markers
- Python GUI acts differently from MATLAB - required several edits
- rendering in Python GUI is slower at high resolution of color interpolation - cstride and rstride were set to help this at lower resolution for the 3D plot
- a 2D plot was also created at a higher color resolution / interpolation 
	- the 2D plot does not seem to have an easy way to make a gridline mesh of the creek mask as in 3D
	- the 3D plot uses plt.plot_surface, which creates a 3D colormesh with gridlines
	- the 2D plot uses plt.pcolormesh
	- some weird things happen when using mesh gridlines, where an entire rectangular domain will show cells, which requires masking based on the target area with actual data
- Creek_Detection.py contains all the functions that are contained in step 3 of the MATLAB code in CHIROL_CREEK_ALGORITHM.m
- Anna's mcclend code has the colormap's label as Elevation (ft) but everything else is labeled as (m). Sam replaced "(ft)" with "(m)", assuming this is something Anna neglected to change.
- If threshold was set to 1, manual input requires clicking on slope and hypsometry curves. Keep clicking until registered - should display small red cross briefly, then go to next figure. Once all thresholds are chosen, final figures will be displayed.
- Creek_Detection.py saves raw creek mask as .h5 file

Step 4:
- Creek_Repair.py file contains all functions needed for this step
- saving to .h5 files instead of .mat files
- MATLAB has a lot of built in functions for image processing, Claude AI created a lot of versions of Python code to smooth and fill holes and noise. The final version has alternate functions for reconnect and repair functions ending in "_diagnostic" - this has debugging print statements and plot outputs. 
- Has scikit-image functions to do a lot of image processing alternatives to MATLAB.
- Overall, a lot of functions look different than MATLAB - Python achieves a smoothed and filtered mask, but with different functions.
- Sam used Claude AI to put this together, and it tested to have very similar outputs for Ashleyville Marsh's ashleyville site compared to MATLAB.

Step 5:
- all functions are found in the Creek_Ordering.py file
- MATLAB uses built in packages for image processing like bwmorph - Python uses the scikit-image package's functions
- any "_diagnostic" or commented out sections in the .py file are generally debugging print statements, to be used if the algorithm is not running properly
- branch point detection was being troublesome for Python - developed a new method in which only certain linkages qualify for either to ensure there are not neighboring branch points
- the process_outlet_detection() function took some iterations through Claude AI to get right - in checking for where to connect the skeleton to the edge of the land mask, the border of the mask is used to look for possible creek outlets within 5 pixels of this border. This may need to be changed if problems arise in the future with no outlets being detected.
- added an output of "skeleton_breached" that saves the skeleton with the outlet breach.
