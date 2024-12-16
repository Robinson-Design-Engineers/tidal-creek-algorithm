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

Section 1:
- had to change out GUI file selection for manual filepath option again - Python GUI should work, Sam just doesn't have experience or see need to debug at this step.
- filename for UserVar .txt file is passed as one argument from readvardef.py file's readvardef function
- had some issues when trying to use simple exec() to load variable names from UserVar .txt file, ChatGPT came up with fix by setting .txt file variables as global variables
- had to change all '%' MATLAB comment symbols to '#' Python comment symbols in UserVar .txt file
- changed '\' backward slash in MATLAB to '/' forward slash for filepaths - backward slash does not seem as universally compatible depending on operating system
- readvardef function is in .py file, rest of step 1 is in .ipynb jupyter notebook cell
- added a "shortname" variable to the UserVar.txt input file - this is used for adding an identifier name to saved files rather than the "name" variable, which is used in titles and inclueds spaces

Section 2:
- changed X2 = X[::resamplestep, ::resamplestep] to X2 = X[::resamplestep] -> X2 was 2D in Matlab, maybe not in Python, gave error when trying to treat it as such. It makes sense this 1D indexing in Python.

Section 3:
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

Section 4:
- Creek_Repair.py file contains all functions needed for this step
- saving to .h5 files instead of .mat files
- MATLAB has a lot of built in functions for image processing, Claude AI created a lot of versions of Python code to smooth and fill holes and noise. The final version has alternate functions for reconnect and repair functions ending in "_diagnostic" - this has debugging print statements and plot outputs. 
- Has scikit-image functions to do a lot of image processing alternatives to MATLAB.
- Overall, a lot of functions look different than MATLAB - Python achieves a smoothed and filtered mask, but with different functions.
- Sam used Claude AI to put this together, and it tested to have very similar outputs for Ashleyville Marsh's ashleyville site compared to MATLAB.

Section 5:
- all functions are found in the Creek_Ordering.py file
- MATLAB uses built in packages for image processing like bwmorph - Python uses the scikit-image package's functions
- any "_diagnostic" or commented out sections in the .py file are generally debugging print statements, to be used if the algorithm is not running properly
- branch point detection was being troublesome for Python - developed a new method in which only certain linkages qualify for either to ensure there are not neighboring branch points
- the process_outlet_detection() function took some iterations through Claude AI to get right - in checking for where to connect the skeleton to the edge of the land mask, the border of the mask is used to look for possible creek outlets within 5 pixels of this border. This may need to be changed if problems arise in the future with no outlets being detected.
- added an output of "skeleton_breached" that saves the skeleton with the outlet breach.
- geodesic distance calculation was formed as a function by ClaudeAI using a graph approach to calculate the distance along the network skeleton rather than euclidean distance, similar to the way geodesic distance is calculated in MATLAB
- MATLAB indexing of matrices relies on linear indexing, where the python version uses row, column indexing, where rows are x and columns are y as is the norm in image matrices; although sometimes linear indices are used
- some arguments of functions seemed to switch x and y for the kth point when calculating normal vectors using the normal_coord functions - this was kept true to the MATLAB unless errors arised, where x and y were switched to what seemed correct and consistent
- instead of just passing isolated points as in MATLAB, these points are removed and set to False in the image matrix
- the creek mask is thickened using binary dilation rather than MATLAB's bwmorph_thicken
- while loop 4 out of 4 was not kept as a while loop in AnnaM's MATLAB code, but was used as a while loop for the python translation
- one bug is noted in github issues tab for outlet point inclusion

Section 6:
- correction GUI works mostly similar to MATLAB
- snapping function is added to select only either branch or end points
- green dots appear for snapped points, the second point does not show a green dot, the creek orders are just immediately updated
- the geodesic distance function formed in python for step 5 is used to create the path between the two selected points
- one bug is noted in github issues tab for inclusion of a branch point in the new order - this is something that can be corrected currently, but could be made more efficient in future
- zooming is allowed in python tkinter GUI, this allows for more accurate selection. Zoom windows will be held even after the creek order plot on the left is updated.
- perhaps future implementation could use point selection to create new branch or end points.

Section 7:
- not many differences - just plotting product of creek ordering and corrections from Sections 5 and 6

Section 8:
