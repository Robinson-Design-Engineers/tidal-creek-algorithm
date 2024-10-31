# generated with Claude AI and reviewed by Sam Kraus, 2024/09/12
import tkinter as tk
from tkinter import filedialog
import os
import numpy as np
import matplotlib.pyplot as plt
# from scipy.io import savemat
import h5py
import easygui

def loadinmultiLiDAR(filename, fileselect, img):
    """
    Python equivalent of MATLAB's loadinmultiLiDAR function.
    """
    # 1. Load in data
    # open file of interest, 'r' means read, store each line as element of list called 'lines'
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    """
    use '.split' to split each line up by white spaces, assume second element of the row is the value of interest
    MATLAB function goes line by line and indexes cerpip install tk
    tain characters (i.e., 15:18), 
    but this seems to work with ascii format for raster files
    """
    ncols = int(lines[0].split()[1]) # MATLAB indexes tline's 15:18, if error try 15:17
    nrows = int(lines[1].split()[1]) # MATLAB indexes tline's 15:18, if error try 15:17
    xc = float(lines[2].split()[1])
    yc = float(lines[3].split()[1])
    gs = float(lines[4].split()[1]) # MATLAB's tline(15:18) for cellsize=0.25, tline(15:17) for cellsize=0.5, tline(15:15) for cellsize=1 or 2 
    nf = int(lines[5].split()[1])
    
    Z = np.loadtxt(filename, skiprows=6)
    
    # Remove null values
    Z[Z == nf] = np.nan


    # 2. Create X and Y mesh
    # x = np.linspace(xc, xc + (ncols - 1) * gs, ncols)
    x = np.arange(xc, xc+ncols*gs, gs)
    # y = np.linspace(yc + (nrows - 1) * gs, yc, nrows)[::-1]
    y = np.arange(yc+(nrows-1)*gs, yc, -gs)
    X, Y = np.meshgrid(x, y)
    
    # Ensure Z has the correct shape - removes first and last rows of Z
    if Z.shape[0] > X.shape[0]:
        Z = Z[1:-1, :]
    elif Z.shape[1] > X.shape[1]:
        Z = Z[:, 1:-1]
    
    # # 3. Save as .mat file
    # name_out = os.path.join('INPUTS', f'{fileselect}.mat')
    # savemat(name_out, {'X': X, 'Y': Y, 'Z': Z})
    
    # name_out2 = os.path.join('INPUTS', f'{fileselect}_metadata.mat')
    # savemat(name_out2, {'gs': gs, 'ncols': ncols, 'nrows': nrows, 'nf': nf, 'x': x, 'y': y, 'xc': xc, 'yc': yc})

    # 3. Save as .h5 file
    name_out = os.path.join('INPUTS', f'{fileselect}.mat')
    with h5py.File(name_out, 'w') as hf:
        hf.create_dataset('X', data=X)
        hf.create_dataset('Y', data=Y)
        hf.create_dataset('Z', data=Z)

    with h5py.File(name_out, 'w') as hf:
        hf.create_dataset('gs', data=gs)
        hf.create_dataset('ncols', data=ncols)
        hf.create_dataset('nrows', data=nrows)
        hf.create_dataset('nf', data=nf)
        hf.create_dataset('x', data=x)
        hf.create_dataset('y', data=y)
        hf.create_dataset('xc', data=xc)
        hf.create_dataset('yc', data=yc)
    
    # 4. Plot data
    if img == 1:
        plt.figure(figsize=(10, 8))
        plt.pcolor(X, Y, Z, shading='auto')
        plt.colorbar()
        plt.title(f"Data from {fileselect}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    
    return X, Y, Z, ncols, nrows, x, y, xc, yc, gs, nf

def LOADINFILES(img=1):
    """
    Python equivalent of MATLAB's LOADINFILES function.
    
    :param img: If img==1, a figure will be drawn
    :return: Various data loaded from the selected file
    """
    # root = tk.Tk()
    # root.withdraw()  # Hide the main window
    
    # file_path = filedialog.askopenfilename(
    #     title='Select a file',
    #     filetypes=[('Text files', '*.txt')],
    #     initialdir='INPUTS'
    # )

    file_path = easygui.fileopenbox(title='Select a file', filetypes=['*.txt'])
    # if file_path:
    #     print(f"Selected file: {file_path}")

    # if __name__ == "__main__":
    #     load_files()
    
    if file_path:
        pathname, fileselect = os.path.split(file_path)
        return loadinmultiLiDAR(file_path, fileselect, img)
    else:
        print('No file selected')
        return None



# # Usage example
# result = loadinfiles()
# if result:
#     X, Y, Z, ncols, nrows, x, y, xc, yc, gs, nf = result
#     print(f"Loaded {nf} points")
#     print(f"Grid size: {gs}")
#     print(f"Number of columns: {ncols}")
#     print(f"Number of rows: {nrows}")