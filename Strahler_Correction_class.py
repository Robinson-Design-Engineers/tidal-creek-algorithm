import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from scipy import ndimage
from skimage.morphology import skeletonize, thin
import scipy.spatial.transform as transform

class CreekNetworkAnalyzer:
    def __init__(self, skeleton, X, Y, creek_order, pts, order_max):
        self.skeleton = skeleton
        self.X = X
        self.Y = Y
        self.creek_order = creek_order
        self.pts = pts
        self.order_max = order_max
        self.corr_order = []
        self.corr_idx = []
        
    def swap_creek_orders(self):
        """Swap creek orders from Strahler to Reverse Strahler"""
        self.creek_order[self.creek_order == 0] = np.nan
        max_order = np.ones_like(self.creek_order) * (self.order_max + 1)
        max_order[np.isnan(self.creek_order)] = np.nan
        self.creek_order2 = max_order - self.creek_order
        
    def create_correction_gui(self):
        """Create GUI for creek order correction"""
        self.root = tk.Tk()
        self.root.title("Creek Order Correction")
        self.root.geometry("1200x800")
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create left and right plot frames
        left_frame = ttk.Frame(main_frame)
        right_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create figures
        self.fig1, self.ax1 = plt.subplots(figsize=(6, 6))
        self.fig2, self.ax2 = plt.subplots(figsize=(6, 6))
        
        # Create canvases
        canvas1 = FigureCanvasTkAgg(self.fig1, left_frame)
        canvas2 = FigureCanvasTkAgg(self.fig2, right_frame)
        canvas1.draw()
        canvas2.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Plot initial data
        self._plot_creek_orders()
        self._plot_skeleton()
        
        # Create control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create buttons and controls
        self.order_var = tk.StringVar(value="1")
        order_menu = ttk.Combobox(control_frame, textvariable=self.order_var,
                                values=[str(i) for i in range(1, 8)])
        order_menu.pack(side=tk.LEFT, padx=5)
        
        correct_btn = ttk.Button(control_frame, text="Correct Creek Segment",
                               command=self._correct_segment)
        correct_btn.pack(side=tk.LEFT, padx=5)
        
        finish_btn = ttk.Button(control_frame, text="Finish Correction",
                               command=self._finish_correction)
        finish_btn.pack(side=tk.LEFT, padx=5)
        
        self.root.mainloop()
        
    def _plot_creek_orders(self):
        """Plot creek orders"""
        self.ax1.clear()
        im = self.ax1.pcolor(self.X, self.Y, self.creek_order2, 
                            cmap='viridis', vmin=1, vmax=7)
        plt.colorbar(im, ax=self.ax1)
        self.ax1.set_title('Creek Orders')
        self.fig1.canvas.draw()
        
    def _plot_skeleton(self):
        """Plot skeleton with branch points"""
        self.ax2.clear()
        skeleton_pic = ndimage.binary_dilation(self.skeleton, iterations=2)
        skeleton_pic = skeleton_pic.astype(float)
        skeleton_pic[skeleton_pic == 0] = np.nan
        
        # Plot branch points
        y_ind, x_ind = np.where(self.pts)
        self.ax2.plot(x_ind, y_ind, '+r')
        
        self.ax2.imshow(skeleton_pic, cmap='gray')
        self.ax2.set_title('Creek Network')
        self.fig2.canvas.draw()
        
    def _correct_segment(self):
        """Handle creek segment correction"""
        # Get two points from user click
        points = plt.ginput(2)
        if not points or len(points) != 2:
            return
            
        x1, y1 = points[0]
        x2, y2 = points[1]
        
        # Find nearest branch points
        y_pts, x_pts = np.where(self.pts)
        pts_coords = np.column_stack((x_pts, y_pts))
        
        dist1 = np.sqrt(np.sum((pts_coords - np.array([x1, y1]))**2, axis=1))
        dist2 = np.sqrt(np.sum((pts_coords - np.array([x2, y2]))**2, axis=1))
        
        nearest1 = pts_coords[np.argmin(dist1)]
        nearest2 = pts_coords[np.argmin(dist2)]
        
        # Calculate shortest path
        self._calculate_shortest_path(nearest1, nearest2)
        
    def _calculate_shortest_path(self, pt1, pt2):
        """Calculate shortest path between two points"""
        # Create seeds
        seed1 = np.zeros_like(self.skeleton, dtype=bool)
        seed2 = np.zeros_like(self.skeleton, dtype=bool)
        
        seed1[int(pt1[1]), int(pt1[0])] = True
        seed2[int(pt2[1]), int(pt2[0])] = True
        
        # Calculate geodesic distance
        d1 = ndimage.distance_transform_edt(~seed1)
        d2 = ndimage.distance_transform_edt(~seed2)
        
        d = d1 + d2
        d[~self.skeleton] = np.inf
        
        # Find path
        path = d == ndimage.minimum_filter(d, size=3)
        path = thin(path)
        
        # Update creek orders
        order = int(self.order_var.get())
        self.creek_order2[path] = order
        self.corr_idx.append([pt1[0], pt1[1], pt2[0], pt2[1]])
        self.corr_order.append(order)
        
        self._plot_creek_orders()
        
    def _finish_correction(self):
        """Finish correction and close GUI"""
        self.root.quit()
        self.root.destroy()
        
    def process_corrected_segments(self):
        """Process corrected segments and calculate metrics"""
        # Implementation of Process_Correctedsegments function
        # This would include calculating Strahler numbers, distances, etc.
        pass

def main():
    # Example usage:
    skeleton = np.load('skeleton.npy')
    X = np.load('X.npy')
    Y = np.load('Y.npy')
    creek_order = np.load('creek_order.npy')
    pts = np.load('pts.npy')
    order_max = 6
    
    analyzer = CreekNetworkAnalyzer(skeleton, X, Y, creek_order, pts, order_max)
    analyzer.swap_creek_orders()
    analyzer.create_correction_gui()
    analyzer.process_corrected_segments()

if __name__ == "__main__":
    main()