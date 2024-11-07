import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
from scipy import ndimage
from skimage.morphology import skeletonize, thin, dilation, disk
from matplotlib.colors import ListedColormap

class CreekNetworkAnalyzer:
    def __init__(self, skeleton, X, Y, creek_order, pts, order_max):
        """Initialize with proper array conversion and validation"""
        self.skeleton = np.asarray(skeleton)
        self.pts = np.asarray(pts)
        
        # Get dimensions from skeleton
        self.rows, self.cols = self.skeleton.shape
        
        # Convert coordinates to indices (0-based)
        self.X = np.arange(self.cols)
        self.Y = np.arange(self.rows)
        self.X, self.Y = np.meshgrid(self.X, self.Y)
            
        # Handle creek order array
        if isinstance(creek_order, list):
            if isinstance(creek_order[0], list):
                self.creek_order = np.zeros((self.rows, self.cols), dtype=float)
                for i, row in enumerate(creek_order[:self.rows]):
                    self.creek_order[i, :len(row)] = row[:self.cols]
            else:
                self.creek_order = np.array(creek_order).reshape(self.rows, self.cols)
        else:
            self.creek_order = np.asarray(creek_order)
        
        self.order_max = int(order_max)
        self.corr_order = []
        self.corr_idx = []
        self.selected_points = []

    def create_correction_gui(self):
        """Create GUI for creek order correction"""
        self.root = tk.Tk()
        self.root.title("Creek Order Correction")
        self.root.geometry("1400x800")

        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title_label = ttk.Label(main_frame, text="Creek Order Corrections", 
                              font=('Arial', 14, 'bold'))
        title_label.pack(pady=5)

        # Create plots frame
        plots_frame = ttk.Frame(main_frame)
        plots_frame.pack(fill=tk.BOTH, expand=True)

        # Left plot
        left_frame = ttk.Frame(plots_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig1 = Figure(figsize=(7, 6))
        self.ax1 = self.fig1.add_subplot(111)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, left_frame)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas1, left_frame)

        # Add "Uncorrected Creek Orders" label
        uncorr_label = ttk.Label(left_frame, text="Uncorrected Creek Orders",
                               font=('Arial', 12, 'bold'))
        uncorr_label.pack()

        # Right plot
        right_frame = ttk.Frame(plots_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))

        self.fig2 = Figure(figsize=(7, 6))
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, right_frame)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas2, right_frame)

        # Add "Correction Window" label
        uncorr_label = ttk.Label(right_frame, text="Correction Window",
                               font=('Arial', 12, 'bold'))
        uncorr_label.pack()

        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))

        # Group the correction controls
        corr_frame = ttk.Frame(control_frame)
        corr_frame.pack(side=tk.RIGHT, padx=10)

        self.select_btn = ttk.Button(corr_frame, 
                                   text="Correct creek segment (2 points)",
                                   command=self.start_selection)
        self.select_btn.pack(side=tk.LEFT, padx=5)

        # Creek order selection
        order_label = ttk.Label(corr_frame, text="Creek Order")
        order_label.pack(side=tk.LEFT, padx=(10, 5))

        self.order_var = tk.StringVar(value="1")
        order_menu = ttk.Combobox(corr_frame, 
                                textvariable=self.order_var,
                                values=[str(i) for i in range(1, self.order_max + 2)],
                                width=5)
        order_menu.pack(side=tk.LEFT, padx=5)

        self.finish_btn = ttk.Button(corr_frame, 
                                   text="Finish correction",
                                   command=self.finish_correction)
        self.finish_btn.pack(side=tk.LEFT, padx=5)

        # Initialize plots
        self._plot_creek_orders()
        self._plot_skeleton()

        # Connect events
        self.canvas2.mpl_connect('button_press_event', self.on_click)
        
        self.root.protocol("WM_DELETE_WINDOW", self.finish_correction)
        self.root.mainloop()

    def _plot_creek_orders(self):
        """Plot creek orders"""
        self.ax1.clear()
        
        # Create discrete colormap: black + 5 colors
        # Define the 5 colors you want to use
        colors = ['#000000', '#451caf', '#1878ff', '#00c2ba', '#c9c200', '#f7fd00']
        # Create the discrete colormap
        discrete_cmap = ListedColormap(colors)
        
        # # Dilate the skeleton for better visibility
        # dilated_skeleton = dilation(self.skeleton, disk(2))
        
        # # Dilate the creek order mask for better visibility
        # dilated_creek_order = dilation(self.creek_order, disk(2))
        # masked_orders = np.ma.masked_where(~dilated_skeleton, dilated_creek_order)
        masked_orders = np.ma.masked_where(~self.skeleton, self.creek_order)

        # Set the aspect ratio to 1:1
        self.ax1.set_aspect('equal')
        
        im = self.ax1.pcolormesh(self.X, self.Y, masked_orders,
                                cmap=discrete_cmap, vmin=1, vmax=7)
        cbar = self.fig1.colorbar(im, ax=self.ax1, label='Creek order')
        cbar.set_ticks(np.arange(1.5, 7.5))  # Center ticks between colors
        cbar.set_ticklabels(np.arange(1, 7))  # Label with creek orders
        self.ax1.invert_yaxis()
        self.canvas1.draw()

    def _plot_skeleton(self):
        """Plot skeleton and branch points"""
        self.ax2.clear()
        
        # Dilate skeleton for better visibility
        dilated_skeleton = dilation(self.skeleton, disk(2))
        
        # Plot dilated skeleton in white on black background
        self.ax2.imshow(dilated_skeleton, cmap='gray')
        
        # Plot branch points
        y_pts, x_pts = np.where(self.pts)
        self.ax2.plot(x_pts, y_pts, 'r+', markersize=10)
        self.canvas2.draw()

    def on_click(self, event):
        """Handle click events on the skeleton plot"""
        if event.inaxes == self.ax2:
            self.selected_points.append((event.xdata, event.ydata))
            self.ax2.plot(event.xdata, event.ydata, 'ro')
            self.canvas2.draw()
            
            if len(self.selected_points) == 2:
                self._process_selected_segment(
                    self.selected_points[0], 
                    self.selected_points[1], 
                    int(self.order_var.get())
                )
                self.selected_points = []
                self._plot_creek_orders()
                self._plot_skeleton()

    def start_selection(self):
        """Start segment selection mode"""
        self.selected_points = []
        print("Click two points on the right plot to select a creek segment")

    def finish_correction(self):
        """Close GUI and finish correction"""
        self.root.quit()
        self.root.destroy()

    def _process_selected_segment(self, pt1, pt2, order):
        """Process the selected segment and update creek orders"""
        x1, y1 = int(round(pt1[0])), int(round(pt1[1]))
        x2, y2 = int(round(pt2[0])), int(round(pt2[1]))
        
        # Create seeds
        seed1 = np.zeros_like(self.skeleton, dtype=bool)
        seed2 = np.zeros_like(self.skeleton, dtype=bool)
        seed1[y1, x1] = True
        seed2[y2, x2] = True
        
        # Calculate shortest path
        d1 = ndimage.distance_transform_edt(~seed1)
        d2 = ndimage.distance_transform_edt(~seed2)
        d = d1 + d2
        d[~self.skeleton] = np.inf
        path = d == ndimage.minimum_filter(d, size=3)
        path = path & self.skeleton
        
        # Update creek order
        self.creek_order[path] = order
        self.corr_idx.append([x1, y1, x2, y2])
        self.corr_order.append(order)

    # def swap_creek_orders(self):
    #     """Swap creek orders from Strahler to Reverse Strahler"""
    #     self.creek_order[self.creek_order == 0] = np.nan
    #     max_order = np.full_like(self.creek_order, float(self.order_max + 1))
    #     max_order[np.isnan(self.creek_order)] = np.nan
    #     self.creek_order2 = max_order - self.creek_order