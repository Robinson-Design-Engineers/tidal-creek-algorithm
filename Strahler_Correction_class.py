# # for checking debugging kernel:
# import sys
# print(sys.executable)

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from matplotlib.figure import Figure # type: ignore
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk # type: ignore
import tkinter as tk
from tkinter import ttk
from scipy import ndimage # type: ignore
from skimage.morphology import skeletonize, thin, dilation, disk # type: ignore
from matplotlib.colors import ListedColormap # type: ignore
import networkx as nx # type: ignore

class CreekNetworkAnalyzer:

    def __init__(self, skeleton, X, Y, creek_order, creek_order_single, pts, order_max, STRAHLER, STRAIGHTDIST, IDXBRANCH, IDXSEG):
        """Initialize with proper array conversion and validation"""

        self.skeleton = np.asarray(skeleton)
        self.pts = np.asarray(pts)
        self.colorbar = None  # Store colorbar reference

        # load variables
        self.STRAHLER = STRAHLER
        self.STRAIGHTDIST = STRAIGHTDIST
        self.IDXBRANCH = IDXBRANCH
        self.IDXSEG = IDXSEG
        
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

        self.creek_order_single = np.asarray(creek_order_single)
        
        self.order_max = int(order_max)
        self.corr_order = []
        self.corr_idx = []
        self.selected_points = []

        self.selection_mode = False  # Track whether we're in selection mode

    def swap_creek_orders(self):
        # 6.1 in MATLAB CHIROL_CREEK_ALGORITHM_2024.m -SamK
        """Swap creek orders from Strahler to Reverse Strahler"""
        self.creek_order[self.creek_order == 0] = np.nan
        max_order = np.full_like(self.creek_order, float(self.order_max + 1))
        max_order = max_order.astype(float)  # Ensure max_order is float
        max_order[np.isnan(self.creek_order)] = np.nan
        self.creek_order_swapped = max_order - self.creek_order

        max_order = np.full_like(self.creek_order_single, float(self.order_max + 1))
        max_order = max_order.astype(float)  # Ensure max_order is float
        max_order[np.isnan(self.creek_order_single)] = np.nan
        self.creek_order_single_swapped = max_order - self.creek_order_single

    def bwmorph_thicken(self, image, iterations):
        """Thicken the image by n iterations"""
        result = image.copy()
        for _ in range(iterations):
            result = dilation(result, disk(1))
        return result

    def create_correction_gui(self):
        # 6.2 in MATLAB CHIROL_CREEK_ALGORITHM_2024.m -SamK
        """Create GUI for creek order correction"""
        self.root = tk.Tk()
        self.root.title("Creek Order Correction")
        self.root.geometry("1400x800")

        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # # Title
        # title_label = ttk.Label(main_frame, text="Creek Order Corrections", 
        #                       font=('Arial', 14, 'bold'))
        # title_label.pack(pady=5)

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
        # Store current axis limits before clearing
        xlim = self.ax1.get_xlim() if self.ax1.get_xlim() != (0, 1) else None
        ylim = self.ax1.get_ylim() if self.ax1.get_ylim() != (0, 1) else None
        
        if self.colorbar is not None:
            self.colorbar.remove()
        self.ax1.clear()
        
        # Create discrete colormap: black + 5 colors
        colors = ['#000000', '#451caf', '#1878ff', '#00c2ba', '#c9c200', '#f7fd00']
        discrete_cmap = ListedColormap(colors)
        
        # Dilate the skeleton for better visibility
        dilated_skeleton = dilation(self.skeleton, disk(1))
        masked_orders = np.ma.masked_where(~dilated_skeleton, self.creek_order_swapped)

        # Set the aspect ratio to 1:1
        self.ax1.set_aspect('equal')
        
        # Adjust coordinates for pcolormesh
        # We need coordinates to be cell edges, so add one more point
        x_edges = np.arange(masked_orders.shape[1] + 1)
        y_edges = np.arange(masked_orders.shape[0] + 1)
        X_edges, Y_edges = np.meshgrid(x_edges, y_edges)
        
        im = self.ax1.pcolormesh(X_edges, Y_edges, masked_orders,
                                cmap=discrete_cmap, 
                                vmin=1, vmax=7,
                                shading='flat')
        
        self.colorbar = self.fig1.colorbar(im, ax=self.ax1, label='Creek order')
        self.colorbar.set_ticks(np.arange(1.5, 7.5))
        self.colorbar.set_ticklabels(np.arange(1, 7))
        
        self.ax1.invert_yaxis()

        # Restore previous zoom if it exists
        if xlim is not None and ylim is not None:
            self.ax1.set_xlim(xlim)
            self.ax1.set_ylim(ylim)
        
        self.canvas1.draw()

    def _plot_skeleton(self):
        """Plot skeleton and branch points"""
        # Store current axis limits before clearing
        xlim = self.ax2.get_xlim() if self.ax2.get_xlim() != (0, 1) else None
        ylim = self.ax2.get_ylim() if self.ax2.get_ylim() != (0, 1) else None
        
        self.ax2.clear()
        
        # Plot dilated skeleton in white on black background
        dilated_skeleton = dilation(self.skeleton, disk(1))
        self.ax2.imshow(dilated_skeleton, cmap='gray')
        
        # Plot branch points
        y_pts, x_pts = np.where(self.pts)
        self.ax2.plot(x_pts, y_pts, 'r+', alpha=0.7, markersize=10)
        
        # Restore previous zoom if it exists
        if xlim is not None and ylim is not None:
            self.ax2.set_xlim(xlim)
            self.ax2.set_ylim(ylim)
        
        self.canvas2.draw()

    # starts 6.3 in MATLAB CHIROL_CREEK_ALGORITHM_2024.m -SamK
    def on_click(self, event):
        """Handle click events on the skeleton plot and snap to nearest branch/endpoint"""
        if event.inaxes == self.ax2 and self.selection_mode:
            # Get clicked coordinates
            clicked_x, clicked_y = event.xdata, event.ydata
            
            # Find all branch/endpoint coordinates
            y_pts, x_pts = np.where(self.pts)
            points = np.column_stack((x_pts, y_pts))
            
            # Calculate distances to all branch/endpoints
            distances = np.sqrt((points[:, 0] - clicked_x)**2 + 
                            (points[:, 1] - clicked_y)**2)
            
            # Find nearest point
            nearest_idx = np.argmin(distances)
            snapped_x = points[nearest_idx, 0]
            snapped_y = points[nearest_idx, 1]
            
            # Store snapped point and its plot handles
            self.selected_points.append((snapped_x, snapped_y))
            
            # Clear previous temporary click point if it exists
            if hasattr(self, 'temp_click'):
                self.temp_click.remove()
            
            # Plot clicked point (temporary)
            self.temp_click = self.ax2.plot(clicked_x, clicked_y, 'ro', alpha=0.3)[0]
            
            # Plot snapped point
            self.ax2.plot(snapped_x, snapped_y, 'go')
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
                self.selection_mode = False  # Disable selection mode after processing

    def start_selection(self):
        """Start segment selection mode"""
        self.selection_mode = True
        self.selected_points = []
        # Clear any existing temporary click point
        if hasattr(self, 'temp_click'):
            self.temp_click.remove()
            delattr(self, 'temp_click')
        self._plot_skeleton()  # Refresh the plot
        print("Click two points on the right plot to select a creek segment")

    def finish_correction(self):
        """Close GUI and finish correction"""
        self.selection_mode = False  # Disable selection mode
        self.root.quit()
        self.root.destroy()

    def _process_selected_segment(self, pt1, pt2, order):
        """Process the selected segment and pythupdate creek orders using network-based path finding"""
        import networkx as nx # type: ignore
        
        x1, y1 = int(round(pt1[0])), int(round(pt1[1]))
        x2, y2 = int(round(pt2[0])), int(round(pt2[1]))
        
        # Create graph from skeleton
        G = nx.Graph()
        y_coords, x_coords = np.where(self.skeleton)
        points = list(zip(y_coords, x_coords))
        
        # Add nodes for all skeleton points
        for p in points:
            G.add_node(p)
        
        # Add edges between neighboring points
        for y, x in points:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    num_y, num_x = y + dy, x + dx
                    if (num_y, num_x) in G.nodes():
                        # Use diagonal distance for weights
                        weight = np.sqrt(2) if abs(dx) == 1 and abs(dy) == 1 else 1
                        G.add_edge((y, x), (num_y, num_x), weight=weight)
        
        # Find shortest path between selected points
        try:
            path_nodes = nx.shortest_path(G, (y1, x1), (y2, x2), weight='weight')
            
            # Create path mask
            path = np.zeros_like(self.skeleton, dtype=bool)
            for y, x in path_nodes:
                path[y, x] = True
                
            # Create dilated path
            dilated_path = dilation(path, disk(1))
            
            # Update creek order with dilated path
            self.creek_order_single_swapped[path] = order
            self.creek_order_swapped[dilated_path] = order
            self.corr_idx.append([x1, y1, x2, y2])
            self.corr_order.append(order)

        except nx.NetworkXNoPath:
            print("No valid path found between selected points")
        except Exception as e:
            print(f"Error finding path: {e}")
    # ends 6.3 in MATLAB CHIROL_CREEK_ALGORITHM_2024.m -SamK

    def process_corrected_segments(self):
        """Process segments that were corrected through the GUI interface."""
        
        # # Initialize arrays if they don't exist - I don't think this should be needed -Sam
        # self.STRAHLER = np.zeros((self.order_max + 1, 100))  # Assuming max 100 segments per order
        # self.STRAIGHTDIST = np.zeros((self.order_max + 1, 100))
        # self.IDXBRANCH = np.zeros((self.order_max + 1, 100))
        # self.IDXSEG = np.zeros((self.order_max + 1, 600))  # 6 coordinates per segment, 100 segments
        
        # Process each corrected segment
        for k in range(len(self.corr_order)):
            # Select correct position in the Strahler table
            order = self.corr_order[k]
            MAX_ORDER = self.order_max + 1
            order = MAX_ORDER - order
            
            # Find first empty column in row
            test = np.sum(self.STRAHLER[order, :])
            if test != 0:
                colind = np.max(np.where(self.STRAHLER[order, :] != 0)[0]) + 1
            else:
                colind = 0
                
            # Find branch points
            B = skeletonize(self.skeleton)
            E = skeletonize(self.skeleton)
            PTS = B + E
            
            # Get coordinates of corrected segment
            CORRIDX_k = self.corr_idx[k]
            xind1, yind1 = CORRIDX_k[0], CORRIDX_k[1]
            xind2, yind2 = CORRIDX_k[2], CORRIDX_k[3]
            
            # Convert to linear indices
            corridx1 = yind1 * self.skeleton.shape[1] + xind1
            corridx2 = yind2 * self.skeleton.shape[1] + xind2
            
            # Create seeds for geodesic distance calculation
            SEED1 = np.zeros_like(self.skeleton, dtype=bool)
            SEED1.flat[corridx1] = True
            SEED1 = self.bwmorph_thicken(SEED1, 2)
            
            SEED2 = np.zeros_like(self.skeleton, dtype=bool)
            SEED2.flat[corridx2] = True
            SEED2 = self.bwmorph_thicken(SEED2, 2)
            
            # Calculate geodesic distances
            D1 = ndimage.distance_transform_edt(~SEED1)
            D2 = ndimage.distance_transform_edt(~SEED2)
            D = D1 + D2
            D = np.round(D * 32) / 32
            
            # Handle infinities and find minimum distance
            D[D == np.inf] = np.nan
            Dmax = np.nanmin(D)
            
            if not np.isinf(Dmax):
                self.STRAHLER[order, colind] = Dmax
                
                # Calculate straight-line distance
                skeletoneucl = np.zeros_like(self.skeleton)
                skeletoneucl[yind1, xind1] = 1
                euclD = ndimage.distance_transform_edt(~skeletoneucl)
                disttopt = euclD[yind2, xind2]
                
                self.STRAIGHTDIST[order, colind] = disttopt
                self.IDXBRANCH[order, colind] = corridx2
                
                # Create mask for points between selected coordinates
                D[np.isnan(D)] = np.inf
                Dmask = ndimage.minimum_filter(D, size=3) == D
                Dmask = Dmask & self.skeleton
                
                # Find coordinates on Dmask closest to selected points
                YDmask, XDmask = np.where(Dmask)
                distances = np.sqrt((XDmask - xind1)**2 + (YDmask - yind1)**2)
                closest_idx = np.argmin(distances)
                y1, x1 = YDmask[closest_idx], XDmask[closest_idx]
                
                distances = np.sqrt((XDmask - xind2)**2 + (YDmask - yind2)**2)
                closest_idx = np.argmin(distances)
                y2, x2 = YDmask[closest_idx], XDmask[closest_idx]
                
                # Calculate distances along segment
                Dtemp = ndimage.distance_transform_edt(~Dmask)
                Dtemp[~Dmask] = np.nan
                maxDtemp = np.nanmax(Dtemp)
                meanDtemp = np.round(np.nanmean(Dtemp))
                
                # Find midpoint
                closest_val = np.nanmin(np.abs(Dtemp - meanDtemp))
                ym, xm = np.where(np.abs(Dtemp - meanDtemp) == closest_val)
                xm, ym = xm[0], ym[0]
                
                # Calculate points at 20% distance from midpoint
                dist = max(1, round(0.2 * maxDtemp))
                
                # Create transformation matrices
                # Translation
                T1 = np.array([
                    [1, 0, xm],
                    [0, 1, ym],
                    [0, 0, 1]
                ])
                
                # Rotation (90 degrees)
                theta = np.pi/2
                T_rot = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
                
                # Scale
                M = np.array([
                    [2, 0, 0],
                    [0, 2, 0],
                    [0, 0, 1]
                ])
                
                # Apply transformations
                def transform_point(x, y, T1, T_rot, M):
                    p = np.array([x, y, 1])
                    T1_inv = np.linalg.inv(T1)
                    p_new = T1 @ T_rot @ M @ T1_inv @ p
                    return p_new[0], p_new[1]
                
                x1new, y1new = transform_point(x1, y1, T1, T_rot, M)
                x2new, y2new = transform_point(x2, y2, T1, T_rot, M)
                
                # Update final coordinates
                x2, y2 = x1new, y1new
                x3, y3 = x2new, y2new
                x1, y1 = xm, ym
                
                # Store segment coordinates
                idx_seg_row = self.IDXSEG[order, :]
                non_zero_count = np.count_nonzero(idx_seg_row)
                colstart = non_zero_count
                self.IDXSEG[order, colstart:colstart+6] = [y1, x1, y2, x2, y3, x3]
                
                # Update creek order visualization
                creek_order_mask = skeletonize(Dmask)
                self.creek_order_single_swapped[creek_order_mask != 0] = creek_order_mask[creek_order_mask != 0] * order
                creek_order_mask = self.bwmorph_thicken(creek_order_mask, 3) # changed (creek_order_mask, 3)
                self.creek_order_swapped[creek_order_mask != 0] = creek_order_mask[creek_order_mask != 0] * order
