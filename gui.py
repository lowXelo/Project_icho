"""
High-level Image Analysis (HIA) Application

This application provides tools for astronomical image registration and analysis.

Main Features:
- Image registration using optical flow methods
- Thumbnail processing and alignment
- Panchromatic image generation and registration
- Simple deformation analysis
- Visualization of registration results
- FITS file handling
- Configurable appearance with coffee themes

The application uses CustomTkinter for the GUI and various scientific libraries
(numpy, scipy, astropy, skimage) for image processing and analysis.

Author: Project ICHO Team
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox,filedialog
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from skimage.registration import optical_flow_ilk
from skimage.transform import warp
from astropy.io import fits
import scipy.signal
import time
from skimage.transform import AffineTransform, ProjectiveTransform
import os
from datetime import datetime

# Initialize the app
ctk.set_appearance_mode("dark")  # Options: "System", "Light", "Dark"
ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"
app = ctk.CTk()  
app.title("HIA")

# Global variables for theme settings
current_appearance = "dark"
current_color_theme = "blue"

# --- STRUCTURE EN DEUX COLONNES ---
app.columnconfigure(0, weight=1)  # Colonne gauche (zone affichage)
app.columnconfigure(1, weight=2)  # Colonne droite (contenu principal)
app.rowconfigure(0, weight=1)  # Main content row
app.rowconfigure(1, weight=0)  # Progress bar row

# Global variables
fichiers_selectionnes = []
states = []
pipline_result = {}
corners = []
global_progress = 0.0
plot_figures = []  # Store figure references to prevent garbage collection

# Global progress bar at the bottom of the app
progress_frame = ctk.CTkFrame(app)
progress_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
progress_label = ctk.CTkLabel(progress_frame, text="Processing: 0%")
progress_label.pack(side=tk.LEFT, padx=5)
global_progress_bar = ctk.CTkProgressBar(progress_frame)
global_progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
global_progress_bar.set(0)

def generate_feature_report():
    """Generate a comprehensive report of all application features."""
    report = """
HIGH LEVEL IMAGE ANALYSIS (HIA) - FEATURE REPORT

GENERAL FEATURES:
-----------------
• File Management: Open and save FITS files and NumPy arrays
• Coffee-themed UI: Customizable appearance with coffee-themed color schemes
• Progress tracking: Global progress bar for monitoring processing tasks
• Error handling: Comprehensive error reporting in the text zone

IMAGE REGISTRATION METHODS:
--------------------------
1. Parameter Search (V1 Search):
   • Optical flow-based registration with parameter optimization
   • Box plot, mean, variance, and processing time analysis
   • Visualization of parameter effects on registration quality
   • Deformation map visualization

2. Direct Calculation (V1 Calculate):
   • Efficient registration with specified parameters
   • Visualization of registration results
   • Correlation coefficient comparison before and after registration
   • Image browser for examining original and registered thumbnails

3. Simple Deformation:
   • Manual geometric transformation definition
   • Visual manipulation of deformation parameters
   • Corner-based transformation specification
   • Comparison of translation, rigid body, scaled rotation, affine, and bilinear transformations

4. Panchromatic Registration:
   • Processing of multiple FITS files into panchromatic images
   • Registration of panchromatic images
   • Direct processing of existing NumPy panchromatic arrays
   • Deformation map visualization in single or all modes

DATA VISUALIZATION:
-----------------
• Statistics view: Box plots and correlation coefficients
• Image browser: Navigation through thumbnail sequences
• Deformation maps: Vector field visualization
• Side-by-side comparison of original and registered images
• Save options for results and processed images

OPTIMIZATION FEATURES:
--------------------
• Selective processing based on user-selected options
• Efficient memory usage by only storing required data
• Improved UI responsiveness during processing
• Better error handling and status updates
• Modular code organization for maintainability

"""
    
    # Create a new window to display the report
    report_window = ctk.CTkToplevel(app)
    report_window.title("HIA Feature Report")
    report_window.geometry("800x600")
    
    # Create a frame for the report
    report_frame = ctk.CTkFrame(report_window)
    report_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Add a scrollable text widget
    report_text = ctk.CTkTextbox(report_frame, wrap="word", font=("Courier", 12))
    report_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Insert the report text
    report_text.insert(tk.END, report)
    
    # Make it read-only
    report_text.configure(state="disabled")
    
    # Add a close button
    close_button = ctk.CTkButton(
        report_frame, 
        text="Close", 
        command=report_window.destroy,
        width=100,
        height=30
    )
    close_button.pack(pady=10)
    
    # Bring window to front
    report_window.lift()
    report_window.focus_set()

def update_progress_bar(value):
    """Update the global progress bar with the given value (0-100)."""
    global global_progress
    global_progress = value
    global_progress_bar.set(value / 100)
    progress_label.configure(text=f"Processing: {int(value)}%")
    app.update_idletasks()  # Force update the UI

# Functions for theme customization
def change_appearance_mode(mode):
    global current_appearance
    current_appearance = mode
    ctk.set_appearance_mode(mode)

def change_color_theme(theme):
    """Change the color theme of the application with coffee-themed colors."""
    global current_color_theme
    current_color_theme = theme
    
    # Define coffee-themed colors for different elements
    if theme == "espresso":
        # Dark coffee theme
        fg_color = "#2C2C2C"  # Very dark brown, almost black
        hover_color = "#483C32"  # Dark taupe
        text_color = "#E6DDCF"  # Light cream
        button_color = "#654321"  # Dark brown
    elif theme == "latte":
        # Light coffee theme
        fg_color = "#E6DDCF"  # Light cream
        hover_color = "#D2B48C"  # Tan
        text_color = "#4A3C2C"  # Dark brown
        button_color = "#C19A6B"  # Medium brown
    elif theme == "cappuccino":
        # Medium coffee theme
        fg_color = "#C19A6B"  # Medium brown
        hover_color = "#A67B5B"  # Caramel
        text_color = "#FFF8E7"  # Off-white
        button_color = "#8B5A2B"  # Darker brown
    elif theme == "mocha":
        # Dark brown coffee theme
        fg_color = "#483C32"  # Dark taupe
        hover_color = "#654321"  # Dark brown
        text_color = "#FFF8DC"  # Cornsilk
        button_color = "#5C4033"  # Medium-dark brown
    else:
        # Default theme (fall back to blue)
        ctk.set_default_color_theme("blue")
        messagebox.showinfo("Theme Changed", f"Color theme changed to {theme}. This will be fully applied when you restart the application.")
        return
        
    # Apply the theme to existing buttons and elements
    try:
        # Try to apply theme to existing elements
        for widget in app.winfo_children():
            if isinstance(widget, ctk.CTkFrame):
                for child in widget.winfo_children():
                    if isinstance(child, ctk.CTkButton):
                        child.configure(
                            fg_color=button_color, 
                            hover_color=hover_color, 
                            text_color=text_color
                        )
                    elif isinstance(child, ctk.CTkLabel):
                        child.configure(text_color=text_color)
        
        # Inform the user
        messagebox.showinfo("Theme Changed", f"Coffee theme changed to {theme}. Some elements will update immediately, full effect on restart.")
    except Exception as e:
        print(f"Error applying theme: {e}")
        
    # This will be fully applied on restart
    ctk.set_default_color_theme("blue")  # We still use the blue base theme, but override colors

# Function to save image as FITS
def save_image_as_fits(image_data, filename=None):
    """
    Save an image as a FITS file with metadata.
    
    Args:
        image_data: 2D numpy array of image data
        filename: Optional filename to save to. If None, a file dialog will be shown.
        
    Returns:
        Boolean indicating success or failure
    """
    if image_data is None:
        messagebox.showerror("Error", "No image data to save")
        return False
        
    # Show file dialog if no filename specified
    if filename is None:
        filename = filedialog.asksaveasfilename(
            defaultextension=".fits",
            filetypes=[("FITS files", "*.fits"), ("All files", "*.*")]
        )
        if not filename:  # User canceled the dialog
            return False
    
    try:
        # Create a new FITS file with metadata
        hdu = fits.PrimaryHDU(image_data)
        
        # Add useful metadata
        hdu.header['DATE'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        hdu.header['CREATOR'] = 'HIA Application'
        hdu.header['SOFTWARE'] = 'Project ICHO'
        hdu.header['DATATYPE'] = str(image_data.dtype)
        hdu.header['DATAMIN'] = np.min(image_data)
        hdu.header['DATAMAX'] = np.max(image_data)
        hdu.header['DATAMEAN'] = np.mean(image_data)
        hdu.header['DATASTD'] = np.std(image_data)
        
        # Write the file
        hdu.writeto(filename, overwrite=True)
        
        # Show success message
        text_zone.config(state="normal")
        text_zone.insert(tk.END, f"Image saved successfully to: {filename}\n")
        text_zone.config(state="disabled")
        
        return True
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save FITS file: {str(e)}")
        return False

# --- CORE FUNCTIONS ---
def mutual_info_matrix(data):
    """Compute mutual information for multi-dimensional data.
    Simple implementation for mutual information between columns"""
    from sklearn.metrics import mutual_info_score
    n_dim = data.shape[1]
    mi_matrix = np.zeros((n_dim, n_dim))
    
    for i in range(n_dim):
        for j in range(n_dim):
            if i == j:
                mi_matrix[i, j] = 1.0
            else:
                mi_matrix[i, j] = mutual_info_score(data[:, i], data[:, j])
    
    return mi_matrix

def find_outliers_2D(data, pclow=25, pchigh=75, thresh=1.5):
    """
    Find outlier pixels in 2D data using interquartile range (IQR) method.
    
    Args:
        data: 2D numpy array of pixel values
        pclow: Lower percentile threshold (default: 25)
        pchigh: Upper percentile threshold (default: 75)
        thresh: Outlier threshold multiplier for IQR (default: 1.5)
        
    Returns:
        Tuple of (outliers, non_outliers) where each is a numpy array of coordinates
    """
    # Calculate percentiles and interquartile range
    q1 = np.percentile(data.flatten(), pclow)
    q3 = np.percentile(data.flatten(), pchigh)
    iqr = q3 - q1
    
    # Calculate threshold boundaries
    lower_fence = q1 - thresh * iqr
    upper_fence = q3 + thresh * iqr
    
    # Create boolean masks for outliers and non-outliers
    outlier_mask = (data < lower_fence) | (data > upper_fence)
    non_outlier_mask = ~outlier_mask
    
    # Find coordinates of outliers and non-outliers
    outliers = np.argwhere(outlier_mask)
    non_outliers = np.argwhere(non_outlier_mask)
    
    return outliers, non_outliers

def load_fits(file_path, Nthumb=80):
    """
    Load a FITS file and preprocess it to remove outliers.
    
    Args:
        file_path: Path to the FITS file
        Nthumb: Number of thumbnails to process (default: 80)
        
    Returns:
        Preprocessed image data with outliers removed
    """
    try:
        # Load FITS data
        with fits.open(file_path) as hdul:
            data = hdul[0].data.astype(float)
        
        # Transpose to get proper orientation for processing
        unregim = np.transpose(data, (1, 2, 0))
        
        # Process each thumbnail to remove outliers
        for tt in range(Nthumb):
            # Apply median filter to get reference values for outlier replacement
            medfi = scipy.signal.medfilt2d(unregim[:, :, tt])
            
            # Find outliers
            outl, _ = find_outliers_2D(unregim[:, :, tt], pclow=10, pchigh=90)
            
            # Replace outliers with median-filtered values
            if len(outl) > 0:  # Only process if outliers exist
                unregim[outl[:, 0], outl[:, 1], tt] = medfi[outl[:, 0], outl[:, 1]]
        
        return unregim
    
    except Exception as e:
        # Log and re-raise the exception for proper error handling
        print(f"Error loading FITS file {file_path}: {e}")
        raise

def apply_ilk(unregim, radius, num_warp, mycref=33, Nthumb=80, defmap_mode="none", defmap_index=0):
    """Apply optical flow ILK registration to a set of thumbnails."""
    im_ref = unregim[:, :, mycref]
    imwarped_array = []
    cc1_array, cc2_array = [], []
    uv_array = []
    
    for k in range(0, Nthumb):
        image = unregim[:, :, k]
        v, u = optical_flow_ilk(im_ref, image,
                              radius=radius,
                              num_warp=num_warp,
                              prefilter=True)
        nr, nc = im_ref.shape
        row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
        image_warped = warp(image, np.array([row_coords + v, col_coords + u]), mode='constant')
        
        if defmap_mode != "none":
            uv_array.append([u, v])

        imwarped_array.append(image_warped)

        cc1 = np.corrcoef(image.flatten(), im_ref.flatten())[0, 1]
        cc2 = np.corrcoef(image_warped.flatten(), im_ref.flatten())[0, 1]

        if k != mycref:
            cc1_array.append(cc1)
            cc2_array.append(cc2)
    
    cc1_mean = np.mean(np.array(cc1_array))
    cc2_mean = np.mean(np.array(cc2_array))
    diff_cc = cc2_mean - cc1_mean
    
    # Display results in text zone
    text_zone.config(state="normal")
    text_zone.insert(tk.END, f"\nRegistration results for {Nthumb} thumbnails:\n")
    text_zone.insert(tk.END, f"cc1_mean = {cc1_mean:.4f} || cc2_mean = {cc2_mean:.4f} || delta = {diff_cc:.4f}\n")
    text_zone.config(state="disabled")
    

    
    imreg_array = np.stack(imwarped_array, axis=0)
    return imreg_array

def process_fits_to_panchromatic(file_paths, radius, num_warp, mycref=33, Nthumb=80, defmap_mode="none", defmap_index=0):
    """Process multiple FITS files into panchromatic images."""
    panchromatique_array = []
    total_files = len(file_paths)
    
    for idx, path in enumerate(file_paths):
        # Update progress - loading file
        update_progress_bar((idx * 100 / total_files) / 2)  # First half for loading/processing
        
        # Load and register thumbnails
        unregim = load_fits(path, Nthumb)
        imreg_array = apply_ilk(unregim, radius, num_warp, mycref, Nthumb, defmap_mode, defmap_index)
        
        # Create panchromatic image by averaging
        im_panchromatique = np.mean(imreg_array, axis=0)
        panchromatique_array.append(im_panchromatique)
        
        # Update progress - second half for processing
        update_progress_bar(50 + (idx * 100 / total_files) / 2)
    
    # Convert to numpy array
    panchromatique_array = np.array(panchromatique_array)
    
    return panchromatique_array

def panchromatic_registration(panchro_array, radius, num_warp, ref=0, defmap_mode="none", defmap_index=0):
    """Register panchromatic images."""
    im_ref = panchro_array[ref]
    l = panchro_array.shape
    print(l)
    registered_images = []
    uv_array = []
    
    for k in range(l[0]):
        if k == ref:
            registered_images.append(panchro_array[k])
        else:
            image = panchro_array[k]
            cc1 = np.corrcoef(image.flatten(), im_ref.flatten())[0, 1]
            
            v, u = optical_flow_ilk(im_ref, image, radius=radius, num_warp=num_warp, prefilter=True)
            nr, nc = im_ref.shape
            row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
            image_warped = warp(image, np.array([row_coords + v, col_coords + u]), mode='constant')
            
            if defmap_mode != "none":
                uv_array.append([u, v])
            
            cc2 = np.corrcoef(image_warped.flatten(), im_ref.flatten())[0, 1]
            diff_cc = cc2 - cc1
            
            text_zone.config(state="normal")
            text_zone.insert(tk.END, f"\nPanchromatic image {k}\n")
            text_zone.insert(tk.END, f"CC1 = {cc1:.4f} (REF)|| CC2 = {cc2:.4f} || delta = {100*diff_cc:.2f} %\n")
            text_zone.config(state="disabled")
            
            registered_images.append(image_warped)
    
    if defmap_mode != "none":
        print("c")
        show_deformation_map_panchro(uv_array, im_ref, defmap_mode, defmap_index)
    
    return np.array(registered_images)




def show_deformation_map_ilk(uv_array, im_ref, defmap_mode, defmap_index):
    """Show the deformation map computed by optical_flow_ilk for ILK thumbnails registration."""
    print("a")
    try:
        if defmap_mode == "single":
            u, v = uv_array[defmap_index-1][0], uv_array[defmap_index-1][1]

            norm = np.sqrt(u**2 + v**2)

            nvec = 20  # Number of vectors to be displayed along each image dimension
            nl, nc = im_ref.shape
            step = max(nl // nvec, nc // nvec)

            y, x = np.mgrid[:nl:step, :nc:step]
            u_ = u[::step, ::step]
            v_ = v[::step, ::step]

            # Create embedded plot
            fig, canvas, frame = create_embedded_plot(title=f"Deformation Map - Thumbnail {defmap_index}")
            
            ax = fig.add_subplot(111)
            im = ax.imshow(norm)
            ax.quiver(x, y, u_, v_, color='r', units='dots', angles='xy', scale_units='xy', lw=3)
            ax.set_title(f"Thumbnail {defmap_index}")
            fig.colorbar(im)
            fig.tight_layout()
            canvas.draw()
            
        elif defmap_mode == "all":
            # Display an error message that 'all' mode is not supported for ILK thumbnails
            text_zone.config(state="normal")
            text_zone.insert(tk.END, "\nError: 'all' mode is not supported for thumbnails registration. Please use 'single' mode.\n")
            text_zone.config(state="disabled")
        elif defmap_mode != "none":
            # Generic error for unsupported modes
            raise ValueError(f"Invalid defmap_mode: '{defmap_mode}'. \nExpected 'single' or 'none'.")
        
    except ValueError as e:
        text_zone.config(state="normal")
        text_zone.insert(tk.END, f"\nError showing deformation map: {e}\n")
        text_zone.config(state="disabled")
    except IndexError as e:
        text_zone.config(state="normal")
        text_zone.insert(tk.END, f"\nError: Invalid index {defmap_index}. Make sure it's within the range of available thumbnails.\n")
        text_zone.config(state="disabled")

def show_deformation_map_panchro(uv_array, im_ref, defmap_mode, defmap_index):
    """Show the deformation map computed by optical_flow_ilk for panchromatic images."""
    print("b")
    try:
        if defmap_mode == "all":
            # Create a figure with multiple subplots for all deformation maps
            if len(uv_array) == 0:
                raise ValueError("No deformation vectors available to display")
                
            rows = min(3, (len(uv_array) + 2) // 3)  # Calculate needed rows
            cols = min(3, len(uv_array))  # Use up to 3 columns
            
            # Create embedded plot
            fig, canvas, frame = create_embedded_plot(title="All Deformation Maps")
            
            # Set up grid for subplots
            grid = plt.GridSpec(rows, cols, figure=fig)
            
            for i in range(min(len(uv_array), rows * cols)):
                u, v = uv_array[i][0], uv_array[i][1]
                norm = np.sqrt(u**2 + v**2)

                nvec = 20  # Number of vectors to be displayed
                nl, nc = im_ref.shape
                step = max(nl // nvec, nc // nvec)

                y, x = np.mgrid[:nl:step, :nc:step]
                u_ = u[::step, ::step]
                v_ = v[::step, ::step]

                row = i // cols
                col = i % cols

                ax = fig.add_subplot(grid[row, col])
                im = ax.imshow(norm)
                ax.quiver(x, y, u_, v_, color='r', units='dots', angles='xy', scale_units='xy', lw=3)
                ax.set_title(f"Panchromatic {i+1}")
                ax.set_axis_off()
                fig.colorbar(im, ax=ax)
            
            fig.tight_layout()
            canvas.draw()
            
        elif defmap_mode == "single":
            if len(uv_array) == 0:
                raise ValueError("No deformation vectors available to display")
                
            if defmap_index <= 0 or defmap_index > len(uv_array):
                raise IndexError(f"Index {defmap_index} is out of range (valid range: 1-{len(uv_array)})")
                
            u, v = uv_array[defmap_index-1][0], uv_array[defmap_index-1][1]
            norm = np.sqrt(u**2 + v**2)

            nvec = 20
            nl, nc = im_ref.shape
            step = max(nl // nvec, nc // nvec)

            y, x = np.mgrid[:nl:step, :nc:step]
            u_ = u[::step, ::step]
            v_ = v[::step, ::step]

            # Create embedded plot
            fig, canvas, frame = create_embedded_plot(title=f"Deformation Map - Panchromatic {defmap_index}")
            
            ax = fig.add_subplot(111)
            im = ax.imshow(norm)
            ax.quiver(x, y, u_, v_, color='r', units='dots', angles='xy', scale_units='xy', lw=3)
            ax.set_title(f"Panchromatic {defmap_index}")
            fig.colorbar(im)
            fig.tight_layout()
            canvas.draw()
            
        else:
            raise ValueError(f"Invalid defmap_mode: '{defmap_mode}'. Expected 'all', 'single', or 'none'.")
            
    except ValueError as e:
        text_zone.config(state="normal")
        text_zone.insert(tk.END, f"\nError showing deformation map: {e}\n")
        text_zone.config(state="disabled")
    except IndexError as e:
        text_zone.config(state="normal")
        text_zone.insert(tk.END, f"\nError: {e}\n")
        text_zone.config(state="disabled")
    except Exception as e:
        text_zone.config(state="normal")
        text_zone.insert(tk.END, f"\nUnexpected error showing deformation map: {e}\n")
        text_zone.config(state="disabled")

def v1_pipelin_youness(les_options, file_unregistered, nump_warp_value, radius_value, defmap_mode="none", defmap_index=0):
    global global_progress
    mycref = 33
    Nthumb = 80
    returned_data = {}
    uv_array = []
    total_files = len(file_unregistered)

    for file_index, i in enumerate(file_unregistered, start=1):  
        file_key = f"file {file_index}"
        returned_data[file_key] = {}
        
        # Update progress for file loading
        update_progress_bar((file_index - 1) * 100 / total_files)

        hdul = fits.open(i)
        unregim = hdul[0].data.astype(float)
        unregim = np.transpose(unregim, (1, 2, 0))

        for tt in np.arange(Nthumb):
            medfi = scipy.signal.medfilt2d(unregim[:, :, tt])
            outl, noout = find_outliers_2D(unregim[:, :, tt], pclow=10, pchigh=90)
            for oo in np.arange(len(outl)):
                unregim[outl[oo][0], outl[oo][1], tt] = medfi[outl[oo][0], outl[oo][1]]

        im_ref = unregim[:, :, mycref]
        cc1_array = []
        cc2_array = []
        RADIUS = range(1, radius_value)
        NUMP_WARP = range(1, nump_warp_value)
        time_exe = []
        total_iterations = len(RADIUS) * len(NUMP_WARP)
        current_iteration = 0

        for i in RADIUS:
            for j in NUMP_WARP:
                current_iteration += 1
                
                # Update progress for each iteration within the file
                file_progress = (file_index - 1) / total_files * 100
                iteration_progress = current_iteration / total_iterations / total_files * 100
                update_progress_bar(file_progress + iteration_progress)
                
                cc1_new = []
                cc2_new = []

                if les_options[3] == 1: t1 = time.time()

                for k in range(Nthumb):
                    image = unregim[:, :, k]
                    v, u = optical_flow_ilk(im_ref, image, radius=i, num_warp=j, prefilter=True)
                    nr, nc = im_ref.shape
                    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
                    image1_warp = warp(image, np.array([row_coords + v, col_coords + u]), mode='reflect')
                    
                    if defmap_mode != "none" and k != mycref and i == RADIUS[-1] and j == NUMP_WARP[-1]:
                        uv_array.append([u, v])

                    cc1 = np.corrcoef(image.flatten(), im_ref.flatten())[0, 1]
                    cc2 = np.corrcoef(image1_warp.flatten(), im_ref.flatten())[0, 1]

                    if k != mycref:
                        cc1_new.append(cc1)
                        cc2_new.append(cc2)

                cc1_array.append(cc1_new)
                cc2_array.append(cc2_new)

                if les_options[3] == 1:
                    t2 = time.time()
                    time_exe.append(t2 - t1)

        cc1_array = np.array(cc1_array)
        cc2_array = np.array(cc2_array)
        data = [cc2_array[i] for i in range(len(cc1_array))]
        returned_data[file_key]["data"] = data

        list_des_moyenne = []
        list_des_ecarts_max = []

        for i in range(len(cc1_array)):
            if les_options[1] == 1:
                list_des_moyenne.append(np.mean(cc2_array[i]))
            if les_options[2] == 1:
                max_dev = abs(max(cc2_array[i]) - np.mean(cc2_array[i]))
                min_dev = abs(min(cc2_array[i]) - np.mean(cc2_array[i]))
                list_des_ecarts_max.append(max(max_dev, min_dev) ** 2)

        if les_options[1] == 1:
            matrix_moy = np.array(list_des_moyenne).reshape(len(RADIUS), len(NUMP_WARP))
            returned_data[file_key]["matrix_moy"] = matrix_moy
            best_index = np.unravel_index(np.argmax(matrix_moy), matrix_moy.shape)

        if les_options[2] == 1:
            matrix_err = np.array(list_des_ecarts_max).reshape(len(RADIUS), len(NUMP_WARP))
            returned_data[file_key]["matrix_err"] = matrix_err

        if les_options[3] == 1:
            matrix_time = np.array(time_exe).reshape(len(RADIUS), len(NUMP_WARP))
            returned_data[file_key]["matrix_time"] = matrix_time

        returned_data[file_key]["Best combination"] = best_index

    # Show deformation map if requested
    if defmap_mode != "none" and uv_array:
        show_deformation_map_ilk(uv_array, im_ref, defmap_mode, defmap_index)
        
    update_progress_bar(100)  # Set progress to 100%
    return returned_data

def v1_direct(les_options, file_unregistered, nump_warp_value=2, radius_value=2, defmap_mode="none", defmap_index=0):
    """
    Process FITS files using direct optical flow registration.
    
    Args:
        les_options: List of boolean options [old_cc, new_cc, image before, image after]
        file_unregistered: List of FITS file paths to process
        nump_warp_value: Number of warp iterations for optical flow (default: 2)
        radius_value: Radius parameter for optical flow (default: 2)
        defmap_mode: Mode for deformation map display ("none", "single") (default: "none")
        defmap_index: Index for single deformation map display (default: 0)
        
    Returns:
        Dictionary of registration results
    """
    global global_progress
    mycref = 33  # Reference thumbnail index
    Nthumb = 80  # Number of thumbnails to process
    returned_data = {}
    uv_array = []
    total_files = len(file_unregistered)
    
    # Check if any calculations are needed at all
    if not any(les_options):
        text_zone.config(state="normal")
        text_zone.insert(tk.END, "\nNo options selected for calculation.\n")
        text_zone.config(state="disabled")
        update_progress_bar(100)
        return returned_data
    
    # Determine what needs to be calculated
    need_cc_calc = les_options[0] or les_options[1]  # old_cc or new_cc
    need_images = les_options[2] or les_options[3]   # image before or image after
    
    try:
        for file_index, file_path in enumerate(file_unregistered, start=1):  
            file_key = f"file {file_index}"
            returned_data[file_key] = {}
            
            # Update progress for file loading stage
            file_progress_base = (file_index - 1) * 100 / total_files
            update_progress_bar(file_progress_base)
            
            # Status update
            text_zone.config(state="normal")
            text_zone.insert(tk.END, f"\nProcessing file {file_index}/{total_files}: {file_path}\n")
            text_zone.config(state="disabled")
            
            try:
                # Load and preprocess FITS data
                unregim = load_fits(file_path, Nthumb)
                
                # Get reference image
                im_ref = unregim[:, :, mycref]
                cc1_array = []
                cc2_array = []
                
                # Placeholders for image collections
                all_original_images = []
                all_warped_images = []
                
                # Process thumbnails
                for k in range(Nthumb):
                    # Update progress for each thumbnail
                    thumb_progress = k * 50 / (Nthumb * total_files)
                    update_progress_bar(file_progress_base + thumb_progress)
                    
                    image = unregim[:, :, k]
                    
                    # Store original image if needed
                    if need_images or need_cc_calc:
                        all_original_images.append(image)
                    
                    # Calculate original correlation coefficient if needed
                    if need_cc_calc and k != mycref:
                        cc1 = np.corrcoef(image.flatten(), im_ref.flatten())[0, 1]
                        cc1_array.append(cc1)
                    
                    # Perform alignment if needed
                    if need_cc_calc or need_images or defmap_mode != "none":
                        v, u = optical_flow_ilk(
                            im_ref, image, 
                            radius=radius_value, 
                            num_warp=nump_warp_value, 
                            prefilter=True
                        )
                        
                        # Create coordinate grid once per thumbnail
                        nr, nc = im_ref.shape
                        row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
                        
                        # Warp the image using the optical flow vectors
                        image1_warp = warp(
                            image, 
                            np.array([row_coords + v, col_coords + u]), 
                            mode='reflect'
                        )
                        
                        # Store warped image if needed
                        if need_images:
                            all_warped_images.append(image1_warp)
                        
                        # Store deformation vectors if requested
                        if defmap_mode != "none" and k != mycref:
                            uv_array.append([u, v])
                        
                        # Calculate warped correlation coefficient if needed
                        if need_cc_calc and k != mycref:
                            cc2 = np.corrcoef(image1_warp.flatten(), im_ref.flatten())[0, 1]
                            cc2_array.append(cc2)
                
                # Store only the requested results
                if les_options[0]:  # old_cc
                    returned_data[file_key]["data_old"] = cc1_array
                    
                if les_options[1]:  # new_cc
                    returned_data[file_key]["data_new"] = cc2_array
                    
                if les_options[2]:  # image before
                    returned_data[file_key]["image_before"] = im_ref
                    returned_data[file_key]["all_original_images"] = all_original_images
                    
                if les_options[3]:  # image after
                    returned_data[file_key]["image_after"] = image1_warp if 'image1_warp' in locals() else None
                    returned_data[file_key]["all_warped_images"] = all_warped_images
                
                # Always store reference index for display purposes
                returned_data[file_key]["reference_idx"] = mycref
                
                # Display processing statistics
                if need_cc_calc:
                    avg_cc1 = np.mean(cc1_array) if cc1_array else 0
                    avg_cc2 = np.mean(cc2_array) if cc2_array else 0
                    improvement = avg_cc2 - avg_cc1 if cc1_array and cc2_array else 0
                    
                    text_zone.config(state="normal")
                    text_zone.insert(tk.END, f"  Average correlation before: {avg_cc1:.4f}\n")
                    text_zone.insert(tk.END, f"  Average correlation after: {avg_cc2:.4f}\n")
                    text_zone.insert(tk.END, f"  Improvement: {improvement:.4f} ({improvement*100:.2f}%)\n")
                    text_zone.config(state="disabled")
                
            except Exception as e:
                text_zone.config(state="normal")
                text_zone.insert(tk.END, f"Error processing file {file_path}: {str(e)}\n")
                text_zone.config(state="disabled")
                # Continue with next file instead of aborting
                continue

        # Show deformation map if requested
        if defmap_mode != "none" and uv_array:
            try:
                show_deformation_map_ilk(uv_array, im_ref, defmap_mode, defmap_index)
            except Exception as e:
                text_zone.config(state="normal")
                text_zone.insert(tk.END, f"Error displaying deformation map: {str(e)}\n")
                text_zone.config(state="disabled")
        
        text_zone.config(state="normal")
        text_zone.insert(tk.END, f"\nProcessing complete for {len(file_unregistered)} files.\n")
        text_zone.config(state="disabled")
        
    except Exception as e:
        text_zone.config(state="normal")
        text_zone.insert(tk.END, f"Error during processing: {str(e)}\n")
        text_zone.config(state="disabled")
    
    update_progress_bar(100)  # Set progress to 100%
    return returned_data

def v_simple(file_unregistered, corners):
    global global_progress
    mycref = 33
    Nthumb = 80
    returned_data = {}
    src = [[50, 50], [150, 50], [150, 150], [50, 150]]
    dst = corners
    total_files = len(file_unregistered)
    
    for file_index, file_path in enumerate(file_unregistered, start=1):
        # Update progress - file loading
        update_progress_bar((file_index - 1) * 100 / total_files)
        
        file_key = f"file {file_index}"
        returned_data[file_key] = []

        hdul = fits.open(file_path)
        unregim = hdul[0].data.astype(float)
        unregim = np.transpose(unregim, (1, 2, 0))

        for tt in np.arange(Nthumb):
            medfi = scipy.signal.medfilt2d(unregim[:, :, tt])
            outl, noout = find_outliers_2D(unregim[:, :, tt], pclow=10, pchigh=90)
            for oo in np.arange(len(outl)):
                unregim[outl[oo][0], outl[oo][1], tt] = medfi[outl[oo][0], outl[oo][1]]

        im_ref = unregim[:, :, mycref]
        cc1_array = []
        cc2_array = []
        cc3_array = []
        cc4_array = []
        cc5_array = []
        cc6_array = []

        for k in range(0, Nthumb):
            # Update progress - processing step
            file_progress = (file_index - 1) / total_files * 100
            step_progress = k / Nthumb / total_files * 100
            update_progress_bar(file_progress + step_progress)
            
            image = unregim[:, :, k]
            original_corners = src
            transformed_corners = dst

            translation, rotation, scale = get_transform_params(original_corners, transformed_corners)

            tx, ty = translation
            theta = np.deg2rad(rotation)
            scale_x, scale_y = scale

            tform_transl = AffineTransform(translation=(tx, ty))
            im_reg_transl = warp(image, tform_transl.inverse)

            tform_rigid = AffineTransform(translation=(tx, ty), rotation=theta)
            im_reg_rigid = warp(image, tform_rigid.inverse)

            tform_sclrot = AffineTransform(scale=(scale_x, scale_y), rotation=theta)
            im_reg_sclrot = warp(image, tform_sclrot.inverse)

            tform_aff = AffineTransform(scale=(scale_x, scale_y), rotation=theta, translation=(tx, ty))
            im_reg_aff = warp(image, tform_aff.inverse)

            tform_bil = ProjectiveTransform()
            tform_bil.estimate(src, dst)
            im_reg_bil = warp(image, tform_bil.inverse)

            cc1 = np.corrcoef(im_ref.flatten(), image.flatten())[0, 1]
            cc2 = np.corrcoef(im_ref.flatten(), im_reg_transl.flatten())[0, 1]
            cc3 = np.corrcoef(im_ref.flatten(), im_reg_rigid.flatten())[0, 1]
            cc4 = np.corrcoef(im_ref.flatten(), im_reg_sclrot.flatten())[0, 1]
            cc5 = np.corrcoef(im_ref.flatten(), im_reg_aff.flatten())[0, 1]
            cc6 = np.corrcoef(im_ref.flatten(), im_reg_bil.flatten())[0, 1]

            if k != mycref:
                cc1_array.append(cc1)
                cc2_array.append(cc2)
                cc3_array.append(cc3)
                cc4_array.append(cc4)
                cc5_array.append(cc5)
                cc6_array.append(cc6)
            
        returned_data[file_key] = [np.mean(cc1_array),np.mean(cc2_array),np.mean(cc3_array),
                                 np.mean(cc4_array),np.mean(cc5_array),np.mean(cc6_array)]

    update_progress_bar(100)  # Set progress to 100%
    return returned_data

def get_transform_params(original_corners, transformed_corners):
    original_corners = np.array(original_corners)
    transformed_corners = np.array(transformed_corners)

    center_original = np.mean(original_corners, axis=0)
    center_transformed = np.mean(transformed_corners, axis=0)

    translation = center_transformed - center_original

    angle_original = np.arctan2(original_corners[1, 1] - center_original[1],
                               original_corners[1, 0] - center_original[0])
    angle_transformed = np.arctan2(transformed_corners[1, 1] - center_transformed[1],
                                  transformed_corners[1, 0] - center_transformed[0])
    rotation = np.rad2deg(angle_transformed - angle_original)

    dist_original = np.linalg.norm(original_corners[0] - original_corners[1])
    dist_transformed = np.linalg.norm(transformed_corners[0] - transformed_corners[1])
    scale_factor = dist_transformed / dist_original

    return translation, rotation, (scale_factor, scale_factor)

# --- FONCTIONS ---


def ouvrir_fichier():
    global fichiers_selectionnes
    fichiers_selectionnes = filedialog.askopenfilenames(
        title="Sélectionnez plusieurs fichiers",
        filetypes=[("Tous les fichiers", "*.*"),
                   ("Fichiers texte", "*.txt"),
                   ("Images", "*.png;*.jpg;*.jpeg"),
                   ("FITS files", "*.fits"),
                   ("NumPy files", "*.npy")]
    )

    if fichiers_selectionnes:
        messagebox.showinfo("Fichiers sélectionnés", f"{len(fichiers_selectionnes)} fichier(s) sélectionné(s) !")
        afficher_chemins()

# --- AFFICHER LES CHEMINS SELECTIONNÉS ---
def afficher_chemins():
    text_zone.config(state="normal")  # Débloque l'édition
    text_zone.delete("1.0", tk.END)  # Efface l'ancien contenu
    for fichier in fichiers_selectionnes:
        text_zone.insert(tk.END, f"{fichier}\n")  # Ajoute chaque fichier
    text_zone.config(state="disabled")  # Bloque l'édition

def enregistrer_fichier():
    messagebox.showinfo("Enregistrer", "Fichier enregistré !")

def quitter_application():
    app.quit()

def a_propos():
    messagebox.showinfo("À propos", "Cette application utilise CustomTkinter")

# --- BARRE DE MENU ---
menu_bar = tk.Menu(app)

# --- MENU FICHIER ---
menu_fichier = tk.Menu(menu_bar, tearoff=0)
menu_fichier.add_command(label="Ouvrir", command=ouvrir_fichier)
menu_fichier.add_command(label="Enregistrer", command=enregistrer_fichier)
menu_fichier.add_separator()
menu_fichier.add_command(label="Quitter", command=quitter_application)

# --- MENU APPEARANCE ---
menu_appearance = tk.Menu(menu_bar, tearoff=0)
# Appearance mode submenu
appearance_mode_menu = tk.Menu(menu_appearance, tearoff=0)
appearance_mode_menu.add_command(label="Light", command=lambda: change_appearance_mode("light"))
appearance_mode_menu.add_command(label="Dark", command=lambda: change_appearance_mode("dark"))
appearance_mode_menu.add_command(label="System", command=lambda: change_appearance_mode("system"))
menu_appearance.add_cascade(label="Appearance Mode", menu=appearance_mode_menu)

# Coffee theme submenu
coffee_theme_menu = tk.Menu(menu_appearance, tearoff=0)
coffee_theme_menu.add_command(label="Espresso (Dark)", command=lambda: change_color_theme("espresso"))
coffee_theme_menu.add_command(label="Latte (Light Brown)", command=lambda: change_color_theme("latte"))
coffee_theme_menu.add_command(label="Cappuccino (Medium)", command=lambda: change_color_theme("cappuccino"))
coffee_theme_menu.add_command(label="Mocha (Dark Brown)", command=lambda: change_color_theme("mocha"))
menu_appearance.add_cascade(label="Coffee Themes", menu=coffee_theme_menu)

# --- MENU OUTILS ---
menu_outils = tk.Menu(menu_bar, tearoff=0)
functions_dict = {
    "V1 search": lambda: update_interface("search"),
    "V1 calculate": lambda: update_interface("use"),
    "simple defortmation": lambda: update_interface("Deformation simple"),
    "Panchromatic Registration": lambda: update_interface("panchro")
}

def update_interface(mode):
    """
    Update the interface to display the selected mode.
    
    Args:
        mode: The interface mode to display ("search", "use", "Deformation simple", or "panchro")
    """
    # Clear existing widgets
    for widget in frame_droite.winfo_children():
        widget.destroy()

    # Create header for the selected mode
    header_label = ctk.CTkLabel(frame_droite, text=f"Mode: {mode}", font=("Arial", 14, "bold"))
    header_label.pack(pady=10)
    
    # Display the appropriate interface based on mode
    if mode == "search":
        create_search_interface()
    elif mode == "use":
        create_direct_calculation_interface()
    elif mode == "Deformation simple":
        create_deformation_interface()
    elif mode == "panchro":
        create_panchromatic_interface()
    else:
        # Display error for unknown mode
        error_label = ctk.CTkLabel(
            frame_droite, 
            text=f"Unknown mode: {mode}", 
            font=("Arial", 14, "bold"),
            text_color="red"
        )
        error_label.pack(pady=20)

def create_search_interface():
    """Create the search mode interface with parameter optimization"""
    # Create header
    label_droite = ctk.CTkLabel(frame_droite, text="🎛️ Actions :", font=("Arial", 14, "bold"))
    label_droite.pack(pady=10)

    # --- CHECKBOXES ---
    checkboxes = []
    checkbox_vars = []
    checkboxes_names = ["box plot", "moyenne", "variance", "temps de traitement"]
    label_checkbox = ctk.CTkLabel(frame_droite, text="✅ Sélectionnez des options :", font=("Arial", 14))
    label_checkbox.pack(pady=10)

    # Create checkboxes
    for i in range(4):
        var = ctk.IntVar(value=0)  # 0 par défaut
        checkbox = ctk.CTkCheckBox(frame_droite, text=checkboxes_names[i], variable=var)
        checkbox.pack(pady=2, anchor="w")
        checkboxes.append(checkbox)
        checkbox_vars.append(var)

    def get_checkbox_states():
        global states 
        states = [var.get() for var in checkbox_vars]

    # Update displayed value functions
    def update_value_radius(value):
        global radius_g
        value_label_radius.configure(text=f"radius: {int(float(value))}")
        radius_g = int(float(value))

    def update_value_nump(value):
        global nump_warp_g
        value_label_nump.configure(text=f"nump warp: {int(float(value))}")
        nump_warp_g = int(float(value))

    # Labels to display current slider values
    value_label_radius = ctk.CTkLabel(frame_droite, text="radius : 1", font=("Arial", 16))
    value_label_nump = ctk.CTkLabel(frame_droite, text="nump warp : 1", font=("Arial", 16))

    # Sliders (1-30)
    value_label_radius.pack(pady=10)
    slider_radius = ctk.CTkSlider(
        frame_droite,
        from_=1,
        to=30,
        number_of_steps=29,
        command=update_value_radius
    )
    slider_radius.set(1)
    slider_radius.pack(pady=10)

    value_label_nump.pack(pady=10)
    slider_nump_warp = ctk.CTkSlider(
        frame_droite,
        from_=1,
        to=30,
        number_of_steps=29,
        command=update_value_nump
    )
    slider_nump_warp.set(1)
    slider_nump_warp.pack(pady=10)

    # Deformation map options
    label_defmap = ctk.CTkLabel(frame_droite, text="Deformation Map Mode:", font=("Arial", 14))
    label_defmap.pack(pady=10)
    defmap_mode_var = tk.StringVar(value="none")
    modes = ["none", "single"]
    for mode in modes:
        rb = ctk.CTkRadioButton(frame_droite, text=mode, variable=defmap_mode_var, value=mode)
        rb.pack(pady=2)
        
    label_defmap_index = ctk.CTkLabel(frame_droite, text="Deformation Map Index:")
    label_defmap_index.pack(pady=5)
    defmap_index_entry = ctk.CTkEntry(frame_droite)
    defmap_index_entry.pack(pady=5)
    defmap_index_entry.insert(0, "1")
    
    def start_processing():
        """Start the search processing"""
        if not fichiers_selectionnes:
            messagebox.showerror("Error", "Please select files first!")
            return

        try:
            # Update UI state
            get_checkbox_states()
            update_progress_bar(0)  # Reset progress bar
            
            # Get deformation map parameters
            defmap_mode = defmap_mode_var.get()
            defmap_index = int(defmap_index_entry.get()) if defmap_mode == "single" else 0
            
            # Process directly without queues
            results = v1_pipelin_youness(
                states, 
                fichiers_selectionnes, 
                int(nump_warp_g), 
                int(radius_g),
                defmap_mode,
                defmap_index
            )
            plot_results(results)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # Start processing button
    custom_button = ctk.CTkButton(
        frame_droite,
        text="Lancer le traitement",
        width=200,
        height=50,
        fg_color="dodgerblue",
        hover_color="deepskyblue",
        text_color="white",
        corner_radius=10,
        border_width=2,
        border_color="white",
        font=("Arial", 16, "bold"),
        command=start_processing
    )
    custom_button.pack(pady=20)

def create_direct_calculation_interface():
    """Create the direct calculation interface"""
    # --- CHECKBOXES ---
    checkboxes = []
    checkbox_vars = []
    checkboxes_names = ["old_cc", "new_cc", "image before", "image after"]
    label_checkbox = ctk.CTkLabel(frame_droite, text="✅ Sélectionnez des options :", font=("Arial", 14))
    label_checkbox.pack(pady=10)

    for i in range(4):
        var = ctk.IntVar(value=0)
        checkbox = ctk.CTkCheckBox(frame_droite, text=checkboxes_names[i], variable=var)
        checkbox.pack(pady=2, anchor="w")
        checkboxes.append(checkbox)
        checkbox_vars.append(var)

    def get_checkbox_states():
        global states 
        states = [var.get() for var in checkbox_vars]

    # Parameter inputs
    radius_label = ctk.CTkLabel(frame_droite, text="Radius:")
    radius_label.pack(pady=5)
    radius_entry = ctk.CTkEntry(frame_droite)
    radius_entry.pack(pady=5)
    radius_entry.insert(0, "2")

    nump_warp_label = ctk.CTkLabel(frame_droite, text="Num Warp:")
    nump_warp_label.pack(pady=5)
    nump_warp_entry = ctk.CTkEntry(frame_droite)
    nump_warp_entry.pack(pady=5)
    nump_warp_entry.insert(0, "2")

    # Deformation map options
    label_defmap = ctk.CTkLabel(frame_droite, text="Deformation Map Mode:", font=("Arial", 14))
    label_defmap.pack(pady=10)
    defmap_mode_var = tk.StringVar(value="none")
    modes = ["none", "single"]
    for mode in modes:
        rb = ctk.CTkRadioButton(frame_droite, text=mode, variable=defmap_mode_var, value=mode)
        rb.pack(pady=2)
        
    label_defmap_index = ctk.CTkLabel(frame_droite, text="Deformation Map Index:")
    label_defmap_index.pack(pady=5)
    defmap_index_entry = ctk.CTkEntry(frame_droite)
    defmap_index_entry.pack(pady=5)
    defmap_index_entry.insert(0, "1")
    
    def start_processing_direct():
        """Start the direct calculation processing"""
        if not fichiers_selectionnes:
            messagebox.showerror("Error", "Please select files first!")
            return

        try:
            # Update UI state and validate options
            get_checkbox_states()
            
            if not any(states):
                messagebox.showwarning("Warning", "Please select at least one option.")
                return
                
            update_progress_bar(0)  # Reset progress bar
            
            # Get deformation map parameters
            defmap_mode = defmap_mode_var.get()
            defmap_index = int(defmap_index_entry.get()) if defmap_mode == "single" else 0
            
            # Process directly without queues
            results = v1_direct(
                states, 
                fichiers_selectionnes, 
                int(nump_warp_entry.get()), 
                int(radius_entry.get()),
                defmap_mode,
                defmap_index
            )
            plot_results_direct(results)
            
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid parameter value: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # Start button
    start_button = ctk.CTkButton(
        frame_droite,
        text="Start Processing",
        command=start_processing_direct,
        width=200,
        height=50,
        fg_color="dodgerblue",
        hover_color="deepskyblue"
    )
    start_button.pack(pady=20)

def create_deformation_interface():
    """Create the simple deformation interface"""
    label_deform = ctk.CTkLabel(frame_droite, text="Déformations simples", font=("Arial", 14, "bold"))
    label_deform.pack(pady=5)

    canvas_size = 200
    canvas = tk.Canvas(frame_droite, width=canvas_size, height=canvas_size, bg="white")
    canvas.pack(pady=5)

    global corners
    corners = [[50, 50], [150, 50], [150, 150], [50, 150]]
    square = canvas.create_polygon(*[coord for point in corners for coord in point], fill="skyblue")

    def clamp_points(points):
        clamped = []
        for x, y in points:
            x = max(0, min(canvas_size, x))
            y = max(0, min(canvas_size, y))
            clamped.append([x, y])
        return clamped

    def update_square():
        global corners
        clamped = clamp_points(corners)
        corners[:] = clamped
        coords = [coord for pt in clamped for coord in pt]
        canvas.coords(square, *coords)
        update_corner_entries()

    def rotate_square(angle_deg):
        global corners
        angle_rad = np.radians(float(angle_deg))
        center = np.mean(corners, axis=0)
        rot = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        corners = [((np.array(p) - center) @ rot.T + center).tolist() for p in corners]
        update_square()

    def scale_square(scale_factor):
        global corners
        scale_factor = float(scale_factor)
        center = np.mean(corners, axis=0)
        corners = [((np.array(p) - center) * scale_factor + center).tolist() for p in corners]
        update_square()

    def move_square(dx, dy):
        global corners
        corners = [[x + dx, y + dy] for x, y in corners]
        update_square()

    corner_entries = []

    def update_corner_entries():
        for i, entry_pair in enumerate(corner_entries):
            x, y = corners[i]
            entry_pair[0].delete(0, tk.END)
            entry_pair[0].insert(0, str(round(x, 1)))
            entry_pair[1].delete(0, tk.END)
            entry_pair[1].insert(0, str(round(y, 1)))

    def on_corner_change():
        global corners
        try:
            corners = [
                [float(entry_x.get()), float(entry_y.get())]
                for entry_x, entry_y in corner_entries
            ]
            update_square()
        except ValueError:
            pass

    ctk.CTkLabel(frame_droite, text="Coins (x, y)").pack(pady=(10, 0))
    for i in range(4):
        frame = ctk.CTkFrame(frame_droite)
        frame.pack()
        label = ctk.CTkLabel(frame, text=f"Coin {i + 1}")
        label.pack(side="left", padx=2)
        entry_x = ctk.CTkEntry(frame, width=60)
        entry_x.pack(side="left")
        entry_y = ctk.CTkEntry(frame, width=60)
        entry_y.pack(side="left")
        entry_x.bind("<Return>", lambda e: on_corner_change())
        entry_y.bind("<Return>", lambda e: on_corner_change())
        corner_entries.append((entry_x, entry_y))

    update_corner_entries()

    ctk.CTkLabel(frame_droite, text="Rotation (°)").pack()
    rotation_slider = ctk.CTkSlider(frame_droite, from_=0, to=360, command=rotate_square)
    rotation_slider.pack()

    ctk.CTkLabel(frame_droite, text="Scale").pack()
    scale_slider = ctk.CTkSlider(frame_droite, from_=0.5, to=2.0, command=scale_square)
    scale_slider.set(1.0)
    scale_slider.pack()

    btn_frame = ctk.CTkFrame(frame_droite)
    btn_frame.pack(pady=5)

    ctk.CTkButton(btn_frame, text="⬅️", command=lambda: move_square(-10, 0)).grid(row=0, column=0)
    ctk.CTkButton(btn_frame, text="➡️", command=lambda: move_square(10, 0)).grid(row=0, column=2)
    ctk.CTkButton(btn_frame, text="⬆️", command=lambda: move_square(0, -10)).grid(row=0, column=1)
    ctk.CTkButton(btn_frame, text="⬇️", command=lambda: move_square(0, 10)).grid(row=1, column=1)

    def lancer():
        """Start simple deformation analysis"""
        if not fichiers_selectionnes:
            messagebox.showerror("Error", "Please select files first!")
            return

        try:
            update_progress_bar(0)  # Reset progress bar
            
            # Process directly without queues
            results = v_simple(fichiers_selectionnes, corners)
            
            # Display results in text zone
            text_zone.config(state="normal")
            text_zone.insert(tk.END, "\n=== Simple Deformation Results ===\n")
            for file_key, value in results.items():
                text_zone.insert(tk.END, f"\n{file_key} :\n")
                text_zone.insert(tk.END, f"Original correlation = {value[0]:.4f}\n")
                text_zone.insert(tk.END, f"Translation: {value[1]:.4f} (Δ: {value[1]-value[0]:.4f})\n")
                text_zone.insert(tk.END, f"Rigid body: {value[2]:.4f} (Δ: {value[2]-value[0]:.4f})\n")
                text_zone.insert(tk.END, f"Scaled rotation: {value[3]:.4f} (Δ: {value[3]-value[0]:.4f})\n")
                text_zone.insert(tk.END, f"Affine: {value[4]:.4f} (Δ: {value[4]-value[0]:.4f})\n")
                text_zone.insert(tk.END, f"Bilinear: {value[5]:.4f} (Δ: {value[5]-value[0]:.4f})\n")
            text_zone.insert(tk.END, "\nNote: Rigid body = translation + rotation\nScaled rotation = scaling + rotation\nAffine = scaling + rotation + translation\n")
            text_zone.config(state="disabled")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    start_button = ctk.CTkButton(
        frame_droite,
        text="Lancer",
        command=lancer,
        width=150,
        height=30,
        fg_color="dodgerblue",
        hover_color="deepskyblue",
        text_color="white",
        corner_radius=10,
        border_width=2,
        border_color="white",
        font=("Arial", 16, "bold")
    )
    start_button.pack(pady=20)

def create_panchromatic_interface():
    """Create the panchromatic registration interface"""
    # Radius input
    radius_label = ctk.CTkLabel(frame_droite, text="Radius:")
    radius_label.pack(pady=5)
    radius_entry = ctk.CTkEntry(frame_droite)
    radius_entry.pack(pady=5)
    radius_entry.insert(0, "2")

    # Num warp input
    num_warp_label = ctk.CTkLabel(frame_droite, text="Num Warp:")
    num_warp_label.pack(pady=5)
    num_warp_entry = ctk.CTkEntry(frame_droite)
    num_warp_entry.pack(pady=5)
    num_warp_entry.insert(0, "10")

    # Reference frame input
    ref_label = ctk.CTkLabel(frame_droite, text="Reference Frame:")
    ref_label.pack(pady=5)
    ref_entry = ctk.CTkEntry(frame_droite)
    ref_entry.pack(pady=5)
    ref_entry.insert(0, "0")

    # Deformation map mode
    defmap_mode_label = ctk.CTkLabel(frame_droite, text="Deformation Map Mode:")
    defmap_mode_label.pack(pady=5)
    defmap_mode_var = tk.StringVar(value="none")
    modes = ["none", "single", "all"]
    for mode in modes:
        rb = ctk.CTkRadioButton(frame_droite, text=mode, variable=defmap_mode_var, value=mode)
        rb.pack(pady=2)

    # Deformation map index
    defmap_index_label = ctk.CTkLabel(frame_droite, text="Deformation Map Index:")
    defmap_index_label.pack(pady=5)
    defmap_index_entry = ctk.CTkEntry(frame_droite)
    defmap_index_entry.pack(pady=5)
    defmap_index_entry.insert(0, "1")

    # Process mode
    process_mode_label = ctk.CTkLabel(frame_droite, text="Processing Mode:")
    process_mode_label.pack(pady=5)
    process_mode_var = tk.StringVar(value="direct")
    modes = [("Direct NPY", "direct"), ("Process FITS", "fits")]
    for text, value in modes:
        rb = ctk.CTkRadioButton(frame_droite, text=text, variable=process_mode_var, value=value)
        rb.pack(pady=2)

    def start_panchro_registration():
        """Start panchromatic registration"""
        if not fichiers_selectionnes:
            messagebox.showerror("Error", "Please select files first!")
            return

        try:
            # Get parameters
            radius = int(radius_entry.get())
            num_warp = int(num_warp_entry.get())
            ref = int(ref_entry.get())
            defmap_index = int(defmap_index_entry.get())
            process_mode = process_mode_var.get()
            defmap_mode = defmap_mode_var.get()
            
            # Reset progress
            update_progress_bar(0)
            
            # Display processing information
            text_zone.config(state="normal")
            text_zone.insert(tk.END, "\n=== Panchromatic Registration ===\n")
            text_zone.insert(tk.END, f"Mode: {process_mode}\n")
            text_zone.insert(tk.END, f"Parameters: radius={radius}, num_warp={num_warp}, ref={ref}\n")
            text_zone.insert(tk.END, f"Deformation map: {defmap_mode} (index: {defmap_index})\n")
            text_zone.insert(tk.END, f"Processing {len(fichiers_selectionnes)} files...\n")
            text_zone.config(state="disabled")
            
            if process_mode == "direct":
                # Process .npy files directly
                valid_files = []
                for file_path in fichiers_selectionnes:
                    if file_path.endswith('.npy'):
                        valid_files.append(file_path)
                    else:
                        messagebox.showwarning("Warning", f"Skipping non-NPY file: {file_path}")
                
                if not valid_files:
                    messagebox.showerror("Error", "No valid .npy files selected.")
                    return
                
                total_files = len(valid_files)
                for i, file_path in enumerate(valid_files):
                    update_progress_bar((i / total_files) * 100)
                    panchro_array = np.load(file_path)
                    registered_images = panchromatic_registration(
                        panchro_array,
                        radius=radius,
                        num_warp=num_warp,
                        ref=ref,
                        defmap_mode=defmap_mode,
                        defmap_index=defmap_index
                    )
                    
                    # Add option to save the registered images
                    if registered_images is not None:
                        save_option = messagebox.askyesno(
                            "Save Result", 
                            f"Would you like to save the registered images from {file_path}?"
                        )
                        if save_option:
                            save_path = filedialog.asksaveasfilename(
                                defaultextension=".npy",
                                filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
                                initialfile=f"registered_{os.path.basename(file_path)}"
                            )
                            if save_path:
                                np.save(save_path, registered_images)
                                text_zone.config(state="normal")
                                text_zone.insert(tk.END, f"Registered images saved to {save_path}\n")
                                text_zone.config(state="disabled")
            else:
                # Process FITS files through the entire pipeline
                valid_files = []
                for file_path in fichiers_selectionnes:
                    if file_path.endswith('.fits'):
                        valid_files.append(file_path)
                    else:
                        messagebox.showwarning("Warning", f"Skipping non-FITS file: {file_path}")
                
                if not valid_files:
                    messagebox.showerror("Error", "No valid FITS files selected.")
                    return
                
                # Process FITS files to create panchromatic array
                panchro_array = process_fits_to_panchromatic(
                    valid_files,
                    radius=radius,
                    num_warp=num_warp,
                    mycref=33,  # Default reference
                    Nthumb=80,   # Default number of thumbnails
                    defmap_mode=defmap_mode,
                    defmap_index=defmap_index
                )
                
                # Register the panchromatic images
                registered_images = panchromatic_registration(
                    panchro_array,
                    radius=radius,
                    num_warp=num_warp,
                    ref=ref,
                    defmap_mode=defmap_mode,
                    defmap_index=defmap_index
                )
                
                # Add option to save the panchromatic images and registered images
                if registered_images is not None:
                    save_option = messagebox.askyesno(
                        "Save Results", 
                        "Would you like to save the panchromatic and registered images?"
                    )
                    if save_option:
                        panchro_path = filedialog.asksaveasfilename(
                            defaultextension=".npy",
                            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
                            initialfile="panchromatic_images.npy"
                        )
                        if panchro_path:
                            np.save(panchro_path, panchro_array)
                            reg_path = panchro_path.replace(".npy", "_registered.npy")
                            np.save(reg_path, registered_images)
                            text_zone.config(state="normal")
                            text_zone.insert(tk.END, f"Panchromatic images saved to {panchro_path}\n")
                            text_zone.insert(tk.END, f"Registered images saved to {reg_path}\n")
                            text_zone.config(state="disabled")
            
            update_progress_bar(100)  # Complete progress
            
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid parameter value: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            import traceback
            traceback.print_exc()

    start_button = ctk.CTkButton(
        frame_droite,
        text="Start Registration",
        command=start_panchro_registration,
        width=200,
        height=50,
        fg_color="dodgerblue",
        hover_color="deepskyblue",
        text_color="white",
        corner_radius=10,
        border_width=2,
        border_color="white",
        font=("Arial", 16, "bold")
    )
    start_button.pack(pady=20)

for name, func in functions_dict.items():
    menu_outils.add_command(label=name, command=func)

# --- MENU AIDE ---
menu_aide = tk.Menu(menu_bar, tearoff=0)
menu_aide.add_command(label="À propos", command=a_propos)
menu_aide.add_separator()
menu_aide.add_command(label="Feature Report", command=generate_feature_report)

# Ajout des menus à la barre
menu_bar.add_cascade(label="Fichier", menu=menu_fichier)
menu_bar.add_cascade(label="Appearance", menu=menu_appearance)
menu_bar.add_cascade(label="Outils", menu=menu_outils)
menu_bar.add_cascade(label="Aide", menu=menu_aide)

# Assigner la barre de menu à la fenêtre principale
app.config(menu=menu_bar)

# --- CADRE DROITE : Zone d'affichage des fichiers ---
frame_droite = ctk.CTkFrame(app)
frame_droite.grid(row=0, column=1, sticky="nswe", padx=10, pady=10)

# --- CADRE GAUCHE : Zone d'affichage des fichiers ---
frame_gauche = ctk.CTkFrame(app)
frame_gauche.grid(row=0, column=0, sticky="nswe", padx=10, pady=10)

label_gauche = ctk.CTkLabel(frame_gauche, text="📂 Fichiers sélectionnés :", font=("Arial", 14, "bold"))
label_gauche.pack(pady=10)

text_zone = tk.Text(frame_gauche, height=20, wrap="word", font=("Arial", 12))
text_zone.pack(pady=5, padx=5, fill="both", expand=True)
text_zone.config(state="disabled")  # Désactive l'édition

def plot_results(returned_data):
    if not returned_data:
        messagebox.showinfo("Error", "No data to display")
        return
        
    # Create a single window for all files
    window = ctk.CTkToplevel(app)
    window.title("Analysis Results")
    window.geometry("900x700")
    window.protocol("WM_DELETE_WINDOW", lambda: close_plot_window(window))
    
    # Create main frame
    main_frame = ctk.CTkFrame(window)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Current file index
    current_idx = [0]
    file_keys = list(returned_data.keys())
    
    # Create figure and canvas
    fig = plt.Figure(figsize=(8, 7), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=main_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Add to global list to prevent garbage collection
    plot_figures.append((fig, canvas, window))
    
    # Navigation buttons frame
    nav_frame = ctk.CTkFrame(main_frame)
    nav_frame.pack(side=tk.BOTTOM, fill=tk.X)
    
    # Info label
    info_label = ctk.CTkLabel(nav_frame, text=f"File: {file_keys[0]} (1/{len(file_keys)})")
    info_label.pack(side=tk.LEFT, padx=20, pady=5)
    
    # Function to update the displayed file
    def update_plot():
        fig.clear()
        
        file_key = file_keys[current_idx[0]]
        matrices = returned_data[file_key]
        valid_matrices = {k: v if k != "Best combination" else afficher_combinaison(k,v) for k, v in matrices.items()}
        
        # Create subplot grid
        rows, cols = 2, 2  # Fixed 2x2 grid
        grid = plt.GridSpec(rows, cols, figure=fig)
        
        # Plot each matrix
        for idx, (matrix_name, matrix) in enumerate(list(valid_matrices.items())[:-1]):
            row, col = idx // cols, idx % cols
            ax = fig.add_subplot(grid[row, col])
            
            if matrix_name == "data":
                ax.boxplot(matrix)
                ax.set_xlabel("Columns")
                ax.set_ylabel("Rows")
            else:
                im = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
                fig.colorbar(im, ax=ax)
                ax.set_xlabel("Columns")
                ax.set_ylabel("Rows")
            
            ax.set_title(matrix_name)
        
        fig.suptitle(f"File: {file_key}", fontsize=14, fontweight="bold")
        fig.tight_layout()
        canvas.draw()
        
        # Update info label
        info_label.configure(text=f"File: {file_key} ({current_idx[0]+1}/{len(file_keys)})")
    
    # Function to navigate to the previous file
    def prev_file():
        if current_idx[0] > 0:
            current_idx[0] -= 1
            update_plot()
    
    # Function to navigate to the next file
    def next_file():
        if current_idx[0] < len(file_keys) - 1:
            current_idx[0] += 1
            update_plot()
    
    # Navigation buttons
    prev_btn = ctk.CTkButton(nav_frame, text="◀ Prev File", width=100, command=prev_file)
    prev_btn.pack(side=tk.LEFT, padx=5, pady=5)
    
    next_btn = ctk.CTkButton(nav_frame, text="Next File ▶", width=100, command=next_file)
    next_btn.pack(side=tk.LEFT, padx=5, pady=5)
    
    # Show the first file
    update_plot()
    
    # Bring window to front
    window.lift()
    window.focus_set()

def plot_results_direct(returned_data):
    if not returned_data:
        messagebox.showinfo("Error", "No data to display")
        return
        
    # Create a single window for all files
    window = ctk.CTkToplevel(app)
    window.title("Image Registration Results")
    window.geometry("1200x800")
    window.protocol("WM_DELETE_WINDOW", lambda: close_plot_window(window))
    
    # Create main frame with layout for controls at the bottom
    main_frame = ctk.CTkFrame(window)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Create frames for different parts of the UI
    plot_frame = ctk.CTkFrame(main_frame)  # Frame for matplotlib plot
    plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Control panel for image browser - Make this more prominent with a border
    browser_frame = ctk.CTkFrame(main_frame, fg_color=("gray85", "gray25"), border_width=1, border_color=("gray70", "gray40"))
    browser_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
    
    # Navigation buttons frame - add visual separation
    nav_frame = ctk.CTkFrame(main_frame, fg_color=("gray80", "gray30"), border_width=1, border_color=("gray70", "gray40"))
    nav_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=(0, 5))
    
    # Current indices
    current_file_idx = [0]
    current_view_mode = [0]  # 0: statistics, 1: image browser
    current_image_idx = [0]  # Current image in the sequence
    view_original = [True]   # Whether to show original or warped image
    
    file_keys = list(returned_data.keys())
    view_modes = ["Statistics", "Image Browser"]
    
    # Create figure and canvas
    fig = plt.Figure(figsize=(10, 7), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)  # Place in plot_frame instead of main_frame
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Add to global list to prevent garbage collection
    plot_figures.append((fig, canvas, window))
    
    # Info label
    info_label = ctk.CTkLabel(nav_frame, text="", font=("Arial", 12, "bold"))
    info_label.pack(side=tk.LEFT, padx=20, pady=5)
    
    # Function to update the displayed content
    def update_display():
        fig.clear()
        
        file_key = file_keys[current_file_idx[0]]
        matrices = returned_data[file_key]
        
        # Get reference index
        ref_idx = matrices.get("reference_idx", 33)
        
        if current_view_mode[0] == 0:
            # Statistics view - only show plots for selected options
            
            # Determine how many plots we need based on available data
            plot_data = []
            plot_titles = []
            
            if "data_old" in matrices:  # Option 1: old_cc
                plot_data.append(("data_old", matrices["data_old"]))
                plot_titles.append("Original Data")
                
            if "data_new" in matrices:  # Option 2: new_cc
                plot_data.append(("data_new", matrices["data_new"]))
                plot_titles.append("Processed Data")
                
            if "image_before" in matrices:  # Option 3: image before
                plot_data.append(("image_before", matrices["image_before"]))
                plot_titles.append("Reference Image")
                
            if "image_after" in matrices:  # Option 4: image after
                plot_data.append(("image_after", matrices["image_after"]))
                plot_titles.append("Sample Aligned Image")
            
            # Calculate grid dimensions based on number of plots
            num_plots = len(plot_data)
            if num_plots == 0:
                # No data to display - show message
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "No plots selected for display", 
                       horizontalalignment='center', verticalalignment='center',
                       fontsize=14)
                ax.axis('off')
            else:
                # Determine grid layout based on number of plots
                if num_plots == 1:
                    rows, cols = 1, 1
                elif num_plots == 2:
                    rows, cols = 1, 2
                elif num_plots <= 4:
                    rows, cols = 2, 2
                else:
                    rows, cols = (num_plots + 2) // 3, 3
                
                grid = plt.GridSpec(rows, cols, figure=fig)
                
                # Display the plots
                for idx, (key_name, data) in enumerate(plot_data):
                    row, col = idx // cols, idx % cols
                    ax = fig.add_subplot(grid[row, col])
                    
                    if "data" in key_name:
                        ax.boxplot(data)
                        ax.set_xlabel("Thumbnails")
                        ax.set_ylabel("Correlation")
                    else:
                        im = ax.imshow(data)
                        fig.colorbar(im, ax=ax)
                    
                    ax.set_title(plot_titles[idx])
            
            # Update info label
            info_label.configure(text=f"File: {file_key} ({current_file_idx[0]+1}/{len(file_keys)}) - View: {view_modes[current_view_mode[0]]}")
            
            # Update browser frame - hide in statistics mode
            for widget in browser_frame.winfo_children():
                widget.destroy()
                
        else:
            # Image browser view - check if we have the necessary image data
            has_original = "all_original_images" in matrices
            has_warped = "all_warped_images" in matrices
            
            if not has_original and not has_warped:
                # No image data to display
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "No image data available to display.\nSelect 'image before' or 'image after' options.", 
                       horizontalalignment='center', verticalalignment='center',
                       fontsize=14)
                ax.axis('off')
                
                # Update info label
                info_label.configure(text=f"File: {file_key} ({current_file_idx[0]+1}/{len(file_keys)}) - No image data available")
                return
            
            # Show a single image with navigation
            ax = fig.add_subplot(111)
            
            # Determine which type of image to display based on availability and user selection
            display_original = has_original and (view_original[0] or not has_warped)
            display_warped = has_warped and (not view_original[0] or not has_original)
            
            if display_original:
                all_images = matrices["all_original_images"]
                suffix = "Original"
            elif display_warped:
                all_images = matrices["all_warped_images"]
                suffix = "Aligned"
            else:
                # This should not happen, but just in case
                info_label.configure(text=f"File: {file_key} - Error: No images available to display")
                return
                
            # Safety check for index
            max_idx = len(all_images) - 1
            if current_image_idx[0] > max_idx:
                current_image_idx[0] = max_idx
            
            # Display the image
            img = all_images[current_image_idx[0]]
            
            # Mark reference image differently
            is_reference = current_image_idx[0] == ref_idx
            ref_marker = " (REFERENCE)" if is_reference else ""
            
            im = ax.imshow(img)
            ax.set_title(f"Image {current_image_idx[0]+1}/{len(all_images)}{ref_marker} - {suffix}")
            fig.colorbar(im, ax=ax)
            
            # Update info label with more details
            info_label.configure(
                text=f"File: {file_key} ({current_file_idx[0]+1}/{len(file_keys)}) - " +
                     f"Image: {current_image_idx[0]+1}/{len(all_images)}{ref_marker} - {suffix}"
            )
            
            # Show browser controls
            update_browser_controls()
        
        fig.tight_layout()
        canvas.draw()
    
    # Function to update browser controls
    def update_browser_controls():
        # Clear existing controls
        for widget in browser_frame.winfo_children():
            widget.destroy()
        
        if current_view_mode[0] == 1:  # Only in image browser mode
            file_key = file_keys[current_file_idx[0]]
            matrices = returned_data[file_key]
            
            has_original = "all_original_images" in matrices
            has_warped = "all_warped_images" in matrices
            
            if not has_original and not has_warped:
                # No image data to display
                return
                
            # Determine which images to use for controls
            if has_original:
                all_images = matrices["all_original_images"]
            else:
                all_images = matrices["all_warped_images"]
            
            # Create a title for the controls
            controls_label = ctk.CTkLabel(browser_frame, text="Image Navigation Controls", font=("Arial", 12, "bold"))
            controls_label.pack(pady=(5, 0))
            
            # Create a horizontal frame for navigation controls
            nav_controls = ctk.CTkFrame(browser_frame, fg_color="transparent")
            nav_controls.pack(fill=tk.X, padx=10, pady=5)
            
            # Image navigation
            prev_10_btn = ctk.CTkButton(nav_controls, text="◀◀ -10", width=60, height=28, 
                                      command=lambda: change_image(-10))
            prev_10_btn.pack(side=tk.LEFT, padx=5, pady=5)
            
            prev_btn = ctk.CTkButton(nav_controls, text="◀ Prev", width=80, height=28,
                                   command=lambda: change_image(-1))
            prev_btn.pack(side=tk.LEFT, padx=5, pady=5)
            
            # Image index slider
            def update_slider(value):
                current_image_idx[0] = int(float(value))
                update_display()
                
            slider_label = ctk.CTkLabel(nav_controls, text="Image Index:", font=("Arial", 12))
            slider_label.pack(side=tk.LEFT, padx=5)
            
            slider = ctk.CTkSlider(
                nav_controls,
                from_=0,
                to=len(all_images)-1,
                number_of_steps=len(all_images)-1,
                command=update_slider,
                width=200
            )
            slider.set(current_image_idx[0])
            slider.pack(side=tk.LEFT, padx=5, pady=5)
            
            # Create slider value label
            slider_value = ctk.CTkLabel(nav_controls, text=f"{current_image_idx[0]+1}", width=30, font=("Arial", 12))
            slider_value.pack(side=tk.LEFT, padx=2)
            
            next_btn = ctk.CTkButton(nav_controls, text="Next ▶", width=80, height=28,
                                   command=lambda: change_image(1))
            next_btn.pack(side=tk.LEFT, padx=5, pady=5)
            
            next_10_btn = ctk.CTkButton(nav_controls, text="+10 ▶▶", width=60, height=28,
                                      command=lambda: change_image(10))
            next_10_btn.pack(side=tk.LEFT, padx=5, pady=5)
            
            # Create a horizontal frame for additional controls
            extra_controls = ctk.CTkFrame(browser_frame, fg_color="transparent")
            extra_controls.pack(fill=tk.X, padx=10, pady=(0, 5))
            
            # Only show toggle button if both original and warped images are available
            if has_original and has_warped:
                # Toggle between original and aligned
                toggle_btn = ctk.CTkButton(
                    extra_controls, 
                    text=f"Show {'Original' if not view_original[0] else 'Aligned'}", 
                    width=120,
                    height=28,
                    command=toggle_view
                )
                toggle_btn.pack(side=tk.LEFT, padx=10, pady=5)
            
            # Save buttons
            save_current = ctk.CTkButton(
                extra_controls, 
                text="Save Current Image", 
                width=120, 
                height=28,
                command=lambda: save_image_as_fits(
                    matrices["all_warped_images"][current_image_idx[0]] if has_warped and not view_original[0] 
                    else matrices["all_original_images"][current_image_idx[0]]
                )
            )
            save_current.pack(side=tk.RIGHT, padx=5, pady=5)
            
            if has_warped:
                save_all = ctk.CTkButton(
                    extra_controls, 
                    text="Save All Aligned Images", 
                    width=160, 
                    height=28,
                    command=lambda: save_all_warped_images_as_fits(matrices["all_warped_images"])
                )
                save_all.pack(side=tk.RIGHT, padx=5, pady=5)
    
    # Function to navigate to previous/next file
    def change_file(delta):
        new_idx = current_file_idx[0] + delta
        if 0 <= new_idx < len(file_keys):
            current_file_idx[0] = new_idx
            current_image_idx[0] = 0  # Reset image index when changing files
            update_display()
    
    # Function to change image in sequence
    def change_image(delta):
        if current_view_mode[0] == 1:  # Only in image browser mode
            file_key = file_keys[current_file_idx[0]]
            matrices = returned_data[file_key]
            
            # Determine which images to use
            if "all_original_images" in matrices and view_original[0]:
                all_images = matrices["all_original_images"]
            elif "all_warped_images" in matrices and not view_original[0]:
                all_images = matrices["all_warped_images"]
            else:
                # Use whatever is available
                all_images = matrices.get("all_original_images", matrices.get("all_warped_images", []))
                
            if not all_images:
                return
                
            new_idx = current_image_idx[0] + delta
            if 0 <= new_idx < len(all_images):
                current_image_idx[0] = new_idx
                update_display()
    
    # Function to toggle view mode
    def toggle_view_mode():
        current_view_mode[0] = 1 - current_view_mode[0]  # Toggle between 0 and 1
        update_display()
    
    # Function to toggle between original and aligned images
    def toggle_view():
        view_original[0] = not view_original[0]
        update_display()
    
    # File navigation buttons
    prev_file_btn = ctk.CTkButton(nav_frame, text="◀ Prev File", width=100, height=28,
                                command=lambda: change_file(-1))
    prev_file_btn.pack(side=tk.LEFT, padx=5, pady=5)
    
    next_file_btn = ctk.CTkButton(nav_frame, text="Next File ▶", width=100, height=28,
                                command=lambda: change_file(1))
    next_file_btn.pack(side=tk.LEFT, padx=5, pady=5)
    
    # View mode buttons
    view_mode_btn = ctk.CTkButton(nav_frame, text="Toggle View Mode", width=140, height=28,
                                command=toggle_view_mode)
    view_mode_btn.pack(side=tk.RIGHT, padx=5, pady=5)
    
    # Initialize display
    update_display()
    
    # Check if we have image data to switch to image browser mode automatically
    file_key = file_keys[0]
    matrices = returned_data[file_key]
    has_images = "all_original_images" in matrices or "all_warped_images" in matrices
    
    if has_images:
        current_view_mode[0] = 1  # Set to image browser mode
        update_display()
    
    # Bring window to front
    window.lift()
    window.focus_set()
    
    return window

def afficher_combinaison(name, value):
    text_zone.config(state="normal")
    text_zone.insert(tk.END, f"{name} = {(int(value[0]),int(value[1]))}\n")
    text_zone.config(state="disabled")

# Class for creating a movable frame
class MovableFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        # Add a handle for moving
        self.handle = ctk.CTkFrame(self, height=25, fg_color=("gray70", "gray30"))
        self.handle.pack(fill="x", side="top")
        
        # Add a label to the handle
        self.title_label = ctk.CTkLabel(self.handle, text="Drag to move")
        self.title_label.pack(side="left", padx=5)
        
        # Add a close button
        self.close_btn = ctk.CTkButton(self.handle, text="×", width=20, height=20, 
                                     command=self.destroy, fg_color="transparent", 
                                     hover_color=("gray80", "gray20"))
        self.close_btn.pack(side="right")
        
        # Content frame
        self.content = ctk.CTkFrame(self)
        self.content.pack(fill="both", expand=True, padx=2, pady=(0, 2))
        
        # Bind events to the handle only, not the entire frame
        self.handle.bind("<Button-1>", self.start_move)
        self.handle.bind("<ButtonRelease-1>", self.stop_move)
        self.handle.bind("<B1-Motion>", self.on_motion)
        
        # Set initial drag state variables
        self.x = None
        self.y = None
        self._dragging = False
        
    def start_move(self, event):
        self.x = event.x_root
        self.y = event.y_root
        self._dragging = True
        
    def stop_move(self, event):
        self._dragging = False
        
    def on_motion(self, event):
        if not self._dragging:
            return
            
        # Calculate the movement delta from root coordinates
        deltax = event.x_root - self.x
        deltay = event.y_root - self.y
        
        # Get current position
        x = self.winfo_x() + deltax
        y = self.winfo_y() + deltay
        
        # Make sure the frame doesn't go off-screen
        max_x = self.master.winfo_width() - 50
        max_y = self.master.winfo_height() - 50
        x = max(0, min(x, max_x))
        y = max(0, min(y, max_y))
        
        # Update position with less frequent screen updates
        self.place(x=x, y=y)
        
        # Store the new reference point
        self.x = event.x_root
        self.y = event.y_root
        
        # Update the window
        self.update_idletasks()
        
    def set_title(self, title):
        self.title_label.configure(text=title)

# Function to create embedded plots
def create_embedded_plot(title="Plot", figsize=(6, 5)):
    """Create a matplotlib figure embedded in a movable frame"""
    # Create a new toplevel window instead of using the main app
    # This prevents rendering issues when moving the frame
    plot_window = ctk.CTkToplevel(app)
    plot_window.title(title)
    plot_window.geometry("800x600")
    plot_window.protocol("WM_DELETE_WINDOW", lambda: close_plot_window(plot_window))
    
    # Create main frame in the window
    main_frame = ctk.CTkFrame(plot_window)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Create figure and canvas
    fig = plt.Figure(figsize=figsize, dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=main_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Add navigation toolbar
    toolbar_frame = ctk.CTkFrame(main_frame)
    toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()
    
    # Add to global list to prevent garbage collection
    plot_figures.append((fig, canvas, plot_window))
    
    # Bring window to front
    plot_window.lift()
    plot_window.focus_set()
    
    return fig, canvas, main_frame

def close_plot_window(window):
    # Find and remove the plot references from the global list
    global plot_figures
    for i, (fig, canvas, win) in enumerate(plot_figures):
        if win == window:
            # Close matplotlib figure to free resources
            plt.close(fig)
            # Remove from our tracking list
            plot_figures.pop(i)
            break
    # Destroy the window
    window.destroy()

# Function to display embedded image with navigation
def display_image_sequence(images, titles=None):
    """Display a sequence of images with navigation buttons"""
    if not images or len(images) == 0:
        messagebox.showinfo("Error", "No images to display")
        return
    
    # Create a toplevel window
    window = ctk.CTkToplevel(app)
    window.title("Image Viewer")
    window.geometry("800x700")
    window.protocol("WM_DELETE_WINDOW", lambda: close_plot_window(window))
    
    # Create main frame
    main_frame = ctk.CTkFrame(window)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Current image index
    current_idx = [0]
    
    # Create figure and canvas
    fig = plt.Figure(figsize=(6, 5), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=main_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Add to global list to prevent garbage collection
    plot_figures.append((fig, canvas, window))
    
    # Navigation buttons frame
    nav_frame = ctk.CTkFrame(main_frame)
    nav_frame.pack(side=tk.BOTTOM, fill=tk.X)
    
    # Save button
    save_btn = ctk.CTkButton(nav_frame, text="Save as FITS", width=100, 
                           command=lambda: save_image_as_fits(images[current_idx[0]]))
    save_btn.pack(side=tk.RIGHT, padx=5, pady=5)
    
    # Info label
    info_label = ctk.CTkLabel(nav_frame, text=f"Image 1/{len(images)}")
    info_label.pack(side=tk.LEFT, padx=20, pady=5)
    
    # Function to update the displayed image
    def update_image():
        fig.clear()
        ax = fig.add_subplot(111)
        im = ax.imshow(images[current_idx[0]])
        if titles and len(titles) > current_idx[0]:
            ax.set_title(titles[current_idx[0]])
        else:
            ax.set_title(f"Image {current_idx[0]+1}/{len(images)}")
        fig.colorbar(im)
        canvas.draw()
        info_label.configure(text=f"Image {current_idx[0]+1}/{len(images)}")
        
    # Function to navigate to the previous image
    def prev_image():
        if current_idx[0] > 0:
            current_idx[0] -= 1
            update_image()
            
    # Function to navigate to the next image
    def next_image():
        if current_idx[0] < len(images) - 1:
            current_idx[0] += 1
            update_image()
    
    # Navigation buttons
    prev_btn = ctk.CTkButton(nav_frame, text="◀", width=40, command=prev_image)
    prev_btn.pack(side=tk.LEFT, padx=5, pady=5)
    
    next_btn = ctk.CTkButton(nav_frame, text="▶", width=40, command=next_image)
    next_btn.pack(side=tk.LEFT, padx=5, pady=5)
    
    # Show the first image
    update_image()
    
    # Bring window to front
    window.lift()
    window.focus_set()
    
    return main_frame

# Function to save all warped images as FITS
def save_all_warped_images_as_fits(warped_images):
    """Save all warped images as a single multi-extension FITS file."""
    if not warped_images or len(warped_images) == 0:
        messagebox.showinfo("Error", "No images to save")
        return False
        
    filename = filedialog.asksaveasfilename(
        defaultextension=".fits",
        filetypes=[("FITS files", "*.fits"), ("All files", "*.*")]
    )
    
    if not filename:  # User canceled the dialog
        return False
    
    try:
        # Convert list to numpy array with correct shape for FITS
        # FITS expects dimensions as (naxis3, naxis2, naxis1)
        warped_array = np.array(warped_images)
        
        # Transpose array to match FITS convention
        if len(warped_array.shape) == 3:
            warped_array = np.transpose(warped_array, (0, 1, 2))
        
        # Create a new FITS file
        hdu = fits.PrimaryHDU(warped_array)
        hdu.writeto(filename, overwrite=True)
        messagebox.showinfo("Success", f"All images saved as {filename}")
        return True
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save FITS file: {str(e)}")
        return False

if __name__ == "__main__":
    # Run the app
    app.mainloop()
