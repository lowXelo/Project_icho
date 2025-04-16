#from alive_progress import alive_bar
from skimage.registration import optical_flow_ilk
from skimage.transform import warp
from multiprocessing import Queue
from astropy.io import fits
import scipy.signal
import numpy as np
import time
import scipy

#=======================================#
#           Function definition         #
#=======================================#

def find_outliers_2D(data, pclow=None, pchigh=None, thresh=None):
    # Set default percentile thresholds
    if pclow is None: pclow = 25
    if pchigh is None: pchigh = 75
    if thresh is None: thresh = 1.5

    # Calculate interquartile range (IQR)
    q1 = np.percentile(data.flatten(), pclow)
    q3 = np.percentile(data.flatten(), pchigh)
    iqr = q3 - q1

    # Define outlier bounds
    lower_fence = q1 - thresh * iqr
    upper_fence = q3 + thresh * iqr

    # Identify outlier and non-outlier indices
    outliers = np.argwhere((data < lower_fence) | (data > upper_fence))
    no_outliers = np.argwhere((data >= lower_fence) & (data <= upper_fence))
    return outliers, no_outliers

#=======================================#
#              read files               #
#=======================================#
def v1_pipelin_youness(les_options, queue, result_queue, file_unregistered, nump_warp_value, radius_value):
    mycref = 33  # Reference frame index
    Nthumb = 80  # Number of image slices
    progress = 0
    returned_data = {}

    for file_index, i in enumerate(file_unregistered, start=1):  
        file_key = f"file {file_index}"
        returned_data[file_key] = {}

        # Load FITS file and format image shape
        hdul = fits.open(i)
        unregim = hdul[0].data.astype(float)
        unregim = np.transpose(unregim, (1, 2, 0))  # Shape: (H, W, T)

        # Apply median filtering and outlier replacement
        for tt in np.arange(Nthumb):
            medfi = scipy.signal.medfilt2d(unregim[:, :, tt])
            outl, noout = find_outliers_2D(unregim[:, :, tt], pclow=10, pchigh=90)
            for oo in np.arange(len(outl)):
                unregim[outl[oo][0], outl[oo][1], tt] = medfi[outl[oo][0], outl[oo][1]]

        # Set reference image
        im_ref = unregim[:, :, mycref]
        cc1_array = []
        cc2_array = []
        RADIUS = range(1, radius_value)
        NUMP_WARP = range(1, nump_warp_value)
        time_exe = []
        total_iterations = len(RADIUS) * len(NUMP_WARP)

        progress += 1  # Initial progress update

        # Loop over radius and num_warp parameters
        for i in RADIUS:
            for j in NUMP_WARP:
                progress += 1
                queue.put(progress / (total_iterations * len(file_unregistered)) * 100)

                cc1_new = []
                cc2_new = []

                if les_options[3] == 1: t1 = time.time()  # Timing if option is set

                # Register each image in the stack to reference
                for k in range(Nthumb):
                    image = unregim[:, :, k]
                    v, u = optical_flow_ilk(im_ref, image, radius=i, num_warp=j, prefilter=True)
                    nr, nc = im_ref.shape
                    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
                    image1_warp = warp(image, np.array([row_coords + v, col_coords + u]), mode='reflect')

                    # Compute correlation before/after warp
                    cc1 = np.corrcoef(image.flatten(), im_ref.flatten())[0, 1]
                    cc2 = np.corrcoef(image1_warp.flatten(), im_ref.flatten())[0, 1]

                    if k != mycref:  # Skip reference slice
                        cc1_new.append(cc1)
                        cc2_new.append(cc2)

                cc1_array.append(cc1_new)
                cc2_array.append(cc2_new)

                if les_options[3] == 1:
                    t2 = time.time()
                    time_exe.append(t2 - t1)

        # Convert to arrays and store
        cc1_array = np.array(cc1_array)
        cc2_array = np.array(cc2_array)
        data = [cc2_array[i] for i in range(len(cc1_array))]
        returned_data[file_key]["data"] = data

        # Extract statistical matrices
        list_des_moyenne = []
        list_des_ecarts_max = []

        for i in range(len(cc1_array)):
            if les_options[1] == 1:
                list_des_moyenne.append(np.mean(cc2_array[i]))
            if les_options[2] == 1:
                max_dev = abs(max(cc2_array[i]) - np.mean(cc2_array[i]))
                min_dev = abs(min(cc2_array[i]) - np.mean(cc2_array[i]))
                list_des_ecarts_max.append(max(max_dev, min_dev) ** 2)

        # Store matrices based on options
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

        # Store optimal parameter combination
        returned_data[file_key]["Best combination"] = best_index

    result_queue.put(returned_data)
    queue.put(100)  # Mark as done
  
    
def v1_direct(les_options, queue, result_queue, file_unregistered, nump_warp_value=2, radius_value=2):
    progress = 0
    queue.put(progress)
    mycref = 33
    Nthumb = 80
    returned_data = {}

    for file_index, file_path in enumerate(file_unregistered, start=1):  
        file_key = f"file {file_index}"
        returned_data[file_key] = {}

        # Load image
        hdul = fits.open(file_path)
        unregim = hdul[0].data.astype(float)
        unregim = np.transpose(unregim, (1, 2, 0))

        # Preprocess with median filtering and outlier removal
        for tt in range(Nthumb):
            progress += 1
            queue.put(progress / 160 * 100)
            medfi = scipy.signal.medfilt2d(unregim[:, :, tt])
            outl, _ = find_outliers_2D(unregim[:, :, tt], pclow=10, pchigh=90)
            for oo in outl:
                unregim[oo[0], oo[1], tt] = medfi[oo[0], oo[1]]

        im_ref = unregim[:, :, mycref]
        cc1_array = []
        cc2_array = []

        # Register each image
        for k in range(Nthumb):
            progress += 1
            queue.put(progress / 160 * 100)
            image = unregim[:, :, k]
            v, u = optical_flow_ilk(im_ref, image, radius=radius_value, num_warp=nump_warp_value, prefilter=True)
            nr, nc = im_ref.shape
            row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
            image1_warp = warp(image, np.array([row_coords + v, col_coords + u]), mode='reflect')

            # Correlation before and after
            cc1 = np.corrcoef(image.flatten(), im_ref.flatten())[0, 1]
            cc2 = np.corrcoef(image1_warp.flatten(), im_ref.flatten())[0, 1]
            if k != mycref:
                cc1_array.append(cc1)
                cc2_array.append(cc2)

        # Final warped image for preview
        image1_warp = warp(unregim[:, :, 0], np.array([row_coords + v, col_coords + u]), mode='reflect')

        # Save outputs
        returned_data[file_key]["data_old"] = cc1_array
        returned_data[file_key]["data_new"] = cc2_array
        returned_data[file_key]["image_before"] = im_ref
        returned_data[file_key]["image_after"] = image1_warp

    result_queue.put(returned_data)
    queue.put(100)

