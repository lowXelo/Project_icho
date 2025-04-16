#=======================================#
#     input modules and functions       #
#=======================================#
import numpy as np
from astropy.io import fits
import scipy
from multiprocessing import Queue
from skimage.transform import warp,AffineTransform,ProjectiveTransform

def find_outliers_2D(data, pclow=None, pchigh=None, thresh=None):
    if pclow is None:
        pclow=25
    if pchigh is None:
        pchigh=75
    if thresh is None:
        thresh=1.5
    q1 = np.percentile(data.flatten(), pclow)
    q3 = np.percentile(data.flatten(), pchigh)
    iqr = q3 - q1
    lower_fence = q1 - thresh * iqr
    upper_fence = q3 + thresh * iqr
    outliers = np.argwhere((data < lower_fence) | (data > upper_fence))
    no_outliers = np.argwhere((data >= lower_fence) & (data <= upper_fence))
    return outliers, no_outliers


def get_transform_params(original_corners, transformed_corners):
    original_corners = np.array(original_corners)
    transformed_corners = np.array(transformed_corners)

    center_original = np.mean(original_corners, axis=0)
    center_transformed = np.mean(transformed_corners, axis=0)

    translation = center_transformed - center_original

    # Compute rotation using two points
    angle_original = np.arctan2(original_corners[1, 1] - center_original[1],
                                original_corners[1, 0] - center_original[0])
    angle_transformed = np.arctan2(transformed_corners[1, 1] - center_transformed[1],
                                   transformed_corners[1, 0] - center_transformed[0])
    rotation = np.rad2deg(angle_transformed - angle_original)

    # Scaling
    dist_original = np.linalg.norm(original_corners[0] - original_corners[1])
    dist_transformed = np.linalg.norm(transformed_corners[0] - transformed_corners[1])
    scale_factor = dist_transformed / dist_original

    return translation, rotation, (scale_factor, scale_factor)

# Main function
def v_simple(queue, result_queue, file_unregistered,corners):
    mycref = 33
    Nthumb = 80
    returned_data = {}
    progress = 0
    src = [[50, 50], [150, 50], [150, 150], [50, 150]]
    dst=corners
    for file_index, file_path in enumerate(file_unregistered, start=1):
        progress += 1
        queue.put(progress * 100 / (Nthumb * len(file_unregistered)) )

        file_key = f"file {file_index}"
        returned_data[file_key] = []

        hdul = fits.open(file_path)
        unregim = hdul[0].data.astype(float)
        unregim = np.transpose(unregim, (1, 2, 0))
        

        # Process each image (Nthumb)
        for tt in np.arange(Nthumb):
            # Process to remove outliers
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
            progress += 1
            queue.put(progress*100 / (Nthumb * len(file_unregistered)) )
            image = unregim[:, :, k]
            # Use get_transform_params for corners (or approximate them if not defined)
            original_corners = src  # Example corners
            transformed_corners = dst

            # Get the transform parameters (translation, rotation, scale)
            translation, rotation, scale = get_transform_params(original_corners, transformed_corners)

            # Apply transformations (example with affine)
            tx, ty = translation
            theta = np.deg2rad(rotation)  # Rotation in radians
            scale_x, scale_y = scale  # Assuming uniform scale for simplicity

            # Apply the transformations (translation, rigid, scaled rotation, affine, bilinear)
            tform_transl = AffineTransform(translation=(tx, ty))
            im_reg_transl = warp(image, tform_transl.inverse)

            tform_rigid = AffineTransform(translation=(tx, ty), rotation=theta)
            im_reg_rigid = warp(image, tform_rigid.inverse)

            tform_sclrot = AffineTransform(scale=(scale_x, scale_y), rotation=theta)
            im_reg_sclrot = warp(image, tform_sclrot.inverse)

            tform_aff = AffineTransform(scale=(scale_x, scale_y), rotation=theta, translation=(tx, ty))
            im_reg_aff = warp(image, tform_aff.inverse)

            # Bilinear transformation (perspective)
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
            
        returned_data[file_key] = [np.mean(cc1_array),np.mean(cc2_array),np.mean(cc3_array),np.mean(cc4_array),np.mean(cc5_array),np.mean(cc6_array)]
    result_queue.put(returned_data)
    queue.put(100)
    print('fin')
