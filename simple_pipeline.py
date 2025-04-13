#=======================================#
#     input modules and functions       #
#=======================================#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from astropy.io import fits
import time
import scipy
from pystackreg import StackReg
from multiprocessing import Queue

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

#plt.style.use(["science","notebook","grid"])

#from alive_progress import alive_bar
from skimage.registration import optical_flow_ilk, optical_flow_tvl1
import cv2 as cv
from skimage.transform import warp,AffineTransform,ProjectiveTransform
import time

#=======================================#
#              read files               #
#=======================================#
# def v_simple(queue, result_queue, file_unregistered, rotation, translation,scale,sour,dest):
#     Nobs=4800
#     mycref=33
#     Nthumb=80
#     iim=3000
#     returned_data={}
#     progress=0
#     for file_index, file_path in enumerate(file_unregistered, start=1):
#         progress+=1
#         queue.put(progress/(Nthumb*len(file_unregistered))*100)


#         file_key = f"file {file_index}"
#         returned_data[file_key] = []  
#         hdul = fits.open(file_unregistered)
#         unregim=hdul[0].data.astype(float)
#         unregim=np.transpose(unregim,(1,2,0))

#         for tt in np.arange(Nthumb):
#             medfi=scipy.signal.medfilt2d(unregim[:,:,tt])
#             outl,noout=find_outliers_2D(unregim[:,:,tt],pclow=10, pchigh=90)
#             for oo in np.arange(len(outl)):
#                 unregim[outl[oo][0],outl[oo][1],tt]=medfi[outl[oo][0],outl[oo][1]]


#         im_ref = unregim[:,:,mycref]

#         cc1_array = []
#         cc2_transl_array = []
#         cc2_rigid_array = []
#         cc2_sclrot_array = []
#         cc2_aff_array = []
#         cc2_bil_array = []

#         start_time = time.time()


#         index_im = 0

#         for k in range(0,Nthumb):
#             progress+=1
#             queue.put(progress/(Nthumb*len(file_unregistered))*100)

#                         # You must define these based on your use case
#             tx, ty = translation[0], translation[1]       # Translation
#             theta = np.deg2rad(rotation)       # Rotation in radians
#             scale_x, scale_y = scale[0], scale[1]  # Scaling

#             # TRANSLATION
#             tform_transl = AffineTransform(translation=(tx, ty))
#             im_reg_transl = warp(image, tform_transl.inverse)

#             # RIGID (translation + rotation)
#             tform_rigid = AffineTransform(translation=(tx, ty), rotation=theta)
#             im_reg_rigid = warp(image, tform_rigid.inverse)

#             # SCALED ROTATION (scaling + rotation)
#             tform_sclrot = AffineTransform(scale=(scale_x, scale_y), rotation=theta)
#             im_reg_sclrot = warp(image, tform_sclrot.inverse)

#             # AFFINE (translation + rotation + scaling)
#             tform_aff = AffineTransform(scale=(scale_x, scale_y), rotation=theta, translation=(tx, ty))
#             im_reg_aff = warp(image, tform_aff.inverse)

#             # BILINEAR / PERSPECTIVE (more complex deformation)
#             src = sour
#             dst = dest
#             tform_bil = ProjectiveTransform()
#             tform_bil.estimate(src, dst)
#             im_reg_bil = warp(image, tform_bil.inverse)

#             cc1 = np.corrcoef(image.flatten(), im_ref.flatten())[0,1]
#             cc2_transl = np.corrcoef(im_reg_transl.flatten(), im_ref.flatten())[0,1]
#             cc2_rigid = np.corrcoef(im_reg_rigid.flatten(), im_ref.flatten())[0,1]
#             cc2_sclrot = np.corrcoef(im_reg_sclrot.flatten(), im_ref.flatten())[0,1]
#             cc2_aff = np.corrcoef(im_reg_aff.flatten(), im_ref.flatten())[0,1]
#             cc2_bil = np.corrcoef(im_reg_bil.flatten(), im_ref.flatten())[0,1]

#             if index_im!=mycref:
#                 cc1_array.append(cc1)
#                 cc2_transl_array.append(cc2_transl)
#                 cc2_rigid_array.append(cc2_rigid)
#                 cc2_sclrot_array.append(cc2_sclrot)
#                 cc2_aff_array.append(cc2_aff)
#                 cc2_bil_array.append(cc2_bil)

#             # print(f"{index_im}. cc1 = {cc1:.4f} (REF) || cc2 = {cc2:.4f} || delta = {diff_:.4f}")
#             index_im+=1


#         cc1_mean = np.mean(np.array(cc1_array))

#         cc2_transl_mean = np.mean(np.array(cc2_transl_array))
#         cc2_rigid_mean = np.mean(np.array(cc2_rigid_array))
#         cc2_sclrot_mean = np.mean(np.array(cc2_sclrot_array))
#         cc2_aff_mean = np.mean(np.array(cc2_aff_array))
#         cc2_bil_mean = np.mean(np.array(cc2_bil_array))


#         diff_transl = cc2_transl_mean - cc1_mean
#         diff_rigid = cc2_rigid_mean - cc1_mean
#         diff_sclrot = cc2_sclrot_mean - cc1_mean
#         diff_aff = cc2_aff_mean - cc1_mean
#         diff_bil = cc2_bil_mean - cc1_mean

#         print(f"\n===================================================================\n")
#         print(f"[TRANSLATION] cc1_mean = {cc1_mean:.4f} || cc2_transl_mean = {cc2_transl_mean:.4f} || delta = {diff_transl:.4f}")
#         print(f"[RIGID BODY] cc1_mean = {cc1_mean:.4f} || cc2_rigid_mean = {cc2_rigid_mean:.4f} || delta = {diff_rigid:.4f}")
#         print(f"[SCALED ROTATION] cc1_mean = {cc1_mean:.4f} || cc2_sclrot_mean = {cc2_sclrot_mean:.4f} || delta = {diff_sclrot:.4f}")
#         print(f"[AFFINE] cc1_mean = {cc1_mean:.4f} || cc2_aff_mean = {cc2_aff_mean:.4f} || delta = {diff_aff:.4f}")
#         print(f"[BILINEAR] cc1_mean = {cc1_mean:.4f} || cc2_bil_mean = {cc2_bil_mean:.4f} || delta = {diff_bil:.4f}\n")

#         print(f"Translation\nRigid body = translation + rotation\nScaled rotation = translation + rotation + scaling")

#         stop_time = time.time()
#         exec_time = stop_time - start_time
#         print(f"Temps exécution : {exec_time:.3f} secondes\n")
#         returned_data[file_key]=[cc1_mean,cc2_transl_mean,cc2_rigid_mean,cc2_sclrot_mean,cc2_aff_mean,cc2_bil_mean,exec_time]
#     result_queue.put(returned_data)
#     queue.put(100)
    


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
    Nobs = 4800
    mycref = 33
    Nthumb = 80
    iim = 3000
    returned_data = {}
    progress = 0
    src=[[100, 100], [200, 100], [200, 200], [100, 200]]
    dst=corners
    for file_index, file_path in enumerate(file_unregistered, start=1):
        progress += 1
        queue.put(progress / (Nthumb * len(file_unregistered)) * 100)

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
            print(k)
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
