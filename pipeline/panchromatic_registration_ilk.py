#=======================================#
#     input modules and functions       #
#=======================================#

with open("RESSOURCES\\import_modules.py") as mymodule:
    exec(mymodule.read())

with open("RESSOURCES\\my_pythfunc.py") as mypythfile:
    exec(mypythfile.read())

from skimage.registration import optical_flow_ilk
import cv2 as cv
from skimage.transform import warp
import time
##############################################

def panchromatic_regristration_ilk(path: str, radius: int, num_warp: int, ref:int = 0):
    '''Register all panchromatic images contains in an array using optical_flow_ilk method. 
    
    Parameters
    -----------------
    path : str
        path of the .npy file containing the array of panchromatic images
    
    radius : int
        parameter for the optical_flow_ilk registration
    
    num_warp: int
        parameter for the optical_flow_ilk registration
    
    ref: int
        array index of the panchromatic image used as ref for registration. default value to 0
    
    Return
    ----------------
    panchromatic_array_registered : numpy.ndarray
        array containing all the registered panchromatic images

    '''

    panchromatic_array_registered = []
    panchromatic_array = np.load(path)
    im_ref = panchromatic_array[ref]

    l = panchromatic_array.shape
    
    start_time = time.time()

    for k in range(0,l[0]):
        if k == ref:
            panchromatic_array_registered.append(panchromatic_array[k,:,:])

        elif k!=ref:
            image = panchromatic_array[k,:,:]

            cc1 = np.corrcoef(image.flatten(), im_ref.flatten())[0,1]
            mi1 = mutual_info_matrix(np.stack([image.flatten(),im_ref.flatten()],axis=1))[0,1]
            

            v, u = optical_flow_ilk(im_ref, image, radius = 2, num_warp=10, prefilter=True)
            nr, nc = im_ref.shape
            row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
            image_warped = warp(image, np.array([row_coords + v, col_coords + u]), mode='reflect')

            
            cc2 = np.corrcoef(image_warped.flatten(), im_ref.flatten())[0,1]
            diff_cc = cc2 - cc1
            mi2 = mutual_info_matrix(np.stack([image_warped.flatten(),im_ref.flatten()],axis=1))[0,1]
            diff_mi = mi2 - mi1
            
            print(f"\nImage panchromatique {k}")
            print(f"CC1 = {cc1:.4f} (REF)|| CC2 = {cc2:.4f} || delta = {100*diff_cc:.2f} %")
            print(f"MI1 = {mi1:.4f} (REF)|| MI2 = {mi2:.4f} || delta = {100*diff_mi:.2f} %\n")

            panchromatic_array_registered.append(image_warped)
    
    
    end_time = time.time()
    print(f"execution time : {(end_time - start_time):.4f}")
    return np.array(panchromatic_array_registered)
        

if __name__ == "__main__":
    regpan = panchromatic_regristration_ilk("panchro_array/panchro_array.npy", radius = 2, num_warp=10)
    print(regpan.shape)
    print(regpan)
    