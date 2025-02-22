#=======================================#
#     input modules and functions       #
#=======================================#

with open("RESSOURCES\\import_modules.py") as mymodule:
    exec(mymodule.read())

with open("RESSOURCES\\my_pythfunc.py") as mypythfile:
    exec(mypythfile.read())


#from alive_progress import alive_bar
from skimage.registration import optical_flow_ilk, optical_flow_tvl1
import cv2 as cv
from skimage.transform import warp

#=======================================#
#              read files               #
#=======================================#

Nobs=4800
mycref=33
Nthumb=80

iim=2000
file_unregistered = "C:\\Users\\valen\\Desktop\\ICHO\\images\\L1a_images_cube2000.fits"
#file_unregistered="L1a_images_cube"+str(iim)+".fits"

hdul = fits.open(file_unregistered)
unregim=hdul[0].data.astype(float)
unregim=np.transpose(unregim,(1,2,0))

for tt in np.arange(Nthumb):
    medfi=scipy.signal.medfilt2d(unregim[:,:,tt])
    outl,noout=find_outliers_2D(unregim[:,:,tt],pclow=10, pchigh=90)
    for oo in np.arange(len(outl)):
        unregim[outl[oo][0],outl[oo][1],tt]=medfi[outl[oo][0],outl[oo][1]]


#=======================================#
#              Plot thumbnail           #
#=======================================#

# plt.figure()
# plt.imshow(from3to2(unregim,8,10))
# plt.title("Set of thumbnail images")
# plt.show()

#=======================================#
#    Registration and performances      #
#=======================================#

im_ref = unregim[:,:,mycref]

cc1_array = []
cc2_array = []

RADIUS = [i for i in range(1,5)]
NUMP_WARP = [i for i in range(1,3)]  

prev_cc2 = 0
best_i = 0
best_j = 0

for i in RADIUS:
    for j in NUMP_WARP:
        cc1_new = []
        cc2_new = []
        index = 0
        for k in range(0,Nthumb):

            image = unregim[:,:,k]

            v, u = optical_flow_ilk(im_ref, image,
                                        radius = i,
                                        num_warp= j,
                                        prefilter = True)
            
            nr, nc = im_ref.shape

            row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')

            image1_warp = warp(image, np.array([row_coords + v, col_coords + u]), mode='reflect')

            cc1 = np.corrcoef(image.flatten(), im_ref.flatten())[0,1]
            cc2 = np.corrcoef(image1_warp.flatten(), im_ref.flatten())[0,1]

            if index!=mycref:
                cc1_new.append(cc1)
                cc2_new.append(cc2)

            if cc2 > prev_cc2:
                prev_cc2 = cc2
                best_i = i
                best_j = j

        
            print(f"cc1 = {cc1} (REF) || cc2 = {cc2} || PARAMS : radius = {i} / nump_warp = {j}")
            # print(index,cc1,cc2)
            # index+=1

    cc1_array.append(cc1_new)
    cc2_array.append(cc2_new)

cc1_array = np.array(cc1_array)
cc2_array = np.array(cc2_array)

print(cc2_array)

for i in range(len(cc1_array)):
    data = [cc2_array[i]]

# Tracer le box plot
plt.boxplot(data, vert=True, patch_artist=True)  # vert=True pour un boxplot vertical, patch_artist=True pour colorier
plt.title("Box Plot des données")
plt.ylabel("Valeurs")
plt.show()

    
