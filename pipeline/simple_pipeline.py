#=======================================#
#     input modules and functions       #
#=======================================#

with open("RESSOURCES\\import_modules.py") as mymodule:
    exec(mymodule.read())

with open("RESSOURCES\\my_pythfunc.py") as mypythfile:
    exec(mypythfile.read())

#plt.style.use(["science","notebook","grid"])


#from alive_progress import alive_bar
from skimage.registration import optical_flow_ilk, optical_flow_tvl1
import cv2 as cv
from skimage.transform import warp
import time

#=======================================#
#              read files               #
#=======================================#

Nobs=4800
mycref=33
Nthumb=80

iim=3000
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
cc2_transl_array = []
cc2_rigid_array = []
cc2_sclrot_array = []
cc2_aff_array = []
cc2_bil_array = []

start_time = time.time()

index_print = 0 #Index for the print loop
prev_cc2 = 0 #To memorize the best (radius, num_warp) couple

index_im = 0

for k in range(0,Nthumb):

    image = unregim[:,:,k]

    sr_transl = StackReg(StackReg.TRANSLATION) #TRANSLATION
    sr_rigid = StackReg(StackReg.RIGID_BODY) #RIGID_BODY
    sr_sclrot = StackReg(StackReg.SCALED_ROTATION) #SCALED_ROTATION
    sr_aff = StackReg(StackReg.AFFINE) #AFFINE
    sr_bil = StackReg(StackReg.BILINEAR) #BILINEAR

    transform_matrix_transl = sr_transl.register(im_ref,image)
    transform_matrix_rigid = sr_rigid.register(im_ref,image)
    transform_matrix_sclrot = sr_sclrot.register(im_ref,image)
    transform_matrix_aff = sr_aff.register(im_ref,image)
    transform_matrix_bil = sr_bil.register(im_ref,image)
    
    im_reg_transl = sr_transl.transform(image)
    im_reg_rigid = sr_rigid.transform(image)
    im_reg_sclrot = sr_sclrot.transform(image)
    im_reg_aff = sr_aff.transform(image)
    im_reg_bil = sr_bil.transform(image)

    cc1 = np.corrcoef(image.flatten(), im_ref.flatten())[0,1]
    cc2_transl = np.corrcoef(im_reg_transl.flatten(), im_ref.flatten())[0,1]
    cc2_rigid = np.corrcoef(im_reg_rigid.flatten(), im_ref.flatten())[0,1]
    cc2_sclrot = np.corrcoef(im_reg_sclrot.flatten(), im_ref.flatten())[0,1]
    cc2_aff = np.corrcoef(im_reg_aff.flatten(), im_ref.flatten())[0,1]
    cc2_bil = np.corrcoef(im_reg_bil.flatten(), im_ref.flatten())[0,1]

    if index_im!=mycref:
        cc1_array.append(cc1)
        cc2_transl_array.append(cc2_transl)
        cc2_rigid_array.append(cc2_rigid)
        cc2_sclrot_array.append(cc2_sclrot)
        cc2_aff_array.append(cc2_aff)
        cc2_bil_array.append(cc2_bil)

    # print(f"{index_im}. cc1 = {cc1:.4f} (REF) || cc2 = {cc2:.4f} || delta = {diff_:.4f}")
    index_im+=1


cc1_mean = np.mean(np.array(cc1_array))

cc2_transl_mean = np.mean(np.array(cc2_transl_array))
cc2_rigid_mean = np.mean(np.array(cc2_rigid_array))
cc2_sclrot_mean = np.mean(np.array(cc2_sclrot_array))
cc2_aff_mean = np.mean(np.array(cc2_aff_array))
cc2_bil_mean = np.mean(np.array(cc2_bil_array))


diff_transl = cc2_transl_mean - cc1_mean
diff_rigid = cc2_rigid_mean - cc1_mean
diff_sclrot = cc2_sclrot_mean - cc1_mean
diff_aff = cc2_aff_mean - cc1_mean
diff_bil = cc2_bil_mean - cc1_mean

print(f"\n===================================================================\n")
print(f"[TRANSLATION] cc1_mean = {cc1_mean:.4f} || cc2_transl_mean = {cc2_transl_mean:.4f} || delta = {diff_transl:.4f}")
print(f"[RIGID BODY] cc1_mean = {cc1_mean:.4f} || cc2_rigid_mean = {cc2_rigid_mean:.4f} || delta = {diff_rigid:.4f}")
print(f"[SCALED ROTATION] cc1_mean = {cc1_mean:.4f} || cc2_sclrot_mean = {cc2_sclrot_mean:.4f} || delta = {diff_sclrot:.4f}")
print(f"[AFFINE] cc1_mean = {cc1_mean:.4f} || cc2_aff_mean = {cc2_aff_mean:.4f} || delta = {diff_aff:.4f}")
print(f"[BILINEAR] cc1_mean = {cc1_mean:.4f} || cc2_bil_mean = {cc2_bil_mean:.4f} || delta = {diff_bil:.4f}\n")

print(f"Translation\nRigid body = translation + rotation\nScaled rotation = translation + rotation + scaling")

stop_time = time.time()
exec_time = stop_time - start_time
print(f"Temps exécution : {exec_time:.3f} secondes\n")



