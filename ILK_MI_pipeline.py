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

mi1_array = []
mi2_array = []

N_RADIUS = 5
N_NW = 10

RADIUS = [i for i in range(1,N_RADIUS+1)]
NUMP_WARP = [i for i in range(1,N_NW+1)]  

start_time = time.time()

best_i = 0 #Used to remember the (radius,num_warp) giving the best cc perf in mean
best_j = 0

index_print = 0 #Index for the print loop
prev_cc2 = 0 #To memorize the best (radius, num_warp) couple

print(f"\n==== Nb. radius: {N_RADIUS} || Nb. num_warp : {N_NW} // Expect {(N_RADIUS)*(N_NW)} iterations === \n")
for i in RADIUS:
    for j in NUMP_WARP:
        mi1_new = []
        mi2_new = []
        index_im = 0
        
        for k in range(0,Nthumb):

            image = unregim[:,:,k]

            v, u = optical_flow_ilk(im_ref, image,
                                        radius = i,
                                        num_warp= j,
                                        prefilter = True)
            
            nr, nc = im_ref.shape

            row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
            image1_warp = warp(image, np.array([row_coords + v, col_coords + u]), mode='reflect')

            mi1 = mutual_info_matrix(np.stack([image.flatten(),im_ref.flatten()],axis=1))[0,1]
            mi2 = mutual_info_matrix(np.stack([image1_warp.flatten(),im_ref.flatten()],axis=1))[0,1]

            if index_im!=mycref:
                mi1_new.append(mi1)
                mi2_new.append(mi2)

            # print(index,cc1,cc2)
            index_im+=1

        mi1_array.append(mi1_new)
        mi2_array.append(mi2_new)

        

        mi1_mean = np.mean(np.array(mi1_new))
        mi2_mean = np.mean(np.array(mi2_new))
        if mi2_mean > prev_cc2:
            prev_cc2 = mi2_mean
            best_i = i
            best_j = j
            best_diff = prev_cc2 - mi1_mean

        diff = mi2_mean - mi1_mean

        print(f"{index_print+1}. mi1_mean = {mi1_mean:.4f} (REF) || mi2_mean = {mi2_mean:.4f} || evo = {diff:.4f} || PARAMS : radius = {i} / nump_warp = {j}")
        index_print+=1


mi1_array = np.array(mi1_array)
mi2_array = np.array(mi2_array)


mi2_mean_array = np.array([np.mean(sublist) for sublist in mi2_array])
mi2_stds_array = np.array([np.std(sublist) for sublist in mi2_array])

_ = np.array([i for i in range(1,len(mi2_mean_array)+1)])


print(f"\n=================================================================\n")
print(f"Best mean perf given for (i,j) = ({best_i},{best_j}) || mi2_mean = {prev_cc2:.4f} (evolution : {best_diff:.4f})\n")

stop_time = time.time()

exec_time = stop_time - start_time
print(f"Temps exécution : {exec_time:.3f} secondes")

# Tracer le box plot
plt.figure()
plt.boxplot(mi2_array.T, vert=True, patch_artist=True)
plt.title("Box Plot des données")
plt.ylabel("Valeurs")
plt.xlabel("Index des séries")

plt.figure()
plt.plot(_,mi2_mean_array*100, 'x-',color="crimson",lw="2")
plt.errorbar(_,mi2_mean_array*100,yerr=mi2_stds_array*100,color="black",fmt="None",capsize=5,elinewidth=0.8)
plt.title("Graphique de l'évolution de l'information mutuelle \nen fonction des couples de paramètres (méthode ilk)", pad = 20)
plt.ylabel("Coefficient de corrélation (%)", labelpad = 15,fontsize=15)
plt.xlabel("Index array", labelpad = 15,fontsize=15)
plt.grid(which = "major", alpha = 0.9)
plt.grid(which = "minor", alpha = 0.3)
plt.show()


