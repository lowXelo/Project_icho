def optimize_ilk_params(unregim, mycref, Nthumb, N_RADIUS, N_NW, graph_mode):

    im_ref = unregim[:,:,mycref]

    RADIUS = [i for i in range(1,N_RADIUS+1)]
    NUMP_WARP = [i for i in range(1,N_NW+1)]  

    cc1_array ,cc2_array = [], []
    best_i, best_j, prev_cc2, index_print = 0, 0, 0, 0


    start_time = time.time()
    print(f"\n==== Nb. radius: {N_RADIUS} || Nb. num_warp : {N_NW} // Expect {(N_RADIUS)*(N_NW)} iterations === \n")

    for i in RADIUS:
        for j in NUMP_WARP:
            cc1_new, cc2_new = [], []
            
            for k in range(0,Nthumb):
                image = unregim[:,:,k]
                v, u = optical_flow_ilk(im_ref, image,
                                            radius = i,
                                            num_warp= j,
                                            prefilter = True)
                nr, nc = im_ref.shape
                row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
                image_warped = warp(image, np.array([row_coords + v, col_coords + u]), mode='reflect')

                cc1 = np.corrcoef(image.flatten(), im_ref.flatten())[0,1]
                cc2 = np.corrcoef(image_warped.flatten(), im_ref.flatten())[0,1]

                if k!=mycref:
                    cc1_new.append(cc1)
                    cc2_new.append(cc2)

                
            cc1_array.append(cc1_new)
            cc2_array.append(cc2_new)

            cc1_mean, cc2_mean = np.mean(np.array(cc1_new)), np.mean(np.array(cc2_new)) 
            if cc2_mean > prev_cc2:
                prev_cc2 = cc2_mean
                best_i = i
                best_j = j
                best_diff = prev_cc2 - cc1_mean

            diff = cc2_mean - cc1_mean

            #print(f"{index_print+1}. cc1_mean = {cc1_mean:.4f} (REF) || cc2_mean = {cc2_mean:.4f} || delta = {diff:.4f} || PARAMS : radius = {i} / nump_warp = {j}")
            index_print+=1

    cc1_array = np.array(cc1_array)
    cc2_array = np.array(cc2_array)


    cc2_mean_array = np.array([np.mean(sublist) for sublist in cc2_array])
    cc2_stds_array = np.array([np.std(sublist) for sublist in cc2_array])

    _ = np.array([i for i in range(1,len(cc2_mean_array)+1)])

    print(f"\n=================================================================\n")
    print(f"Best mean perf given for (i,j) = ({best_i},{best_j}) || cc2_mean = {prev_cc2:.4f} (evolution : {best_diff:.4f})\n")

    stop_time = time.time()
    exec_time = stop_time - start_time
    print(f"Temps exécution : {exec_time:.3f} secondes")


    if (graph_mode == "boxplot"):
        #BOX PLOT
        plt.figure()
        plt.boxplot(cc2_array.T, vert=True, patch_artist=True)
        plt.title("Box Plot des données")
        plt.ylabel("Valeurs")
        plt.xlabel("Index des séries")

    elif (graph_mode == "errorbar"):
        #ERRORBAR PLOT
        plt.figure()
        plt.plot(_,cc2_mean_array*100, 'x-',color="crimson",lw="2")
        plt.errorbar(_,cc2_mean_array*100,yerr=cc2_stds_array*100,color="black",fmt="None",capsize=5,elinewidth=0.8)
        plt.title("Graphique de l'évolution du coefficient de corrélation \nen fonction des couples de paramètres (méthode ilk)", pad = 20)
        plt.ylabel("Coefficient de corrélation (%)", labelpad = 15,fontsize=15)
        plt.xlabel("Index array", labelpad = 15,fontsize=15)
        plt.grid(which = "major", alpha = 0.9)
        plt.grid(which = "minor", alpha = 0.3)
        plt.show()

    radius, num_warp = best_i, best_j
    return radius, num_warp


def load_fits(file_path):
    hdul = fits.open(file_path)
    data = hdul[0].data.astype(float)
    
    unregim = np.transpose(data, (1, 2, 0))

    for tt in np.arange(Nthumb):
        medfi=scipy.signal.medfilt2d(unregim[:,:,tt])
        outl,noout=find_outliers_2D(unregim[:,:,tt],pclow=10, pchigh=90)
        for oo in np.arange(len(outl)):
            unregim[outl[oo][0],outl[oo][1],tt]=medfi[outl[oo][0],outl[oo][1]]
    
    return unregim


def apply_ilk(unregim, mycref, radius, num_warp):
    im_ref = unregim[:, :, mycref]
    cc1_array, cc2_array = [], []

    for k in range(0,Nthumb):
                image = unregim[:,:,k]
                v, u = optical_flow_ilk(im_ref, image,
                                            radius = radius,
                                            num_warp= num_warp,
                                            prefilter = True)
                nr, nc = im_ref.shape
                row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
                image_warped = warp(image, np.array([row_coords + v, col_coords + u]), mode='reflect')

                cc1 = np.corrcoef(image.flatten(), im_ref.flatten())[0,1]
                cc2 = np.corrcoef(image_warped.flatten(), im_ref.flatten())[0,1]

                if k!=mycref:
                    cc1_array.append(cc1)
                    cc2_array.append(cc2)
    
    cc1_mean = np.mean(np.array(cc1_array))
    cc2_mean = np.mean(np.array(cc2_array))
    diff = cc2_mean - cc1_mean

    print(f"cc1_mean = {cc1_mean:.4f} || cc2__mean = {cc2_mean:.4f} || delta = {diff:.4f}")

    return image_warped


if __name__ == "__main__":

    #=======================================#
    #     input modules and functions       #
    #=======================================#

    with open("RESSOURCES\\import_modules.py") as mymodule:
        exec(mymodule.read())

    with open("RESSOURCES\\my_pythfunc.py") as mypythfile:
        exec(mypythfile.read())

    plt.style.use(["science","notebook","grid"])
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

    path_1 = "SCARBO\\L1a_images_cube10.fits"
    path_2 = "SCARBO\\L1a_images_cube11.fits"
    path_3 = "SCARBO\\L1a_images_cube12.fits"
    path_4= "SCARBO\\L1a_images_cube13.fits"
    path_5 = "SCARBO\\L1a_images_cube14.fits"
    path_6 = "SCARBO\\L1a_images_cube15.fits"
    path_7 = "SCARBO\\L1a_images_cube16.fits"
    path_8 = "SCARBO\\L1a_images_cube17.fits"
    path_9 = "SCARBO\\L1a_images_cube18.fits"
    path_10 = "SCARBO\\L1a_images_cube19.fits"

    path_ = [path_1, path_2, path_3, path_4, path_5, path_6, path_7, path_8, path_9, path_10]
   
    N_RADIUS = 3
    N_NW = 10

    imreg_array = []

    for path in path_:
        unregim = load_fits(path)
        # radius, num_warp = optimize_ilk_params(unregim, mycref, Nthumb, N_RADIUS, N_NW, graph_mode=None)
        im_w = apply_ilk(unregim, mycref, radius=2, num_warp=10)
        imreg_array.append(im_w)
    
    print(len(imreg_array))

    # for im in imreg_array:
    #     plt.figure()
    #     plt.imshow(im)
    #     plt.title("Set of thumbnail images")
    #     plt.show()






