import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.registration import optical_flow_ilk, optical_flow_tvl1
import cv2 as cv
from skimage.transform import warp

# Générer une image de base (un carré blanc sur fond noir)
image_size = (200, 200)
image = np.zeros(image_size, dtype=np.uint8)
cv2.rectangle(image, (50, 50), (150, 150), 255, -1)

# Définir une matrice de transformation (translation + rotation)
angle = 20  # Rotation en degrés
tx, ty = 30, 20  # Translation en pixels

# Calculer la matrice de transformation
center = (image_size[1] // 2, image_size[0] // 2)
rotation_matrix = cv2.getRotationMatrix2D(center,angle, 1.0)
rotation_matrix[:, 2] += [tx, ty]  # Ajouter la translation

# Appliquer la transformation
transformed_image = cv2.warpAffine(image, rotation_matrix, (image_size[1], image_size[0]))

v, u = optical_flow_ilk(image, transformed_image,radius = 2,num_warp= 40,prefilter = True)

nr, nc = image.shape

row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')

image1_warp = warp(image, np.array([row_coords + v, col_coords + u]), mode='reflect')

cc1 = np.corrcoef(image.flatten(), transformed_image.flatten())[0,1]
cc2 = np.corrcoef(image.flatten(), image1_warp.flatten())[0,1]

# Affichage des images
fig, ax = plt.subplots(1, 3, figsize=(8, 4))
ax[0].imshow(image, cmap='gray')
ax[0].set_title("Image originale")
ax[0].axis("off")

ax[1].imshow(transformed_image, cmap='gray')
ax[1].set_title("Image transformée (rotation + translation)")
ax[1].axis("off")

ax[2].imshow(image1_warp, cmap='gray')
ax[2].set_title("Image recallé (rotation + translation)")
ax[2].axis("off")
print(cc1,cc2)
plt.show()

