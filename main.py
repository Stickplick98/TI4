import cv2
import numpy as np

# Charger l'image
image = cv2.imread("C:\\TraitementImages3eme\\Images\\balanes.png", cv2.IMREAD_GRAYSCALE)

# Appliquer un seuillage adaptatif pour obtenir une image binaire
_, binary_image = cv2.threshold(image, 10, 255, cv2.THRESH_OTSU)

# Trouver les contours dans l'image binaire
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Définir une taille minimale et maximale pour les objets que vous souhaitez détecter
min_size = 1000  # ajustez en fonction de vos besoins
max_size = 50000  # ajustez en fonction de vos besoins

min_size2 = 50  # ajustez en fonction de vos besoins
max_size2 = 800  # ajustez en fonction de vos besoins

# Filtrer les contours en fonction de la taille
filtered_contours = [cnt for cnt in contours if min_size < cv2.contourArea(cnt) < max_size]
filtered_contours2 = [cnt for cnt in contours if min_size2 < cv2.contourArea(cnt) < max_size2]
# Créer une image pour afficher les résultats
result_image = np.zeros_like(image)
result_image2 = np.zeros_like(image)

# Dessiner les contours filtrés sur l'image résultante
cv2.drawContours(result_image, filtered_contours, -1, (255), thickness=cv2.FILLED)
cv2.drawContours(result_image2, filtered_contours2, -1, (255), thickness=cv2.FILLED)

finalgros = cv2.bitwise_and(image, result_image)
finalpetit = cv2.bitwise_and(image, result_image2)

# Afficher les images
cv2.imshow("Image test", result_image)
cv2.imshow("Image originale", image)
cv2.imshow("Image seuillée", binary_image)
cv2.imshow("gros final", finalgros)
cv2.imshow("petit final", finalpetit)
##cv2.imshow("Résultat de la segmentation en niveau de gris", result_gray)

cv2.waitKey(0)
cv2.destroyAllWindows()

"""
import cv2
import numpy as np

# Charger l'image
image = cv2.imread("C:\\TraitementImages3eme\\Images\\balanes.png", cv2.IMREAD_GRAYSCALE)

_, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((11, 11), np.uint8)
image_eroded = cv2.erode(binary_image,kernel)


mask = cv2.bitwise_and(binary_image, image_eroded)

grayscale_image = cv2.cvtColor(image_eroded, cv2.COLOR_GRAY2BGR)

cv2.imshow("Image originale", image)
cv2.imshow("Image binaire", binary_image)
cv2.imshow("Image erodée", image_eroded)
cv2.imshow("Image erodée grise", grayscale_image)
cv2.imshow("image ?", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
import cv2
import numpy as np

# Charger l'image
image = cv2.imread("C:\\TraitementImages3eme\\Images\\balanes.png", cv2.IMREAD_GRAYSCALE)

# Appliquer l'érosion pour réduire les objets
kernel = np.ones((11, 11), np.uint8)
image_eroded = cv2.erode(image, kernel)

# Créer un masque en utilisant l'image érodée
mask = cv2.bitwise_not(image_eroded)  # Inverser le masque

# Restaurer la valeur d'origine des pixels en dehors des objets érodés
result_image = image.copy()
result_image[mask == 0] = 0  # Mettre à zéro les pixels en dehors du masque

# Afficher les images
cv2.imshow("Image originale", image)
cv2.imshow("Image érodée", image_eroded)
cv2.imshow("Masque", mask)
cv2.imshow("Résultat", result_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""
