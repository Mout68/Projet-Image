#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import cmath

def etirement_histog(image):
    if image.ndim == 3:
        image = convertir_en_niveaux_de_gris(image)
    rows, cols = image.shape
    # Dimensions de l'image
    rows, cols = image.shape


    # Calcul le min et max de l'
    
    Min = 255 ; Max = 0
    
    for u in range(rows):
        for v in range(cols):
            if image[u, v] < Min :
                Min = image[u, v]
            if image[u, v] > Max :
                Max = image[u, v]
    
    print("Min Image = ", Min, "Max Image = ", Max)
    
    # Création de l'image étirée
    et_image = np.zeros_like(image, dtype=np.uint8)
    
    for u in range(rows):
        for v in range(cols):
            et_image[u, v] = int(255*(image[u,v]-Min)/(Max-Min))
    
    return et_image

def egalisation_histogramme(image):

    # Calculer l'histogramme (256 niveaux de gris)
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # Calculer la fonction de distribution cumulative (CDF)
    cdf = hist.cumsum()
    cdf_normalisee = cdf * 255 / cdf[-1]  # Normalisation entre 0 et 255

    # Appliquer la transformation aux pixels de l'image
    image_egalisee = np.interp(image.flatten(), bins[:-1], cdf_normalisee).reshape(image.shape).astype(np.uint8)

    return image_egalisee

def specification_histogramme(source, reference):

    # Calculer les histogrammes et les CDF des deux images
    hist_src, bins_src = np.histogram(source.flatten(), 256, [0, 256])
    hist_ref, bins_ref = np.histogram(reference.flatten(), 256, [0, 256])

    cdf_src = hist_src.cumsum()  # CDF de l'image source
    cdf_ref = hist_ref.cumsum()  # CDF de l'image de référence

    cdf_src_normalized = cdf_src * 255 / cdf_src[-1]  # Normalisation entre 0 et 255
    cdf_ref_normalized = cdf_ref * 255 / cdf_ref[-1]  # Normalisation entre 0 et 255

    # Création de la correspondance des niveaux de gris
    mapping = np.zeros(256, dtype=np.uint8)
    
    for i in range(256):
        diff = np.abs(cdf_ref_normalized - cdf_src_normalized[i])
        mapping[i] = np.argmin(diff)  # Trouver le niveau de gris correspondant

    # Appliquer la transformation aux pixels de l'image source
    image_specifiee = mapping[source]

    return image_specifiee

def filtre_median(image,kernel_size=3):
    # Dimensions de l'image
    hauteur, largeur = image.shape
    offset = kernel_size // 2
    image_filtre = np.zeros((hauteur, largeur), dtype=np.uint8)
    for i in range(offset, hauteur - offset):
        for j in range(offset, largeur - offset):
            voisinage = image[i-offset:i+offset+1, j-offset:j+offset+1]
            image_filtre[i, j] = np.median(voisinage)
                
    return image_filtre

def filtre_nagao(image):
    
    # Dimensions de l'image
    hauteur = image.shape[0]  # ✅ hauteur
    largeur = image.shape[1]  # ✅ largeur
    
    # Définition des régions de Nagao (indices pour une fenêtre 5x5)
    regions = [
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],  # Haut-gauche
        [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4)],  # Haut-centre
        [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2)],  # Milieu-gauche
        [(2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4)],  # Milieu-droite
        [(3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2)],  # Bas-gauche
        [(3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4)],  # Bas-centre
        [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (2, 3), (3, 2), (3, 3)],  # Diag-gauche
        [(0, 3), (0, 4), (1, 3), (1, 4), (2, 1), (2, 2), (3, 1), (3, 2)],   # Diag-droite
        [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)] #Fenêtre centrale 3x3
    ]
    
    # Création d'une image pour stocker le résultat
    image_filtre = np.zeros((hauteur, largeur), dtype=np.uint8)

    # Appliquer le filtre de Nagao
    for i in range(2, hauteur - 2):
        for j in range(2, largeur - 2):
            # Extraire la fenêtre 5x5 autour du pixel (i, j)
            patch = image[i - 2:i + 3, j - 2:j + 3]
            
            # Liste pour stocker les moyennes et variances des régions
            variances = []
            moyennes = []

            for region in regions:
                valeurs = [patch[x, y] for x, y in region]  # Extraire les pixels de la région
                moyenne = np.mean(valeurs)
                variance = np.var(valeurs)

                moyennes.append(moyenne)
                variances.append(variance)

            # Sélectionner la région avec la plus petite variance
            meilleure_region = np.argmin(variances)
            
            # Remplacer le pixel central par la moyenne de cette région
            image_filtre[i, j] = moyennes[meilleure_region]

    return image_filtre

def addition2images(image1, image2):
    image1=cv.resize(image1, (image2.shape[1], image2.shape[0]), interpolation= cv.INTER_LINEAR)
    return np.minimum(image1.astype(np.int16) + image2.astype(np.int16), 255).astype(np.uint8)

def soustraction2images(image1, image2):
    image1=cv.resize(image1, (image2.shape[1], image2.shape[0]), interpolation= cv.INTER_LINEAR)
    return np.maximum(image1.astype(np.int16) - image2.astype(np.int16), 0).astype(np.uint8)

def detect_lines_avec_Hough(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    return image

def convolution(image, filtre):
    # Dimensions de l'image et du filtre
    if len(image.shape) == 3:
        image = convertir_en_niveaux_de_gris(image)
    haut, larg = len(image), len(image[0])
    fhaut, flarg = len(filtre), len(filtre[0])

    # Paddings pour l'image afin d'appliquer le filtre sur les bords
    pad_haut = fhaut // 2
    pad_larg = flarg // 2

    # Création de l'image avec padding
    pad_image = np.pad(image, ((pad_haut, pad_haut), (pad_larg, pad_larg)), mode='constant', constant_values=0)

    # Image de sortie après convolution
    output_image = np.zeros((haut, larg))

    # Application du filtre (convolution)
    for i in range(haut):
        for j in range(larg):
            region = pad_image[i:i + fhaut, j:j + flarg]  # Récupère la région de l'image sur laquelle le filtre est appliqué
            pixel_value = np.sum(region * filtre)  # Somme des produits de la région et du filtre
            output_image[i, j] = pixel_value

    # Normalisation si nécessaire (on limite les valeurs entre 0 et 255)
    output_image = np.clip(output_image, 0, 255)

    return output_image


def appliquer_filtre_sobel(image):

    sobel_x = [[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]]

    sobel_y = [[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]]


    gradient_x = convolution(image, sobel_x)
    gradient_y = convolution(image, sobel_y)
    sobel_magnitude = []

    for i in range(len(gradient_x)):
        row = []
        for j in range(len(gradient_x[0])):
            magnitude = sqrt(gradient_x[i][j] ** 2 + gradient_y[i][j] ** 2)
            row.append(magnitude)
        sobel_magnitude.append(row)

    max_value = max(max(row) for row in sobel_magnitude)
    for i in range(len(sobel_magnitude)):
        for j in range(len(sobel_magnitude[0])):
            sobel_magnitude[i][j] = (sobel_magnitude[i][j] / max_value) * 255

    sobel_magnitude = [[int(pixel) for pixel in row] for row in sobel_magnitude]

    return sobel_magnitude

def dft(x): #Discrete Fourier Transform
    n = len(x)
    x_transform = [0] * n
    for k in range(n):
        for m in range(n):

            angle = -2j * cmath.pi * k * m / n
            x_transform[k] += x[m] * cmath.exp(angle)
    return x_transform

def idft(x_transform): #Inverse Discrete Fourier Transform
    n = len(x_transform)
    x = [0] * n
    for m in range(n):
        for k in range(n):
            angle = 2j * cmath.pi * k * m / n
            x[m] += x_transform[k] * cmath.exp(angle)
    x = [val / n for val in x]
    return x

def fft(x): #Fast Fourier Transform
    n = len(x)
    if n <= 1:
        return x
    even = fft(x[0::2])
    odd = fft(x[1::2])
    t = [cmath.exp(2j * cmath.pi * k / n) * odd[k] for k in range(n // 2)]
    return [even[k] + t[k] for k in range(n // 2)] + [even[k] - t[k] for k in range(n // 2)]

def ifft(x_transform): #Inverse Fast Fourier Transform
    n = len(x_transform)
    x_conj = [x.conjugate() for x in x_transform]
    x = fft(x_conj)
    return [val.conjugate() / n for val in x]

def transformation_lut(image):
    # Créer une LUT simple (0-255)
    lut = np.arange(256, dtype=np.uint8)
    lut = np.clip(lut + 50, 0, 255)  # Ajouter 50 à chaque pixel, puis on le limite entre 0 et 255

    transformed_image = np.zeros_like(image, dtype=np.uint8)
    
    # Parcourir tous les pixels de l'image
    for i in range(image.shape[0]):  # Parcours des lignes
        for j in range(image.shape[1]):  # Parcours des colonnes
            pixel_value = image[i, j]  # Valeur du pixel
            transformed_image[i, j] = lut[pixel_value]  # Appliquer la LUT

    return transformed_image

def appliquerOperatorET(Image1, Image2):
    # Redimensionner les deux images à la même taille
    # On prend la taille de la première image, et on redimensionne la deuxième image pour qu'elle ait la même taille
    Image1=cv.resize(Image1, (Image2.shape[1], Image2.shape[0]), interpolation= cv.INTER_LINEAR)
    
    # Initialiser une image vide pour stocker le résultat
    result = np.zeros_like(Image1, dtype=np.uint8)
    
    # Appliquer l'opération AND sur chaque pixel
    for i in range(Image1.shape[0]):
        for j in range(Image1.shape[1]):
            pixel1 = Image1[i, j]  # Valeur du pixel dans la première image
            pixel2 = Image2[i, j]  # Valeur du pixel dans la deuxième image
            
            # Appliquer l'opérateur AND entre les deux pixels
            result[i, j] = pixel1 & pixel2  # Le résultat est également un nombre entre 0 et 255
    
    return result

def appliquerOperatorOU(Image1, Image2):
    # Redimensionner les deux images à la même taille
    # On prend la taille de la première image, et on redimensionne la deuxième image pour qu'elle ait la même taille
    Image1=cv.resize(Image1, (Image2.shape[1], Image2.shape[0]), interpolation= cv.INTER_LINEAR)
    
    # Initialiser une image vide pour stocker le résultat
    result = np.zeros_like(Image1, dtype=np.uint8)
    
    # Appliquer l'opération AND sur chaque pixel
    for i in range(Image1.shape[0]):
        for j in range(Image1.shape[1]):
            pixel1 = Image1[i, j]  # Valeur du pixel dans la première image
            pixel2 = Image2[i, j]  # Valeur du pixel dans la deuxième image
            
            # Appliquer l'opérateur AND entre les deux pixels
            result[i, j] = pixel1 | pixel2  # Le résultat est également un nombre entre 0 et 255
    
    return result

def methode_otsu(image):
    # Calculer l'histogramme (256 niveaux de gris)
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    total_pixels = image.size

    # Probabilités de chaque niveau de gris
    probas = hist / total_pixels

    # Variables pour le meilleur seuil
    meilleur_seuil = 0
    variance_max = 0

    # Moyenne globale (pondérée)
    moyenne_totale = np.dot(np.arange(256), probas)

    # Variables pour les calculs dans la boucle
    poids_0 = 0
    moyenne_0 = 0

    for seuil in range(256):
        poids_0 += probas[seuil]
        if poids_0 == 0:
            continue

        poids_1 = 1 - poids_0
        if poids_1 == 0:
            break

        moyenne_0 += seuil * probas[seuil]
        moyenne_1 = (moyenne_totale - moyenne_0) / poids_1

        moyenne_classe_0 = moyenne_0 / poids_0

        # Variance inter-classe
        variance = poids_0 * poids_1 * (moyenne_classe_0 - moyenne_1)**2

        if variance > variance_max:
            variance_max = variance
            meilleur_seuil = seuil

    # Appliquer le seuillage avec le seuil optimal
    image_seuillee = (image >= meilleur_seuil).astype(np.uint8) * 255

    return image_seuillee, meilleur_seuil


def convertir_en_niveaux_de_gris(image):
    # Suppose que l'image est en format BGR (comme avec OpenCV)
    if len(image.shape) == 3 and image.shape[2] == 3:
        B, G, R = image[:,:,0], image[:,:,1], image[:,:,2]
        gris = 0.114 * B + 0.587 * G + 0.299 * R
        return gris.astype(np.uint8)
    else:
        # L'image est déjà en niveaux de gris
        return image

def rapport_inverse_contraste(image):
    """
    Applique le rapport inverse de contraste (inversion de l'intensité)
    pour mettre en valeur les faibles contrastes.
    
    Paramètre :
    - image : image en niveaux de gris (uint8)

    Retour :
    - image transformée
    """
    gray=convertir_en_niveaux_de_gris(image)
    return 255 - gray

def compression_image_bilineaire(image, facteur_echelle=0.5):
    """
    Compresse une image en appliquant une interpolation bilinéaire manuelle.

    Paramètres :
    - image : image d'entrée (grayscale)
    - facteur_echelle : entre 0 et 1 pour compresser (ex: 0.5 pour 50%)

    Retour :
    - image redimensionnée
    """

    image=convertir_en_niveaux_de_gris(image)
    
    hauteur = image.shape[0]  # ✅ hauteur
    largeur = image.shape[1]  # ✅ largeur
    
    new_hauteur = int(hauteur * facteur_echelle)
    new_largeur = int(largeur * facteur_echelle)

    image_reduite = np.zeros((new_hauteur, new_largeur), dtype=np.uint8)

    for i in range(new_hauteur):
        for j in range(new_largeur):
            # Coordonnées dans l'image d'origine
            #x = i / facteur_echelle
            #y = j / facteur_echelle
            x = i * (hauteur / new_hauteur)
            y = j * (largeur / new_largeur)
            
            x0 = int(np.floor(x))
            x1 = min(x0 + 1, hauteur - 1)
            y0 = int(np.floor(y))
            y1 = min(y0 + 1, largeur - 1)

            # Poids
            a = x - x0
            b = y - y0

            # Interpolation bilinéaire
            val = (1 - a) * (1 - b) * image[x0, y0] + \
                  a * (1 - b) * image[x1, y0] + \
                  (1 - a) * b * image[x0, y1] + \
                  a * b * image[x1, y1]

            image_reduite[i, j] = int(val)

    return image_reduite

def multiplier_image_avec_ratio(image, ratio):
    """
    Multiplie les valeurs des pixels d'une image par un ratio donné.

    Paramètres :
    - image : image d'entrée (niveaux de gris ou couleur)
    - ratio : facteur multiplicatif (ex : 0.5 pour assombrir, 2.0 pour éclaircir)

    Retour :
    - image modifiée avec les valeurs limitées entre 0 et 255
    """
    # Convertir en float pour éviter les débordements
    image_float = image.astype(np.float32)
    
    # Appliquer le ratio
    image_modifiee = image_float * ratio

    # Clipper les valeurs pour rester entre 0 et 255
    image_modifiee = np.clip(image_modifiee, 0, 255)

    # Reconvertir en uint8
    return image_modifiee.astype(np.uint8)

def lireImage(pathImage):
    image=cv.imread(pathImage)
    return image

def afficherDeuxImages(image1, image2,  titre1='Image 1', titre2='Image 2'):
     
    f, ax = plt.subplot_mosaic([['origi', 'result']], figsize=(7, 3.5)) 

    ax['origi'].imshow(image1)
    ax['origi'].set_title(titre1)
    ax['origi'].axis('off')

    ax['result'].imshow(image2)
    ax['result'].set_title(titre2)
    ax['result'].axis('off')
    
def afficherTroisImages(image1, image2, image3, titre1='Image 1', titre2='Image 2', titre3='Image 3'):
    f, ax = plt.subplot_mosaic([['origi', 'result', 'troisieme']], figsize=(10, 3.5)) 

    ax['origi'].imshow(image1)
    ax['origi'].set_title(titre1)
    ax['origi'].axis('off')

    ax['result'].imshow(image2)
    ax['result'].set_title(titre2)
    ax['result'].axis('off')

    ax['troisieme'].imshow(image3)
    ax['troisieme'].set_title(titre3)
    ax['troisieme'].axis('off')