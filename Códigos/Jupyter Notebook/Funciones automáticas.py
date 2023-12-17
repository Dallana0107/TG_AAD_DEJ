#!/usr/bin/env python
# coding: utf-8

# # Realce de características PILLOW

# In[ ]:


#REVISADO
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

# Abrir una imagen
imagen = Image.open('miche.jpg')

# Aplicar el filtro de realce de detalles
imagen_realzada = imagen.filter(ImageFilter.DETAIL)

# Crear una figura con dos subtramas para mostrar las imágenes
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Mostrar la imagen original en la primera subtrama
axs[0].imshow(imagen)
axs[0].set_title('Imagen Original')
axs[0].axis('off')

# Mostrar la imagen realzada en la segunda subtrama
axs[1].imshow(imagen_realzada)
axs[1].set_title('Imagen Realzada')
axs[1].axis('off')

# Ajustar el espacio entre subtramas
plt.tight_layout()

# Mostrar la figura con las subtramas
plt.show()


# In[ ]:


#REVISADO
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

# Abrir una imagen
imagen = Image.open('miche.jpg')

# Aplicar el filtro de realce de bordes
imagen_realzada = imagen.filter(ImageFilter.EDGE_ENHANCE)

# Crear una figura con dos subtramas para mostrar las imágenes
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Mostrar la imagen original en la primera subtrama
axs[0].imshow(imagen)
axs[0].set_title('Imagen Original')
axs[0].axis('off')

# Mostrar la imagen realzada en la segunda subtrama
axs[1].imshow(imagen_realzada)
axs[1].set_title('Imagen Realzada')
axs[1].axis('off')

# Ajustar el espacio entre subtramas
plt.tight_layout()

# Mostrar la figura con las subtramas
plt.show()


# # Mejora de contraste SKIMAGE

# In[ ]:


#REVISADO
from skimage import io, exposure
import matplotlib.pyplot as plt

# Cargar la imagen
imagen = io.imread('lun7.jpg')

# Mejorar el contraste utilizando ecualización del histograma
imagen_mejorada = exposure.equalize_hist(imagen)

# Crear una figura con dos subtramas
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Mostrar la imagen original en la primera subtrama
axes[0].imshow(imagen, cmap='gray')
axes[0].set_title('Imagen Original')

# Mostrar la imagen mejorada en la segunda subtrama
axes[1].imshow(imagen_mejorada, cmap='gray')
axes[1].set_title('Imagen Mejorada')

# Ajustar los ejes
for ax in axes:
    ax.axis('off')

# Mostrar la figura
plt.show()


# # Mejora de contraste OPENCV

# In[ ]:


#REVISADO
import cv2
import matplotlib.pyplot as plt

# Cargar la imagen en color
imagen = cv2.imread('lun1.jpg')

# Dividir la imagen en sus canales de color (BGR)
canal_b, canal_g, canal_r = cv2.split(imagen)

# Aplicar ecualización de histograma a cada canal por separado
canal_b_eq = cv2.equalizeHist(canal_b)
canal_g_eq = cv2.equalizeHist(canal_g)
canal_r_eq = cv2.equalizeHist(canal_r)

# Combinar los canales equilibrados para obtener la imagen en color equilibrada
imagen_eq = cv2.merge((canal_b_eq, canal_g_eq, canal_r_eq))

# Crear una figura de Matplotlib para mostrar las imágenes
plt.figure(figsize=(12, 6))

# Mostrar la imagen original con título
plt.subplot(131)
plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
plt.title('Imagen Original')

# Mostrar la imagen con contraste mejorado en escala de grises con título
plt.subplot(132)
plt.imshow(imagen_eq, cmap='gray')
plt.title('Imagen con Contraste Mejorado (Escala de Grises)')

# Mostrar la imagen con contraste mejorado en color con título
plt.subplot(133)
plt.imshow(cv2.cvtColor(imagen_eq, cv2.COLOR_BGR2RGB))
plt.title('Imagen con Contraste Mejorado (Color)')

# Ajustar el espacio entre las imágenes
plt.tight_layout()

# Mostrar el gráfico con todas las imágenes
plt.show()


# # Filtro de realce laplaciano OPENCV

# In[ ]:


#REVISADO
import cv2
import matplotlib.pyplot as plt

# Cargar la imagen de entrada
imagen = cv2.imread('miche.jpg')

# Aplicar un filtro de realce laplaciano
imagen_realzada = cv2.Laplacian(imagen, cv2.CV_64F)

# Convertir la imagen de salida a tipo uint8
imagen_realzada = cv2.convertScaleAbs(imagen_realzada)

# Crear una figura con dos subtramas para mostrar las imágenes
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Mostrar la imagen original en la primera subtrama
axs[0].imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
axs[0].set_title('Imagen Original')
axs[0].axis('off')

# Mostrar la imagen realzada de bordes en la segunda subtrama
axs[1].imshow(imagen_realzada, cmap='gray')
axs[1].set_title('Imagen Realzada de Bordes')
axs[1].axis('off')

# Ajustar el espacio entre subtramas
plt.tight_layout()

# Mostrar la figura con las subtramas
plt.show()


# In[ ]:


#REVISADO
import cv2
import matplotlib.pyplot as plt

# Cargar la imagen de entrada
imagen = cv2.imread('miche.jpg')

# Aplicar un filtro de realce mediante el gradiente Sobel
imagen_realzada = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)

# Convertir la imagen de salida a tipo uint8
imagen_realzada = cv2.convertScaleAbs(imagen_realzada)

# Crear una figura con dos subtramas para mostrar las imágenes
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Mostrar la imagen original en la primera subtrama
axs[0].imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
axs[0].set_title('Imagen Original')
axs[0].axis('off')

# Mostrar la imagen realzada de bordes en la segunda subtrama
axs[1].imshow(imagen_realzada, cmap='gray')
axs[1].set_title('Imagen Realzada de Bordes')
axs[1].axis('off')

# Ajustar el espacio entre subtramas
plt.tight_layout()

# Mostrar la figura con las subtramas
plt.show()


# In[ ]:


#REVISADO
#MEJORA DE CONTRASTE
import cv2
import matplotlib.pyplot as plt

imagen = cv2.imread('miche.jpg')
imagen_ecualizada = cv2.equalizeHist(cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY))

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
axs[0].set_title('Imagen Original')
axs[0].axis('off')

axs[1].imshow(imagen_ecualizada, cmap='gray')
axs[1].set_title('Ecualización de Histograma')
axs[1].axis('off')

plt.tight_layout()
plt.show()


# In[ ]:


#REVISADO
#BRILLO Y CONTRASTE
import cv2
import matplotlib.pyplot as plt

imagen = cv2.imread('miche.jpg', 0)  # Cargar como escala de grises
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
imagen_mejorada = clahe.apply(imagen)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(imagen, cmap='gray')
axs[0].set_title('Imagen Original')
axs[0].axis('off')

axs[1].imshow(imagen_mejorada, cmap='gray')
axs[1].set_title('CLAHE (Mejora de Contraste y Brillo)')
axs[1].axis('off')

plt.tight_layout()
plt.show()


# In[ ]:


#REVISADO
# CONTRASTE 
import cv2
import matplotlib.pyplot as plt

imagen = cv2.imread('miche.jpg')
imagen_ecualizada = cv2.equalizeHist(cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY))

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
axs[0].set_title('Imagen Original')
axs[0].axis('off')

axs[1].imshow(imagen_ecualizada, cmap='gray')
axs[1].set_title('Ecualización de Histograma')
axs[1].axis('off')

plt.tight_layout()
plt.show()


# In[ ]:


#REVISADO
# BRILLO Y CONTRASTE
import cv2
import matplotlib.pyplot as plt

imagen = cv2.imread('miche.jpg', 0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
imagen_mejorada = clahe.apply(imagen)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(imagen, cmap='gray')
axs[0].set_title('Imagen Original')
axs[0].axis('off')

axs[1].imshow(imagen_mejorada, cmap='gray')
axs[1].set_title('CLAHE (Mejora de Contraste y Brillo)')
axs[1].axis('off')

plt.tight_layout()
plt.show()


# In[ ]:


#REVISADO
# CONTRASTE 
import cv2
import matplotlib.pyplot as plt

imagen = cv2.imread('miche.jpg')
imagen_lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(imagen_lab)

# Aplicar ecualización de histograma al canal L (luminosidad)
l_ecualizado = cv2.equalizeHist(l)
imagen_ecualizada = cv2.merge((l_ecualizado, a, b))
imagen_ecualizada = cv2.cvtColor(imagen_ecualizada, cv2.COLOR_LAB2BGR)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
axs[0].set_title('Imagen Original')
axs[0].axis('off')

axs[1].imshow(cv2.cvtColor(imagen_ecualizada, cv2.COLOR_BGR2RGB))
axs[1].set_title('Ecualización de Histograma (Contraste Mejorado)')
axs[1].axis('off')

plt.tight_layout()
plt.show()


# In[ ]:


# NITIDEZ 
#REVISADO
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

imagen = Image.open('lun4.jpg')
imagen_realzada = imagen.filter(ImageFilter.SHARPEN)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(imagen)
axs[0].set_title('Imagen Original')
axs[0].axis('off')

axs[1].imshow(imagen_realzada)
axs[1].set_title('Realce de Nitidez')
axs[1].axis('off')

plt.tight_layout()
plt.show()


# In[ ]:




