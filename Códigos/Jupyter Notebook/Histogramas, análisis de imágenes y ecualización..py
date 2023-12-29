#!/usr/bin/env python
# coding: utf-8

# # Análisis de 8 bits por canal

# In[ ]:


#ANALISIS DE 8 BITS POR CANAL
import cv2

# Cargar la imagen
imagen = cv2.imread('imagen.jpg')

# Obtener la profundidad de bits de la imagen
profundidad_bits = imagen.dtype

# Comprobar la profundidad de bits
if profundidad_bits == 'uint8':
    print('La imagen tiene 8 bits por canal.')
else:
    print('La imagen no tiene 8 bits por canal.')


# # Imágenes e histogramas RGB separados

# In[ ]:


#Código para 1 imagen
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen en formato BGR (OpenCV carga las imágenes en formato BGR por defecto)
imagen = cv2.imread('imagen.jpg')

# Separa los canales de color (R, G, B) de la imagen
canal_b, canal_g, canal_r = cv2.split(imagen)

# Calcula los histogramas de cada canal
hist_b = cv2.calcHist([canal_b], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([canal_g], [0], None, [256], [0, 256])
hist_r = cv2.calcHist([canal_r], [0], None, [256], [0, 256])

# Crear subtramas para mostrar la imagen y los histogramas
fig, axs = plt.subplots(1, 4, figsize=(16, 4))

# Mostrar la imagen original en la primera columna
axs[0].imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
axs[0].axis('off')  # Ocultar ejes
axs[0].set_title('Imagen Original')

# Mostrar el histograma del canal Rojo (R) en la segunda columna
axs[1].plot(hist_r, color='red')
axs[1].set_title('Histograma Rojo (R)')
axs[1].set_xlabel('Valor de Píxel')
axs[1].set_ylabel('Frecuencia')

# Mostrar el histograma del canal Verde (G) en la tercera columna
axs[2].plot(hist_g, color='green')
axs[2].set_title('Histograma Verde (G)')
axs[2].set_xlabel('Valor de Píxel')
axs[2].set_ylabel('Frecuencia')

# Mostrar el histograma del canal Azul (B) en la cuarta columna
axs[3].plot(hist_b, color='blue')
axs[3].set_title('Histograma Azul (B)')
axs[3].set_xlabel('Valor de Píxel')
axs[3].set_ylabel('Frecuencia')

# Ajustar el espacio entre subtramas
plt.tight_layout()

# Mostrar las subtramas
plt.show()


# In[ ]:


#Código para 2 imagenes
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la primera imagen en formato BGR
imagen1 = cv2.imread('im5.jpg')

# Cargar la segunda imagen en formato BGR
imagen2 = cv2.imread('imagen_realzada.jpg')

# Lista de imágenes
imagenes = [imagen1, imagen2]

# Crear subtramas para mostrar las imágenes y los histogramas
fig, axs = plt.subplots(len(imagenes), 4, figsize=(16, 4 * len(imagenes)))

for i, imagen in enumerate(imagenes):
    # Separa los canales de color (R, G, B) de la imagen
    canal_b, canal_g, canal_r = cv2.split(imagen)

    # Calcula los histogramas de cada canal
    hist_b = cv2.calcHist([canal_b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([canal_g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([canal_r], [0], None, [256], [0, 256])

    # Mostrar la imagen en la primera columna
    axs[i, 0].imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    axs[i, 0].axis('off')  # Ocultar ejes
    axs[i, 0].set_title(f'Imagen {i + 1}')

    # Mostrar el histograma del canal Rojo (R) en la segunda columna
    axs[i, 1].plot(hist_r, color='red')
    axs[i, 1].set_title('Histograma Rojo (R)')
    axs[i, 1].set_xlabel('Valor de Píxel')
    axs[i, 1].set_ylabel('Frecuencia')

    # Mostrar el histograma del canal Verde (G) en la tercera columna
    axs[i, 2].plot(hist_g, color='green')
    axs[i, 2].set_title('Histograma Verde (G)')
    axs[i, 2].set_xlabel('Valor de Píxel')
    axs[i, 2].set_ylabel('Frecuencia')

    # Mostrar el histograma del canal Azul (B) en la cuarta columna
    axs[i, 3].plot(hist_b, color='blue')
    axs[i, 3].set_title('Histograma Azul (B)')
    axs[i, 3].set_xlabel('Valor de Píxel')
    axs[i, 3].set_ylabel('Frecuencia')

# Ajustar el espacio entre subtramas
plt.tight_layout()

# Mostrar las subtramas
plt.show()


# In[ ]:


#Código para 3 imagenes
import cv2
import matplotlib.pyplot as plt

def mostrar_histogramas(imagen, titulo):
    # Convertir de BGR a RGB para mostrar con Matplotlib
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    # Crear la figura
    plt.figure(figsize=(12, 4))

    # Mostrar la imagen
    plt.subplot(1, 4, 1)
    plt.imshow(imagen_rgb)
    plt.title(titulo)
    plt.axis('off')

    # Mostrar histograma R
    plt.subplot(1, 4, 2)
    histograma_r = cv2.calcHist([imagen], [0], None, [256], [0, 256])
    plt.plot(histograma_r, color='r')
    plt.xlim([0, 256])
    plt.title('Histograma R')

    # Mostrar histograma G
    plt.subplot(1, 4, 3)
    histograma_g = cv2.calcHist([imagen], [1], None, [256], [0, 256])
    plt.plot(histograma_g, color='g')
    plt.xlim([0, 256])
    plt.title('Histograma G')

    # Mostrar histograma B
    plt.subplot(1, 4, 4)
    histograma_b = cv2.calcHist([imagen], [2], None, [256], [0, 256])
    plt.plot(histograma_b, color='b')
    plt.xlim([0, 256])
    plt.title('Histograma B')

    # Ajustar diseño
    plt.tight_layout()

    # Mostrar la figura
    plt.show()

# Leer las imágenes
imagen_lun1 = cv2.imread('imagen6.jpg')
imagen_lun2 = cv2.imread('imagen_agudizada.jpg')
imagen_realzada = cv2.imread('imagen_realzada.jpg')

# Mostrar la primera imagen con sus histogramas
mostrar_histogramas(imagen_lun1, 'Imagen modulo')

# Mostrar la segunda imagen con sus histogramas
mostrar_histogramas(imagen_lun2, 'Imagen para maix')

# Mostrar la tercera imagen con sus histogramas
mostrar_histogramas(imagen_realzada, 'Imagen jupyter')




# # Ecualización con un parámetro gamma para toda la imagen

# In[ ]:


# Ecualización con un parámetro gamma para toda la imagen
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Cargar la imagen RGB
imagen_rgb = cv2.imread('imagen.jpg')

# Definir el parámetro gamma (ajusta este valor según tus necesidades)
gamma = 1.5

# Aplicar la corrección gamma a cada canal de color
canal_b_corregido = np.power(imagen_rgb[:, :, 0] / 255.0, gamma) * 255.0
canal_g_corregido = np.power(imagen_rgb[:, :, 1] / 255.0, gamma) * 255.0
canal_r_corregido = np.power(imagen_rgb[:, :, 2] / 255.0, gamma) * 255.0

# Combinar los canales corregidos en una imagen RGB corregida
imagen_corregida_rgb = cv2.merge((canal_b_corregido, canal_g_corregido, canal_r_corregido)).astype(np.uint8)

# Calcular los histogramas de la imagen original y la imagen corregida
hist_b_original = cv2.calcHist([imagen_rgb[:, :, 0]], [0], None, [256], [0, 256])
hist_g_original = cv2.calcHist([imagen_rgb[:, :, 1]], [0], None, [256], [0, 256])
hist_r_original = cv2.calcHist([imagen_rgb[:, :, 2]], [0], None, [256], [0, 256])

hist_b_corregido = cv2.calcHist([canal_b_corregido.astype(np.uint8)], [0], None, [256], [0, 256])
hist_g_corregido = cv2.calcHist([canal_g_corregido.astype(np.uint8)], [0], None, [256], [0, 256])
hist_r_corregido = cv2.calcHist([canal_r_corregido.astype(np.uint8)], [0], None, [256], [0, 256])

# Mostrar las imágenes y los histogramas RGB en dos filas
plt.figure(figsize=(12, 8))

# Primera fila: Imagen original y histogramas RGB
plt.subplot(2, 4, 1), plt.imshow(cv2.cvtColor(imagen_rgb, cv2.COLOR_BGR2RGB)), plt.title('Imagen Original')
plt.subplot(2, 4, 2), plt.plot(hist_b_original, color='blue'), plt.title('Histograma Canal B')
plt.subplot(2, 4, 3), plt.plot(hist_g_original, color='green'), plt.title('Histograma Canal G')
plt.subplot(2, 4, 4), plt.plot(hist_r_original, color='red'), plt.title('Histograma Canal R')

# Segunda fila: Imagen corregida y histogramas RGB corregidos
plt.subplot(2, 4, 5), plt.imshow(cv2.cvtColor(imagen_corregida_rgb, cv2.COLOR_BGR2RGB)), plt.title('Imagen Corregida')
plt.subplot(2, 4, 6), plt.plot(hist_b_corregido, color='blue'), plt.title('Histograma Canal B Corregido')
plt.subplot(2, 4, 7), plt.plot(hist_g_corregido, color='green'), plt.title('Histograma Canal G Corregido')
plt.subplot(2, 4, 8), plt.plot(hist_r_corregido, color='red'), plt.title('Histograma Canal R Corregido')

plt.tight_layout()
plt.show()



# # Prueba de ecualización con variación del parámetro gamma en un rango [0.8 - 1.5] con un espacio de 0.1

# In[ ]:


# Prueba de ecualización con variación del parámetro gamma en un rango [0.8 - 1.5] con un espacio de 0.1
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen en formato BGR (OpenCV carga las imágenes en formato BGR por defecto)
imagen = cv2.imread('imagen.jpg')

# Separa los canales de color (R, G, B) de la imagen
canal_b, canal_g, canal_r = cv2.split(imagen)

# Definir los valores iniciales y finales de gamma
inicio_gamma = 0.8
fin_gamma = 1.5
paso_gamma = 0.1

# Crear una lista de valores de gamma en el rango especificado
valores_gamma = np.arange(inicio_gamma, fin_gamma + paso_gamma, paso_gamma)

# Crear subtramas para mostrar las imágenes y los histogramas
filas = len(valores_gamma)
columnas = 4  # 1 para la imagen original y 3 para los histogramas

fig, axs = plt.subplots(filas, columnas, figsize=(12, 3 * filas))

for i, gamma in enumerate(valores_gamma):
    # Aplicar la corrección gamma a cada canal con los valores actuales de gamma
    canal_b_corregido = np.power(canal_b / 255.0, gamma) * 255.0
    canal_g_corregido = np.power(canal_g / 255.0, gamma) * 255.0
    canal_r_corregido = np.power(canal_r / 255.0, gamma) * 255.0

    # Combinar los canales corregidos en una imagen RGB corregida
    imagen_corregida_rgb = cv2.merge((canal_b_corregido, canal_g_corregido, canal_r_corregido)).astype(np.uint8)

    # Calcular los histogramas de cada canal para la imagen corregida
    hist_b_corregido = cv2.calcHist([canal_b_corregido.astype(np.uint8)], [0], None, [256], [0, 256])
    hist_g_corregido = cv2.calcHist([canal_g_corregido.astype(np.uint8)], [0], None, [256], [0, 256])
    hist_r_corregido = cv2.calcHist([canal_r_corregido.astype(np.uint8)], [0], None, [256], [0, 256])

    # Mostrar la imagen corregida en la primera columna
    axs[i, 0].imshow(cv2.cvtColor(imagen_corregida_rgb, cv2.COLOR_BGR2RGB))
    axs[i, 0].axis('off')  # Ocultar ejes
    axs[i, 0].set_title(f'Gamma = {gamma:.1f}')

    # Mostrar el histograma del canal Rojo (R) para la imagen corregida en la segunda columna
    axs[i, 1].plot(hist_r_corregido, color='red')
    axs[i, 1].set_title('Histograma Rojo (R) Corregido')
    axs[i, 1].set_xlabel('Valor de Píxel')
    axs[i, 1].set_ylabel('Frecuencia')

    # Mostrar el histograma del canal Verde (G) para la imagen corregida en la tercera columna
    axs[i, 2].plot(hist_g_corregido, color='green')
    axs[i, 2].set_title('Histograma Verde (G) Corregido')
    axs[i, 2].set_xlabel('Valor de Píxel')
    axs[i, 2].set_ylabel('Frecuencia')

    # Mostrar el histograma del canal Azul (B) para la imagen corregida en la cuarta columna
    axs[i, 3].plot(hist_b_corregido, color='blue')
    axs[i, 3].set_title('Histograma Azul (B) Corregido')
    axs[i, 3].set_xlabel('Valor de Píxel')
    axs[i, 3].set_ylabel('Frecuencia')

# Ajustar el espacio entre subtramas
plt.tight_layout()

# Mostrar las subtramas
plt.show()


# # Ecualización estándar - Parámetros de gamma individuales (R-G-B)

# In[ ]:


# Ecualización estándar - Parámetros de gamma individuales (R-G-B)
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen en formato BGR (OpenCV carga las imágenes en formato BGR por defecto)
imagen = cv2.imread('miche.jpg')

# Separa los canales de color (R, G, B) de la imagen
canal_b, canal_g, canal_r = cv2.split(imagen)

# Calcula los histogramas de cada canal para la imagen original
hist_b = cv2.calcHist([canal_b], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([canal_g], [0], None, [256], [0, 256])
hist_r = cv2.calcHist([canal_r], [0], None, [256], [0, 256])

# Definir los parámetros gamma para cada canal (ajusta estos valores según tus necesidades)
gamma_r = 1.12  # Parámetro gamma para el canal Rojo (R)
gamma_g = 1.22  # Parámetro gamma para el canal Verde (G)
gamma_b = 1.32  # Parámetro gamma para el canal Azul (B)

# Aplicar la corrección gamma a cada canal
canal_b_corregido = np.power(canal_b / 255.0, gamma_b) * 255.0
canal_g_corregido = np.power(canal_g / 255.0, gamma_g) * 255.0
canal_r_corregido = np.power(canal_r / 255.0, gamma_r) * 255.0

# Combinar los canales corregidos en una imagen RGB corregida
imagen_corregida_rgb = cv2.merge((canal_b_corregido, canal_g_corregido, canal_r_corregido)).astype(np.uint8)

# Calcula los histogramas de cada canal para la imagen corregida
hist_b_corregido = cv2.calcHist([canal_b_corregido.astype(np.uint8)], [0], None, [256], [0, 256])
hist_g_corregido = cv2.calcHist([canal_g_corregido.astype(np.uint8)], [0], None, [256], [0, 256])
hist_r_corregido = cv2.calcHist([canal_r_corregido.astype(np.uint8)], [0], None, [256], [0, 256])

# Crear subtramas para mostrar la imagen original, la imagen corregida y los histogramas
fig, axs = plt.subplots(2, 4, figsize=(16, 8))

# Mostrar la imagen original en la primera columna
axs[0, 0].imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
axs[0, 0].axis('off')  # Ocultar ejes
axs[0, 0].set_title('Imagen Original')

# Mostrar el histograma del canal Rojo (R) para la imagen original en la segunda columna
axs[0, 1].plot(hist_r, color='red')
axs[0, 1].set_title('Histograma Rojo (R) Original')
axs[0, 1].set_xlabel('Valor de Píxel')
axs[0, 1].set_ylabel('Frecuencia')

# Mostrar el histograma del canal Verde (G) para la imagen original en la tercera columna
axs[0, 2].plot(hist_g, color='green')
axs[0, 2].set_title('Histograma Verde (G) Original')
axs[0, 2].set_xlabel('Valor de Píxel')
axs[0, 2].set_ylabel('Frecuencia')

# Mostrar el histograma del canal Azul (B) para la imagen original en la cuarta columna
axs[0, 3].plot(hist_b, color='blue')
axs[0, 3].set_title('Histograma Azul (B) Original')
axs[0, 3].set_xlabel('Valor de Píxel')
axs[0, 3].set_ylabel('Frecuencia')

# Mostrar la imagen corregida en la quinta columna
axs[1, 0].imshow(cv2.cvtColor(imagen_corregida_rgb, cv2.COLOR_BGR2RGB))
axs[1, 0].axis('off')  # Ocultar ejes
axs[1, 0].set_title('Imagen Corregida')

# Mostrar el histograma del canal Rojo (R) para la imagen corregida en la sexta columna
axs[1, 1].plot(hist_r_corregido, color='red')
axs[1, 1].set_title('Histograma Rojo (R) Corregido')
axs[1, 1].set_xlabel('Valor de Píxel')
axs[1, 1].set_ylabel('Frecuencia')

# Mostrar el histograma del canal Verde (G) para la imagen corregida en la séptima columna
axs[1, 2].plot(hist_g_corregido, color='green')
axs[1, 2].set_title('Histograma Verde (G) Corregido')
axs[1, 2].set_xlabel('Valor de Píxel')
axs[1, 2].set_ylabel('Frecuencia')

# Mostrar el histograma del canal Azul (B) para la imagen corregida en la octava columna
axs[1, 3].plot(hist_b_corregido, color='blue')
axs[1, 3].set_title('Histograma Azul (B) Corregido')
axs[1, 3].set_xlabel('Valor de Píxel')
axs[1, 3].set_ylabel('Frecuencia')

# Ruta y nombre de archivo para guardar la imagen ecualizada
nombre_archivo_ecualizado = 'imagen_ecualizada.jpg'

# Guardar la imagen ecualizada en la misma carpeta que tu cuaderno de Jupyter
cv2.imwrite(nombre_archivo_ecualizado, imagen_corregida_rgb)

# Imprimir un mensaje de confirmación
print(f'Imagen ecualizada guardada como "{nombre_archivo_ecualizado}" en la carpeta actual.')

# Ajustar el espacio entre subtramas
plt.tight_layout()

# Mostrar las subtramas
plt.show()



# # Comparación de los histogramas de 3 imágenes

# In[ ]:


#COMPARAR LOS 3 HISTOGRAMAS EN UNA SOLA GRAFICA
import cv2
import matplotlib.pyplot as plt

def mostrar_histogramas(imagen1, imagen2, imagen3):
    # Convertir de BGR a RGB para mostrar con Matplotlib
    imagen1_rgb = cv2.cvtColor(imagen1, cv2.COLOR_BGR2RGB)
    imagen2_rgb = cv2.cvtColor(imagen2, cv2.COLOR_BGR2RGB)
    imagen3_rgb = cv2.cvtColor(imagen3, cv2.COLOR_BGR2RGB)

    # Crear la figura
    plt.figure(figsize=(12, 4))

    # Mostrar histograma R
    plt.subplot(1, 3, 1)
    histograma_r_1 = cv2.calcHist([imagen1], [0], None, [256], [0, 256])
    histograma_r_2 = cv2.calcHist([imagen2], [0], None, [256], [0, 256])
    histograma_r_3 = cv2.calcHist([imagen3], [0], None, [256], [0, 256])
    plt.plot(histograma_r_1, color='r', label='Imagen Original')
    plt.plot(histograma_r_2, color='g', label='Imagen Maix')
    plt.plot(histograma_r_3, color='b', label='Imagen Procesada')
    plt.xlim([0, 256])
    plt.title('Histograma R')
    plt.legend()

    # Mostrar histograma G
    plt.subplot(1, 3, 2)
    histograma_g_1 = cv2.calcHist([imagen1], [1], None, [256], [0, 256])
    histograma_g_2 = cv2.calcHist([imagen2], [1], None, [256], [0, 256])
    histograma_g_3 = cv2.calcHist([imagen3], [1], None, [256], [0, 256])
    plt.plot(histograma_g_1, color='r', label='Imagen Original')
    plt.plot(histograma_g_2, color='g', label='Imagen Maix')
    plt.plot(histograma_g_3, color='b', label='Imagen Procesada')
    plt.xlim([0, 256])
    plt.title('Histograma G')
    plt.legend()

    # Mostrar histograma B
    plt.subplot(1, 3, 3)
    histograma_b_1 = cv2.calcHist([imagen1], [2], None, [256], [0, 256])
    histograma_b_2 = cv2.calcHist([imagen2], [2], None, [256], [0, 256])
    histograma_b_3 = cv2.calcHist([imagen3], [2], None, [256], [0, 256])
    plt.plot(histograma_b_1, color='r', label='Imagen Original')
    plt.plot(histograma_b_2, color='g', label='Imagen maix')
    plt.plot(histograma_b_3, color='b', label='Imagen procesada')
    plt.xlim([0, 256])
    plt.title('Histograma B')
    plt.legend()

    # Ajustar diseño
    plt.tight_layout()

    # Mostrar la figura
    plt.show()

# Leer las imágenes
imagen_original = cv2.imread('imagen.jpg')
imagen_modulo = cv2.imread('imagen_agudizada.jpg')
imagen_procesada = cv2.imread('imagen_realzada.jpg')  # Cambiar el nombre de la imagen si es diferente

# Mostrar los histogramas de las tres imágenes en una sola gráfica
mostrar_histogramas(imagen_original, imagen_modulo, imagen_procesada)



# In[ ]:




