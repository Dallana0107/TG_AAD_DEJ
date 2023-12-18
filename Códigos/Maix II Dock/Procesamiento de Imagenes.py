from maix import display, camera
while True:
    display.show(camera.capture())

from maix import camera, display
import os
import time

camera.config(size=(1080, 1080))
#carpeta donde se guardarán las imágenes
ruta_carpeta = '/root/lizet'

# Inicializar un contador para los nombres de archivo
contador = 1

#Esperar 3 segundos antes de comenzar a capturar imágenes
time.sleep(3)

while True:
    # Capturar una imagen
    img = camera.capture()
    
    # Generar un nombre de archivo único con el contador
    nombre_archivo = f'imagen{contador}.jpg'
    ruta_completa = os.path.join(ruta_carpeta, nombre_archivo)
    
    # Guardar la imagen en la carpeta con el nombre generado
    img.save(ruta_completa)
    
    # Mostrar la imagen en la pantalla (opcional)
    display.show(img)
    
    #Incrementar el contador 
    contador += 1
    
    # Esperar 3 segundos antes de capturar la siguiente imagen
    time.sleep(3)

import cv2
import numpy as np
import os

# Ruta de la carpeta de procesamiento
carpeta_procesamiento = '/root/proce/'

# Cargar la imagen en formato BGR (OpenCV carga las imágenes en formato BGR por defecto)
imagen = cv2.imread('/root/liz/imagen9.jpg')

# Separa los canales de color (R, G, B) de la imagen
canal_b, canal_g, canal_r = cv2.split(imagen)

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

# Ruta y nombre de archivo para guardar la imagen ecualizada
nombre_archivo_ecualizado = 'imagen1.jpg'
ruta_ecualizado = os.path.join(carpeta_procesamiento, nombre_archivo_ecualizado)

# Guardar la imagen ecualizada en la misma carpeta que tu cuaderno de Jupyter
cv2.imwrite(nombre_archivo_ecualizado, imagen_corregida_rgb)
# Guardar la imagen ecualizada en la carpeta de procesamiento
cv2.imwrite(ruta_ecualizado, imagen_corregida_rgb)

# Imprimir un mensaje de confirmación con la carpeta de procesamiento
print(f'Imagen ecualizada guardada como "{nombre_archivo_ecualizado}" en la carpeta de procesamiento.')


# Cargar la imagen agudizada
imagen_agudizacion = cv2.imread(nombre_archivo_ecualizado)

# Crear el kernel de agudización
kernel_agudizacion = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])

# Aplicar el filtro de agudización
imagen_agudizada = cv2.filter2D(imagen_agudizacion, -1, kernel_agudizacion)

# Ruta y nombre de archivo para guardar la imagen agudizada
nombre_archivo_agudizada = 'imagen2.jpg'
ruta_agudizada = os.path.join(carpeta_procesamiento, nombre_archivo_agudizada)

# Guardar la imagen ecualizada en la misma carpeta que tu cuaderno de Jupyter
cv2.imwrite(nombre_archivo_agudizada, imagen_agudizada)
# Guardar la imagen agudizada en la carpeta de procesamiento
cv2.imwrite(ruta_agudizada, imagen_agudizada)

# Imprimir un mensaje de confirmación con la carpeta de procesamiento
print(f'Imagen agudizada guardada como "{nombre_archivo_agudizada}" en la carpeta de procesamiento.')

# Cargar la imagen suavizada
imagen_suavizada = cv2.imread(nombre_archivo_agudizada)

# Crear el kernel de suavizado (filtro de promedio)
kernel_suavizado = np.array([[1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]]) / 9.0  # Normalizar el kernel para que la suma sea 1

# Aplicar el filtro de suavizado
imagen_suavizada = cv2.filter2D(imagen_suavizada, -1, kernel_suavizado)

# Ruta y nombre de archivo para guardar la imagen suavizada
nombre_archivo_suavizada = 'imagen3.jpg'
ruta_suavizada = os.path.join(carpeta_procesamiento, nombre_archivo_suavizada)

# Guardar la imagen ecualizada en la misma carpeta que tu cuaderno de Jupyter
cv2.imwrite(nombre_archivo_suavizada, imagen_suavizada)
# Guardar la imagen suavizada en la carpeta de procesamiento
cv2.imwrite(ruta_suavizada, imagen_suavizada)

# Imprimir un mensaje de confirmación con la carpeta de procesamiento
print(f'Imagen suavizada guardada como "{nombre_archivo_suavizada}" en la carpeta de procesamiento.')


# Cargar la imagen realzada
imagen_realzada = cv2.imread(nombre_archivo_suavizada)

# Crear el kernel de realce de características
kernel_realce = np.array([[0, -0.5, 0],
                          [-0.5,  3, -0.5],
                          [0, -0.5, 0]])

# Aplicar el filtro de realce de características
imagen_realzada = cv2.filter2D(imagen_realzada, -1, kernel_realce)

# Ruta y nombre de archivo para guardar la imagen realzada
nombre_archivo_realzada = 'imagen4.jpg'
ruta_realzada = os.path.join(carpeta_procesamiento, nombre_archivo_realzada)
# Guardar la imagen ecualizada en la misma carpeta que tu cuaderno de Jupyter
cv2.imwrite(nombre_archivo_realzada, imagen_realzada)
# Guardar la imagen realzada en la carpeta de procesamiento
cv2.imwrite(ruta_realzada, imagen_realzada)

# Imprimir un mensaje de confirmación con la carpeta de procesamiento
print(f'Imagen realzada guardada como "{nombre_archivo_realzada}" en la carpeta de procesamiento.')


