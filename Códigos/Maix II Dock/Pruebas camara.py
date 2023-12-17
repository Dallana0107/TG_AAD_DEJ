from maix import display, camera
while True:
    display.show(camera.capture())

from maix import camera, display
img = camera.capture()
img.save('/root/tmp.jpg')
display.show(img)

from maix import camera, display, image
camera.config(size=(640, 360))
while True:
    img = camera.capture()
    display.show(img)

from maix import camera, display
import os
import time

camera.config(size=(1080, 1080))
#carpeta donde se guardarán las imágenes
ruta_carpeta = '/root/d'

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

from maix import image, display, camera
color = [
        [(0, 100, -128, -23, -128, 127)], #绿色
        [(10, 100, 30, 127, -37, 127)], #红色
        [(40, 100, -25, 42, 7, 127)], #黄色
        [(0, 100, -128, 127, -128, -46)], #蓝色
        ]  # 0.5.0 以后蓝色的 lab 阈值，0.4.9 之前为 [(13, 11, -91, 54, 48, -28)]
font_color = [ # 边框和文字颜色，暂时都用白色
    (255,255,255), # 绿色
    (255,255,255), # 红色
    (255,255,255), # 黄色
    (255,255,255)  # 白色
]
name_color = ('green', 'red', 'yellow', 'blue')
while True:
    img = camera.capture()
    for n in range(0,4):
        blobs = img.find_blobs(color[n])    #在图片中查找lab阈值内的颜色色块
        if blobs:
            for i in blobs:
                if i["w"]>15 and i["h"]>15:
                    img.draw_rectangle(i["x"], i["y"], i["x"] + i["w"], i["y"] + i["h"], 
                                       color=font_color[n], thickness=1) #将找到的颜色区域画出来
                    img.draw_string(i["x"], i["y"], name_color[n], scale = 0.8, 
                              color = font_color[0], thickness = 1) #在红色背景图上写下hello worl
    display.show(img)


from maix import camera, display, image
camera.config(size=(640, 360))
while True:
    img = camera.capture()
    img = img.lens_corr(strength=1.8, zoom=1.0)
    display.show(img)

