"""
Script para a câmera Intel RealSense D455f.
Abre uma janela ao vivo mostrando a imagem colorida e a distância
(em metros) do ponto central. Ao apertar 's', salva a imagem atual
(já com a distância desenhada) em um arquivo .png.

Requisitos:
    pip install pyrealsense2 opencv-python numpy

Uso:
    python3 d455f_foto_distancia.py

Pressione 's' para salvar uma foto.
Pressione 'q' para sair.
"""

import pyrealsense2 as rs  # fala com a câmera
import numpy as np         # organiza a imagem como números
import cv2                  # mostra e salva a imagem

# Cria o gerenciador da câmera
pipeline = rs.pipeline()
config = rs.config()

# Pede a distância (depth) e a imagem colorida (color), 640x480, 30 fps
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Liga a câmera
pipeline.start(config)

# Encaixa a imagem de distância com a imagem colorida
align = rs.align(rs.stream.color)

contador_fotos = 0  # usado para não sobrescrever a foto anterior

try:
    while True:  # loop ao vivo
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # Acha o pixel do centro
        h, w = color_image.shape[:2]
        cx, cy = w // 2, h // 2

        # Mede a distância nesse ponto
        distancia = depth_frame.get_distance(cx, cy)

        # Desenha o marcador e o texto (isso fica gravado na foto também,
        # já que desenhamos direto em cima da imagem antes de salvar)
        cv2.circle(color_image, (cx, cy), 6, (0, 255, 0), -1)
        texto = f"Distancia: {distancia:.2f} m"
        cv2.putText(
            color_image, texto, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        # Mostra a janela ao vivo
        cv2.imshow("RealSense D455f - Camera + Distancia", color_image)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord('s'):
            # Salva a imagem exatamente como está sendo mostrada na tela
            contador_fotos += 1
            nome_arquivo = f"foto_{contador_fotos}.png"
            cv2.imwrite(nome_arquivo, color_image)
            print(f"Foto salva como {nome_arquivo}")

        elif tecla == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()