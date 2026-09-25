"""
Script simples para a câmera Intel RealSense D455f.
Mostra a imagem colorida em tempo real e a distância (em metros)
do ponto central da imagem.

Requisitos:
    pip install pyrealsense2 opencv-python numpy

Uso:
    python3 d455f_distancia.py

Pressione 'q' para sair.
"""

import pyrealsense2 as rs
import numpy as np
import cv2

# --- Configuração do pipeline ---
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Inicia o stream
profile = pipeline.start(config)

# Alinha o depth ao color (para que os pixels correspondam)
align = rs.align(rs.stream.color)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Converte para arrays numpy
        color_image = np.asanyarray(color_frame.get_data())

        # Coordenadas do centro da imagem
        h, w = color_image.shape[:2]
        cx, cy = w // 2, h // 2

        # Distância em metros no ponto central
        distancia = depth_frame.get_distance(cx, cy)

        # Desenha um círculo e o texto com a distância
        cv2.circle(color_image, (cx, cy), 6, (0, 255, 0), -1)
        texto = f"Distancia: {distancia:.2f} m"
        cv2.putText(
            color_image, texto, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        cv2.imshow("RealSense D455f - Camera + Distancia", color_image)

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()