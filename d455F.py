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

# Biblioteca oficial da Intel para comunicação com câmeras RealSense
import pyrealsense2 as rs
# Usado para manipular os frames como arrays (matrizes) de pixels
import numpy as np
# Usado para exibir a janela de vídeo e desenhar textos/formas na imagem
import cv2

# --- Configuração do pipeline ---
# O "pipeline" é o objeto que gerencia a captura de dados da câmera
pipeline = rs.pipeline()

# O "config" define quais streams (fluxos de dados) queremos ativar
config = rs.config()

# Ativa o stream de profundidade (depth): resolução 640x480, formato Z16
# (cada pixel guarda a distância em milímetros), 30 quadros por segundo
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Ativa o stream de cor (RGB): resolução 640x480, formato BGR8
# (padrão de cores usado pelo OpenCV), 30 quadros por segundo
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Inicia efetivamente a câmera com as configurações acima
profile = pipeline.start(config)

# O sensor de profundidade e o sensor de cor ficam em posições
# ligeiramente diferentes na câmera. O "align" recalcula o stream de
# profundidade para que cada pixel de depth corresponda ao mesmo
# pixel do stream de cor (fundamental para medir a distância no
# ponto certo da imagem colorida).
align = rs.align(rs.stream.color)

try:
    # Loop principal: roda continuamente até o usuário apertar 'q'
    while True:
        # Espera e captura um novo conjunto de frames (depth + color)
        frames = pipeline.wait_for_frames()

        # Aplica o alinhamento entre os dois streams
        aligned_frames = align.process(frames)

        # Extrai separadamente o frame de profundidade e o de cor
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Se algum dos frames não chegou corretamente, pula esta iteração
        if not depth_frame or not color_frame:
            continue

        # Converte o frame de cor (formato da RealSense) para um
        # array numpy, que é o formato que o OpenCV entende
        color_image = np.asanyarray(color_frame.get_data())

        # Descobre a altura (h) e largura (w) da imagem
        h, w = color_image.shape[:2]

        # Calcula as coordenadas do pixel central da imagem
        cx, cy = w // 2, h // 2

        # Consulta, no frame de profundidade, a distância (em metros)
        # até o objeto que está no pixel central (cx, cy)
        distancia = depth_frame.get_distance(cx, cy)

        # Desenha um círculo verde preenchido no ponto central,
        # apenas para indicar visualmente onde a distância é medida
        cv2.circle(color_image, (cx, cy), 6, (0, 255, 0), -1)

        # Monta o texto com a distância formatada com 2 casas decimais
        texto = f"Distancia: {distancia:.2f} m"

        # Escreve o texto no canto superior esquerdo da imagem
        cv2.putText(
            color_image, texto, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        # Exibe a imagem (com o círculo e o texto) em uma janela
        cv2.imshow("RealSense D455f - Camera + Distancia", color_image)

        # Aguarda 1 ms por uma tecla; se for 'q', encerra o loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Bloco executado sempre ao sair do loop (mesmo em caso de erro):
    # garante que a câmera seja liberada corretamente
    pipeline.stop()
    # Fecha todas as janelas abertas pelo OpenCV
    cv2.destroyAllWindows()