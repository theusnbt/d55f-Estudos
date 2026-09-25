"""
Protótipo: detecta PULO e MÃOS LEVANTADAS usando a câmera D455f
(imagem colorida) e o YOLOv8-Pose para achar os pontos do corpo.

Substitui o MediaPipe, que ainda não tem suporte ao Python 3.14.
O YOLOv8-Pose roda em cima do PyTorch e detecta 17 pontos do corpo
(ombros, pulsos, quadris, etc.) em cada frame.

Na primeira execução, a biblioteca baixa automaticamente o arquivo
de modelo "yolov8n-pose.pt" (alguns MB) da internet.

Requisitos:
    pip install pyrealsense2 opencv-python numpy ultralytics

Uso:
    python3 d455f_pulo_maos.py

Pressione 'q' para sair.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from collections import deque
from ultralytics import YOLO

# ---------------- Configurações ----------------

HISTORICO_FRAMES = 10      # quantos frames de histórico guardamos do quadril
LIMIAR_PULO = 0.06         # variação mínima (proporção da altura da imagem) para contar como pulo
COOLDOWN_PULO = 15         # frames de espera antes de poder contar outro pulo
CONFIANCA_MINIMA = 0.5     # confiança mínima de cada ponto do corpo para ser considerado válido

# Índices dos pontos do corpo no formato COCO usado pelo YOLOv8-Pose
NARIZ = 0
OMBRO_ESQ, OMBRO_DIR = 5, 6
PULSO_ESQ, PULSO_DIR = 9, 10
QUADRIL_ESQ, QUADRIL_DIR = 11, 12

# Carrega o modelo (baixa automaticamente na primeira vez)
modelo = YOLO("yolov8n-pose.pt")

# ---------------- Câmera ----------------

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# ---------------- Estado da detecção ----------------

historico_quadril_y = deque(maxlen=HISTORICO_FRAMES)
cooldown_atual = 0
contador_pulos = 0
maos_levantadas = False


def ponto_valido(keypoints_conf, indice):
    """Confere se o ponto do corpo foi detectado com confiança suficiente."""
    return keypoints_conf[indice] >= CONFIANCA_MINIMA


try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        h, w = color_image.shape[:2]

        # Roda a detecção de pose nesse frame (verbose=False evita log poluindo o terminal)
        resultados = modelo.predict(color_image, verbose=False)

        maos_levantadas = False

        if len(resultados) > 0 and resultados[0].keypoints is not None and len(resultados[0].keypoints.xy) > 0:
            # Pega a primeira pessoa detectada
            pontos_xy = resultados[0].keypoints.xy[0].cpu().numpy()      # posições (x, y) em pixels
            pontos_conf = resultados[0].keypoints.conf[0].cpu().numpy()  # confiança de cada ponto

            if (ponto_valido(pontos_conf, OMBRO_ESQ) and ponto_valido(pontos_conf, OMBRO_DIR)
                    and ponto_valido(pontos_conf, PULSO_ESQ) and ponto_valido(pontos_conf, PULSO_DIR)):

                ombros_y = (pontos_xy[OMBRO_ESQ][1] + pontos_xy[OMBRO_DIR][1]) / 2
                pulsos_y = (pontos_xy[PULSO_ESQ][1] + pontos_xy[PULSO_DIR][1]) / 2

                # Em coordenadas de imagem, y menor = mais alto na tela
                if pulsos_y < ombros_y:
                    maos_levantadas = True

            if ponto_valido(pontos_conf, QUADRIL_ESQ) and ponto_valido(pontos_conf, QUADRIL_DIR):
                quadril_y = (pontos_xy[QUADRIL_ESQ][1] + pontos_xy[QUADRIL_DIR][1]) / 2

                # Guarda a altura do quadril já normalizada pela altura da imagem (0 a 1)
                historico_quadril_y.append(quadril_y / h)

                if cooldown_atual == 0 and len(historico_quadril_y) == HISTORICO_FRAMES:
                    variacao = max(historico_quadril_y) - min(historico_quadril_y)
                    if variacao > LIMIAR_PULO and historico_quadril_y[-1] == min(historico_quadril_y):
                        contador_pulos += 1
                        cooldown_atual = COOLDOWN_PULO

            # Desenha o esqueleto detectado por cima da imagem (ajuda a debugar)
            color_image = resultados[0].plot()

        if cooldown_atual > 0:
            cooldown_atual -= 1

        # ---------------- Interface na tela ----------------

        cv2.putText(
            color_image, f"Pulos: {contador_pulos}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )

        cor_maos = (0, 255, 0) if maos_levantadas else (100, 100, 100)
        texto_maos = "MAOS LEVANTADAS!" if maos_levantadas else "maos abaixadas"
        cv2.putText(
            color_image, texto_maos, (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_maos, 2
        )

        cv2.imshow("Deteccao de Pulo e Maos - D455f (YOLOv8-Pose)", color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()