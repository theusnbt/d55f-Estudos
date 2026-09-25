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

import pyrealsense2 as rs   # controla a câmera Intel RealSense (D455f)
import numpy as np          # manipulação de arrays/matrizes (imagens)
import cv2                  # exibição de janela, desenho de texto/overlay
from collections import deque   # fila de tamanho fixo (histórico de frames)
from ultralytics import YOLO    # carrega e roda o modelo YOLOv8-Pose

# ---------------- Configurações ----------------

HISTORICO_FRAMES = 10      # quantos frames de histórico guardamos do quadril
LIMIAR_PULO = 0.06         # variação mínima (proporção da altura da imagem) para contar como pulo
COOLDOWN_PULO = 15         # frames de espera antes de poder contar outro pulo
CONFIANCA_MINIMA = 0.5     # confiança mínima de cada ponto do corpo para ser considerado válido

# Índices dos pontos do corpo no formato COCO usado pelo YOLOv8-Pose
# (cada pessoa detectada retorna um array de 17 pontos nessa ordem fixa)
NARIZ = 0
OMBRO_ESQ, OMBRO_DIR = 5, 6
PULSO_ESQ, PULSO_DIR = 9, 10
QUADRIL_ESQ, QUADRIL_DIR = 11, 12

# Carrega o modelo (baixa automaticamente na primeira vez, se não existir localmente)
modelo = YOLO("yolov8n-pose.pt")

# ---------------- Câmera ----------------

pipeline = rs.pipeline()          # objeto que gerencia o fluxo de dados da câmera
config = rs.config()              # objeto de configuração do stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# ^ ativa o stream de cor: resolução 640x480, formato BGR (padrão OpenCV), 30 fps
pipeline.start(config)            # inicia a câmera com essas configurações

# ---------------- Estado da detecção ----------------

historico_quadril_y = deque(maxlen=HISTORICO_FRAMES)
# ^ guarda as últimas N posições verticais (normalizadas) do quadril;
#   ao encher, descarta automaticamente o valor mais antigo

cooldown_atual = 0        # contador regressivo; >0 significa "não contar pulo agora"
contador_pulos = 0        # total de pulos detectados
maos_levantadas = False   # estado atual (frame a frame) das mãos


def ponto_valido(keypoints_conf, indice):
    """Confere se o ponto do corpo foi detectado com confiança suficiente."""
    return keypoints_conf[indice] >= CONFIANCA_MINIMA


try:
    while True:
        frames = pipeline.wait_for_frames()      # espera o próximo conjunto de frames
        color_frame = frames.get_color_frame()   # extrai só o frame colorido

        if not color_frame:
            continue  # frame inválido/ausente nesta iteração: pula para a próxima

        color_image = np.asanyarray(color_frame.get_data())  # converte para array NumPy (formato OpenCV)
        h, w = color_image.shape[:2]                          # altura e largura da imagem (usadas para normalizar)

        # Roda a detecção de pose nesse frame (verbose=False evita log poluindo o terminal)
        resultados = modelo.predict(color_image, verbose=False)

        maos_levantadas = False  # reseta a cada frame: só fica True se detectar neste frame

        # Verifica se houve resultado, se contém keypoints e se pelo menos 1 pessoa foi detectada
        if len(resultados) > 0 and resultados[0].keypoints is not None and len(resultados[0].keypoints.xy) > 0:
            # Pega a primeira pessoa detectada
            pontos_xy = resultados[0].keypoints.xy[0].cpu().numpy()      # posições (x, y) em pixels de cada ponto
            pontos_conf = resultados[0].keypoints.conf[0].cpu().numpy()  # confiança de cada ponto (0 a 1)

            # --- Detecção de mãos levantadas ---
            if (ponto_valido(pontos_conf, OMBRO_ESQ) and ponto_valido(pontos_conf, OMBRO_DIR)
                    and ponto_valido(pontos_conf, PULSO_ESQ) and ponto_valido(pontos_conf, PULSO_DIR)):

                ombros_y = (pontos_xy[OMBRO_ESQ][1] + pontos_xy[OMBRO_DIR][1]) / 2  # altura média dos ombros
                pulsos_y = (pontos_xy[PULSO_ESQ][1] + pontos_xy[PULSO_DIR][1]) / 2  # altura média dos pulsos

                # Em coordenadas de imagem, y menor = mais alto na tela
                if pulsos_y < ombros_y:
                    maos_levantadas = True  # pulsos acima dos ombros => mãos levantadas

            # --- Detecção de pulo (baseada na oscilação da altura do quadril) ---
            if ponto_valido(pontos_conf, QUADRIL_ESQ) and ponto_valido(pontos_conf, QUADRIL_DIR):
                quadril_y = (pontos_xy[QUADRIL_ESQ][1] + pontos_xy[QUADRIL_DIR][1]) / 2  # altura média do quadril

                # Guarda a altura do quadril já normalizada pela altura da imagem (0 a 1)
                historico_quadril_y.append(quadril_y / h)

                # Só avalia se não está em cooldown e já há histórico completo (10 frames)
                if cooldown_atual == 0 and len(historico_quadril_y) == HISTORICO_FRAMES:
                    variacao = max(historico_quadril_y) - min(historico_quadril_y)  # amplitude do movimento
                    # Conta pulo se a variação passou do limiar E o quadril está no ponto mais alto agora
                    # (valor mínimo de y = ponto mais alto na imagem)
                    if variacao > LIMIAR_PULO and historico_quadril_y[-1] == min(historico_quadril_y):
                        contador_pulos += 1
                        cooldown_atual = COOLDOWN_PULO  # ativa o cooldown para não recontar o mesmo pulo

            # Desenha o esqueleto detectado por cima da imagem (ajuda a debugar)
            color_image = resultados[0].plot()

        # Decrementa o cooldown a cada frame até liberar a contagem de um novo pulo
        if cooldown_atual > 0:
            cooldown_atual -= 1

        # ---------------- Interface na tela ----------------

        # Desenha o contador de pulos no canto superior esquerdo
        cv2.putText(
            color_image, f"Pulos: {contador_pulos}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )

        # Define cor e texto de status conforme o estado das mãos
        cor_maos = (0, 255, 0) if maos_levantadas else (100, 100, 100)  # verde se levantadas, cinza se não
        texto_maos = "MAOS LEVANTADAS!" if maos_levantadas else "maos abaixadas"
        cv2.putText(
            color_image, texto_maos, (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_maos, 2
        )

        cv2.imshow("Deteccao de Pulo e Maos - D455f (YOLOv8-Pose)", color_image)  # exibe o frame processado

        if cv2.waitKey(1) & 0xFF == ord('q'):  # espera 1ms por tecla; 'q' encerra o loop
            break

finally:
    # Executa sempre, mesmo em caso de erro ou interrupção: libera a câmera e fecha as janelas
    pipeline.stop()
    cv2.destroyAllWindows()