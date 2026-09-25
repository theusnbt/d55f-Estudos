"""
Mini-jogo de pesca (estilo Stardew Valley) usando a câmera D455f.

Como jogar:
    - Um "peixe" (ponto amarelo) se move sozinho, para cima e para baixo,
      dentro de uma barra na tela.
    - Você controla uma "barra verde" se aproximando ou se afastando
      da câmera: chegar perto sobe a barra, afastar desce a barra.
    - Mantenha o peixe DENTRO da barra verde para encher a barra de
      progresso (lado direito). Se o peixe ficar fora, o progresso cai.
    - Encha o progresso até o fim para pescar. Se o progresso zerar,
      o peixe escapa.

Requisitos:
    pip install pyrealsense2 opencv-python numpy

Uso:
    python3 d455f_pesca.py

Pressione 'r' para reiniciar depois de ganhar/perder.
Pressione 'q' para sair.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import random

# ---------------- Configurações do jogo ----------------

# Distância (em metros) que corresponde ao topo e à base da área de pesca.
# Ajuste esses valores conforme o espaço que você tem na frente da câmera.
DIST_PERTO = 0.4   # ficar mais perto que isso = barra vai pro topo
DIST_LONGE = 1.0   # ficar mais longe que isso = barra vai pro fundo

# Área de pesca (a "vara") na tela
AREA_X = 500
AREA_Y = 40
AREA_LARGURA = 60
AREA_ALTURA = 400

# Barra verde que o jogador controla dentro da área de pesca
BARRA_ALTURA = 90

# Peixe
PEIXE_RAIO = 10
PEIXE_VELOCIDADE = 3  # o quanto o peixe se move por frame em direção ao alvo

# Progresso (barra lateral)
PROGRESSO_MAX = 100
PROGRESSO_INICIAL = 40
GANHO_DENTRO = 0.8     # progresso ganho por frame quando o peixe está na barra
PERDA_FORA = 1.2       # progresso perdido por frame quando o peixe está fora


def distancia_para_y(distancia):
    """Converte a distância medida pela câmera na posição vertical
    da barra verde dentro da área de pesca."""
    distancia = max(DIST_PERTO, min(DIST_LONGE, distancia))

    # perto (DIST_PERTO) deve virar o topo da área; longe (DIST_LONGE) a base
    proporcao = (distancia - DIST_PERTO) / (DIST_LONGE - DIST_PERTO)

    y = AREA_Y + int(proporcao * (AREA_ALTURA - BARRA_ALTURA))
    return y


def novo_alvo_peixe():
    """Sorteia uma nova posição para onde o peixe vai nadar."""
    return random.randint(AREA_Y + PEIXE_RAIO, AREA_Y + AREA_ALTURA - PEIXE_RAIO)


# ---------------- Configuração da câmera ----------------

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

align = rs.align(rs.stream.color)

# ---------------- Estado inicial do jogo ----------------

progresso = PROGRESSO_INICIAL
peixe_y = AREA_Y + AREA_ALTURA // 2
peixe_alvo = novo_alvo_peixe()
estado = "jogando"  # pode virar "ganhou" ou "perdeu"

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # Mede a distância no centro da imagem (é isso que controla a vara)
        h, w = color_image.shape[:2]
        cx, cy = w // 2, h // 2
        distancia = depth_frame.get_distance(cx, cy)

        

        if estado == "jogando":
            # Move a barra verde conforme a distância
            barra_y = distancia_para_y(distancia)

            # Move o peixe em direção ao alvo atual; ao chegar perto,
            # sorteia um novo alvo (faz o peixe "nadar" de forma natural)
            if peixe_y < peixe_alvo:
                peixe_y += PEIXE_VELOCIDADE
            elif peixe_y > peixe_alvo:
                peixe_y -= PEIXE_VELOCIDADE

            if abs(peixe_y - peixe_alvo) < PEIXE_VELOCIDADE:
                peixe_alvo = novo_alvo_peixe()

            # Verifica se o peixe está dentro da barra verde
            peixe_dentro = barra_y <= peixe_y <= barra_y + BARRA_ALTURA

            if peixe_dentro:
                progresso += GANHO_DENTRO
            else:
                progresso -= PERDA_FORA

            progresso = max(0, min(PROGRESSO_MAX, progresso))

            if progresso >= PROGRESSO_MAX:
                estado = "ganhou"
            elif progresso <= 0:
                estado = "perdeu"

        # ---------------- Desenho na tela ----------------

        # Contorno da área de pesca (a "vara")
        cv2.rectangle(
            color_image, (AREA_X, AREA_Y),
            (AREA_X + AREA_LARGURA, AREA_Y + AREA_ALTURA),
            (255, 255, 255), 2
        )

        # Barra verde controlada pela distância
        cor_barra = (0, 255, 0) if estado == "jogando" else (0, 150, 0)
        cv2.rectangle(
            color_image, (AREA_X, barra_y),
            (AREA_X + AREA_LARGURA, barra_y + BARRA_ALTURA),
            cor_barra, -1
        )

        # Peixe (círculo amarelo)
        cv2.circle(
            color_image, (AREA_X + AREA_LARGURA // 2, int(peixe_y)),
            PEIXE_RAIO, (0, 255, 255), -1
        )

        # Barra de progresso, ao lado da área de pesca
        prog_x = AREA_X + AREA_LARGURA + 30
        prog_altura_preenchida = int((progresso / PROGRESSO_MAX) * AREA_ALTURA)
        cv2.rectangle(
            color_image, (prog_x, AREA_Y),
            (prog_x + 25, AREA_Y + AREA_ALTURA),
            (200, 200, 200), 2
        )
        cv2.rectangle(
            color_image,
            (prog_x, AREA_Y + AREA_ALTURA - prog_altura_preenchida),
            (prog_x + 25, AREA_Y + AREA_ALTURA),
            (0, 200, 255), -1
        )

        # Texto de distância (útil para calibrar DIST_PERTO / DIST_LONGE)
        cv2.putText(
            color_image, f"Distancia: {distancia:.2f} m", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

        # Mensagem de vitória/derrota
        if estado == "ganhou":
            cv2.putText(
                color_image, "Peixe fisgado! Aperte 'r' para jogar de novo",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        elif estado == "perdeu":
            cv2.putText(
                color_image, "O peixe escapou... Aperte 'r' para tentar de novo",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )

        cv2.imshow("Pesca - RealSense D455f", color_image)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            break
        elif tecla == ord('r') and estado != "jogando":
            progresso = PROGRESSO_INICIAL
            peixe_y = AREA_Y + AREA_ALTURA // 2
            peixe_alvo = novo_alvo_peixe()
            estado = "jogando"

finally:
    pipeline.stop()
    cv2.destroyAllWindows()