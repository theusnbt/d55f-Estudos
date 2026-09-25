import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline() # controla a captura de imagens da camera
config = rs.config() # cria uma variavel que vai abrir as configurações da câmera


config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # Profundidade da camera
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # Coloração da camera


profile = pipeline.start(config) # inciando a camera com as configs definidas
align = rs.align(rs.stream.color) # cria um alinhamento na câmera, fazendo com que a mesma tenha cor e imagem

try:
    while True:
        frames = pipeline.wait_for_frames() # espera oor novos frames (ações) da câmera     
        aligned_frames = align.process(frames) # quando recebidos, ele alinha os frames para estarem corretos


        depth_frame = aligned_frames.get_depth_frame() # definindo variavel que ira receber os dados de profundidade da camera
        color_frame = aligned_frames.get_color_frame() # definindo variavel que ira receber a imagem colorida

        if not depth_frame or not color_frame:
            continue
        # Bloco acima: se não receber profundidade ou a cor, ele volta para o inicio do loop para tentar novamente

        color_image = np.asanyarray(color_frame.get_data()) # transforma os dados de imagem colorida e transforma em um array legivel a linguagem 

        h, w = color_image.shape[:2] # shape: informa as dimensões da imagem e o h e w recene, altura e largura por meio do [:2]
        cx, cy = w // 2, h // 2 # calcula o ponto central da imagem sendo cx e cy o centro pegando metade de H e W

        distancia = depth_frame.get_distance(cx, cy) #pega a distancia no ponto central da imagem

        cv2.circle(color_image, (cx, cy), 6, (0, 255, 0), -1) # define um circulo no ponto central da imagem

        texto = f"Distância: {distancia:.2f}M"
        cv2.putText( # inserindo texto dentro da imagem
            color_image, texto, (20, 40), # definindo em qual imagem o texto vai aparecer
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2 # defuinindo a fonte, tamanho, cor e espessura
        ) 
        cv2.imshow("Realsense d455f - camera + distancia", color_image) # exibe uma imagem na tela com texto e cor

        if cv2.waitKey(1) & 0xFF == ord('q'): #ela fecha a janela quando a tecla q é pressionada
            break # sai do laço while, fazendo o mesmo ficar false
finally:
    pipeline.stop() # para a camera, no caso desliga ela
    cv2.destroyAllWindows # fecha as janela da imagem
