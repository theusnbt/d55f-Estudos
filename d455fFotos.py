import os
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import pyrealsense2 as rs


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("D455f - Captura de Fotos")
        self.root.geometry("760x560")
        self.root.minsize(640, 480)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.current_frame = None
        self.preview_image = None

        self.frame_label = tk.Label(root, bg="black")
        self.frame_label.pack(padx=12, pady=(12, 8), fill="both", expand=True)

        self.status_var = tk.StringVar(value="Iniciando câmera...")
        self.status_label = tk.Label(root, textvariable=self.status_var, anchor="w")
        self.status_label.pack(fill="x", padx=12, pady=(0, 8))

        button_frame = tk.Frame(root)
        button_frame.pack(padx=12, pady=(0, 12), fill="x")

        self.capture_button = tk.Button(
            button_frame,
            text="Tirar foto",
            command=self.capture_photo,
            width=18,
            height=2,
            font=("Arial", 11, "bold"),
            bg="#2E86DE",
            fg="white",
        )
        self.capture_button.pack(side="left", padx=(0, 12))

        self.save_button = tk.Button(
            button_frame,
            text="Salvar como...",
            command=self.save_photo,
            width=18,
            height=2,
            font=("Arial", 11),
        )
        self.save_button.pack(side="left")

        self.start_camera()

    def start_camera(self):
        try:
            self.pipeline.start(self.config)
            for _ in range(30):
                self.pipeline.wait_for_frames()

            self.status_var.set("Câmera pronta. Preview em tempo real.")
            self.update_preview()
        except Exception as exc:
            self.status_var.set(f"Erro ao iniciar a câmera: {exc}")
            messagebox.showerror("Erro da câmera", f"Não foi possível abrir a câmera RealSense.\n\n{exc}")

    def update_preview(self):
        if self.pipeline is None:
            return

        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                self.root.after(30, self.update_preview)
                return

            color_image = np.asanyarray(color_frame.get_data())
            self.current_frame = color_image
            self.display_frame(color_image)

        except Exception as exc:
            self.status_var.set(f"Erro no preview: {exc}")
            return

        self.root.after(30, self.update_preview)

    def display_frame(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (640, 480))

        _, buffer = cv2.imencode(".ppm", image_rgb)
        photo = tk.PhotoImage(data=buffer.tobytes())

        self.frame_label.configure(image=photo)
        self.frame_label.image = photo

    def capture_photo(self):
        if self.current_frame is None:
            self.status_var.set("Nenhuma imagem capturada ainda.")
            return

        default_name = f"foto_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filename = filedialog.asksaveasfilename(
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("Todos os arquivos", "*.*")],
            title="Salvar foto",
        )

        if not filename:
            self.status_var.set("Captura cancelada.")
            return

        cv2.imwrite(filename, cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR))
        self.status_var.set(f"Foto salva em: {os.path.abspath(filename)}")

    def save_photo(self):
        self.capture_photo()

    def on_close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()