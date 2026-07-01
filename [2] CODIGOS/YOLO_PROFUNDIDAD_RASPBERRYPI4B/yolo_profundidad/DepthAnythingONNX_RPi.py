import os
from pathlib import Path

# Reduce mensajes innecesarios de ONNXRuntime antes de crear la sesión.
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

import cv2
import numpy as np
import onnxruntime as ort


class DepthAnythingONNX:
    """
    Ejecución de Depth Anything V2 en formato ONNX para Raspberry Pi 4B.

    Entrada:
        frame_bgr: imagen en formato BGR de OpenCV.

    Salida:
        depth_norm: mapa de profundidad relativa normalizado en [0, 1]
                    y redimensionado al tamaño original de frame_bgr.

    Nota importante:
        El modelo depth_anything_v2_vits.onnx usado normalmente en este proyecto
        espera entrada fija 518x518. Esta clase lee el tamaño directamente desde
        el ONNX para evitar errores del tipo:
        Got: 256 Expected: 518.
    """

    def __init__(self, model_path, input_size=518, num_threads=2):
        self.model_path = Path(model_path)
        self.input_size_default = int(input_size)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo ONNX: {self.model_path}\n"
                "Coloca depth_anything_v2_vits.onnx dentro de la carpeta models/."
            )

        if num_threads is not None and int(num_threads) > 0:
            os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
            os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_threads))
            os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
            os.environ.setdefault("NUMEXPR_NUM_THREADS", str(num_threads))

        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3

        if num_threads is not None and int(num_threads) > 0:
            sess_options.intra_op_num_threads = int(num_threads)
            sess_options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        input_info = self.session.get_inputs()[0]
        output_info = self.session.get_outputs()[0]

        self.input_name = input_info.name
        self.output_name = output_info.name

        self.input_h, self.input_w = self._resolver_tamano_entrada(input_info.shape)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        print("[INFO] Depth Anything ONNX cargado correctamente")
        print(f"[INFO] Modelo: {self.model_path}")
        print(f"[INFO] Entrada ONNX: {self.input_name}")
        print(f"[INFO] Salida ONNX: {self.output_name}")
        print(f"[INFO] Tamaño esperado por el modelo: {self.input_w}x{self.input_h}")
        print(f"[INFO] Hilos CPU ONNX: {num_threads}")

    def _resolver_tamano_entrada(self, input_shape):
        """
        Resuelve el tamaño real de entrada del modelo.
        Casos típicos:
            [1, 3, 518, 518]
            ['batch', 3, 'height', 'width']
        """
        if len(input_shape) == 4:
            h_model = input_shape[2]
            w_model = input_shape[3]

            if isinstance(h_model, int) and isinstance(w_model, int):
                return int(h_model), int(w_model)

        return self.input_size_default, self.input_size_default

    def preprocess(self, frame_bgr):
        """
        Preprocesamiento para Depth Anything V2:
            BGR -> RGB
            resize -> tamaño esperado por el ONNX
            normalización ImageNet
            HWC -> CHW -> NCHW
        """
        if frame_bgr is None:
            raise ValueError("frame_bgr es None. No se recibió una imagen válida.")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(
            frame_rgb,
            (self.input_w, self.input_h),
            interpolation=cv2.INTER_CUBIC,
        )

        img = resized.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std

        blob = np.transpose(img, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        return blob.astype(np.float32)

    def infer(self, frame_bgr):
        """
        Ejecuta Depth Anything sobre una imagen BGR.

        Retorna:
            depth_norm: matriz float32 en [0, 1], con el mismo tamaño que frame_bgr.
        """
        if frame_bgr is None:
            raise ValueError("frame_bgr es None. No se recibió una imagen válida.")

        h_original, w_original = frame_bgr.shape[:2]
        blob = self.preprocess(frame_bgr)

        output = self.session.run(
            [self.output_name],
            {self.input_name: blob},
        )[0]

        depth = np.squeeze(output).astype(np.float32)

        if depth.ndim != 2:
            raise RuntimeError(f"Salida ONNX inesperada. Forma obtenida: {output.shape}")

        depth = cv2.resize(
            depth,
            (w_original, h_original),
            interpolation=cv2.INTER_CUBIC,
        )

        dmin = float(np.min(depth))
        dmax = float(np.max(depth))

        if dmax - dmin < 1e-8:
            depth_norm = np.zeros_like(depth, dtype=np.float32)
        else:
            depth_norm = (depth - dmin) / (dmax - dmin)

        return depth_norm.astype(np.float32)

    def infer_colormap(self, frame_bgr):
        """
        Ejecuta inferencia y devuelve:
            depth_norm: mapa normalizado [0, 1]
            depth_color: mapa coloreado BGR
        """
        depth_norm = self.infer(frame_bgr)
        depth_color = colorizar_profundidad(depth_norm)
        return depth_norm, depth_color

    def calcular_profundidad_roi(self, depth_norm, x1, y1, x2, y2, margen=0.15):
        """Wrapper orientado a objetos para calcular profundidad relativa en una ROI."""
        return valor_roi_profundidad(depth_norm, x1, y1, x2, y2, margen=margen)


def colorizar_profundidad(depth_norm):
    """
    Convierte un mapa de profundidad normalizado [0, 1]
    a imagen BGR coloreada para visualización.
    """
    if depth_norm is None:
        raise ValueError("depth_norm es None.")

    depth_uint8 = np.clip(depth_norm * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)


def valor_roi_profundidad(depth_norm, x1, y1, x2, y2, margen=0.15):
    """
    Calcula una medida robusta de profundidad relativa dentro de una caja.

    Se usa la mediana de una región interna para reducir el efecto de bordes,
    fondo y ruido cerca del contorno de la caja de YOLO.
    """
    if depth_norm is None:
        return None

    h, w = depth_norm.shape[:2]

    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h))

    if x2 <= x1 or y2 <= y1:
        return None

    dx = int((x2 - x1) * margen)
    dy = int((y2 - y1) * margen)

    xa = max(0, x1 + dx)
    xb = min(w, x2 - dx)
    ya = max(0, y1 + dy)
    yb = min(h, y2 - dy)

    if xb <= xa or yb <= ya:
        xa, xb, ya, yb = x1, x2, y1, y2

    roi = depth_norm[ya:yb, xa:xb]

    if roi.size == 0:
        return None

    return float(np.median(roi))


def guardar_mapa_profundidad(depth_norm, ruta_salida):
    """Guarda el mapa de profundidad coloreado."""
    depth_color = colorizar_profundidad(depth_norm)
    cv2.imwrite(str(ruta_salida), depth_color)
    return depth_color


def crear_comparacion(frame_bgr, depth_color, ruta_salida):
    """Crea una imagen: original | profundidad."""
    h, w = frame_bgr.shape[:2]

    depth_color = cv2.resize(
        depth_color,
        (w, h),
        interpolation=cv2.INTER_AREA,
    )

    comparacion = np.hstack([frame_bgr, depth_color])
    cv2.imwrite(str(ruta_salida), comparacion)
    return comparacion


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prueba de Depth Anything V2 ONNX en Raspberry Pi."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="models/depth_anything_v2_vits.onnx",
        help="Ruta del modelo ONNX.",
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Ruta de una imagen de prueba.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="depth_test.png",
        help="Ruta de salida del mapa de profundidad.",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="Número de hilos CPU para ONNXRuntime.",
    )

    args = parser.parse_args()

    depth_model = DepthAnythingONNX(args.model, num_threads=args.threads)

    if args.image is None:
        print("[INFO] Módulo cargado correctamente.")
        print("[INFO] Para probar con imagen usa:")
        print("python DepthAnythingONNX_RPi.py --image prueba.jpg")
    else:
        frame = cv2.imread(args.image)

        if frame is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {args.image}")

        _, depth_color = depth_model.infer_colormap(frame)
        cv2.imwrite(args.output, depth_color)
        print(f"[INFO] Mapa de profundidad guardado en: {args.output}")
