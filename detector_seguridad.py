import cv2
import torch
import numpy as np
import subprocess

# Se sobreescribe subprocess.check_output para evitar que intente obtener la GPU con 'nvidia-smi'
subprocess.check_output = lambda *a, **kw: b'custom'

# Importación del modelo YOLOv7 y utilidades
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# Función para ajustar el tamaño de la imagen a la entrada del modelo manteniendo la relación de aspecto
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = im.shape[:2]  # Altura y ancho original
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # Escala de redimensionamiento
    if not scaleup:
        r = min(r, 1.0)  # Evita agrandar imágenes pequeñas

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # Nuevo tamaño sin relleno
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # Relleno requerido
    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # Ajuste a múltiplos de 32 (para modelos YOLO)

    dw /= 2  # División del relleno para ambos lados
    dh /= 2

    # Redimensionar imagen y agregar bordes
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, r, (dw, dh)

def main():
    device = select_device('')  # Selección de GPU si está disponible, si no, CPU

    # Cargar el modelo entrenado YOLOv7
    model_path = 'yolov7.pt'
    model = attempt_load(model_path, map_location=device)
    model.eval()

    # Cargar el video de entrada
    video_path = 'videobus.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video {video_path}")

    # Obtener dimensiones y configuración del video
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Preparar el archivo de salida
    out = cv2.VideoWriter('video_salida.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # Diccionario de clases relevantes que se desean detectar
    clases = {
        0: 'persona',
        24: 'mochila',
        65: 'puerta',
        15: 'brazo',
        16: 'mano',
        17: 'pierna',
        18: 'zapato'
    }

    riesgo_activo = False  # Indica si hay un pasajero en la zona de peligro

    while True:
        ret, frame_orig = cap.read()
        if not ret:
            print("Fin del video o no se pudo leer el frame")
            break

        h, w = frame_orig.shape[:2]

        # Definir zona de riesgo en el centro inferior del frame (por ejemplo, cerca de las puertas)
        box_w, box_h = w // 5, h // 5
        x1_zone = (w - box_w) // 2 - box_w // 4
        x2_zone = (w + box_w) // 2 + box_w // 4
        y2_zone = (h + box_h) // 2
        y1_zone = y2_zone - 3 * box_h

        # Dibujar zona de riesgo en rojo
        cv2.rectangle(frame_orig, (x1_zone, y1_zone), (x2_zone, y2_zone), (0, 0, 255), 2)

        # Preprocesar imagen para el modelo
        img, ratio, (dw, dh) = letterbox(frame_orig, new_shape=640)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().to(device) / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

        with torch.no_grad():
            pred = model(img_tensor)[0]  # Inferencia

        pred = non_max_suppression(pred, 0.25, 0.45)  # Filtrado de detecciones por confianza e IoU
        riesgo_detectado = False  # Marca si hay detección dentro de zona de riesgo

        # Procesar detecciones
        for det in pred:
            if det is not None and len(det):
                # Escalar coordenadas al tamaño original del frame
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], frame_orig.shape).round()

                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    class_id = int(cls)
                    nombre = clases.get(class_id, None)

                    if nombre:
                        # Dibujar caja y etiqueta de clase
                        cv2.rectangle(frame_orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f'{nombre} {conf:.2f}'
                        cv2.putText(frame_orig, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # Verificar si la caja colisiona con la zona de riesgo
                        if not (x2 < x1_zone or x1 > x2_zone or y2 < y1_zone or y1 > y2_zone):
                            riesgo_detectado = True

        # Control de estado para mostrar alertas
        if riesgo_detectado and not riesgo_activo:
            print("⚠️ Pasajero entre las puertas ⚠️")
            riesgo_activo = True
        elif not riesgo_detectado and riesgo_activo:
            print("✅ Seguro cerrar")
            riesgo_activo = False

        # Mostrar mensaje visual en pantalla
        mensaje = "Pasajero entre las puertas" if riesgo_activo else "Seguro cerrar"
        color_texto = (0, 0, 255) if riesgo_activo else (0, 255, 0)
        cv2.putText(frame_orig, mensaje, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color_texto, 2)

        # Mostrar frame procesado y guardar en archivo de salida
        cv2.imshow('Detección YOLOv7', frame_orig)
        out.write(frame_orig)

        # Salida con tecla ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Liberar recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
