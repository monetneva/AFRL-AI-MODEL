import cv2
import numpy as np
from ultralytics import YOLO
import sys

# Agregar ruta del repo Deep SORT (NO pongas el path al archivo .pb aquí)
sys.path.append('/Users/nicolasgonzalez/Desktop/deep_sort-master 2')

from deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort import nn_matching
from deep_sort.detection import Detection
from tools import generate_detections as gdet


class Track:
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox


class Tracker:
    def __init__(self):
        max_cosine_distance = 0.4
        nn_budget = None
        encoder_model_filename = '/Users/nicolasgonzalez/Desktop/Model/mars-small128.pb'
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric)
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)
        self.tracks = []

    def update(self, frame, detections):
        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])
            self.update_tracks()
            return

        bboxes = np.asarray([d[:-1] for d in detections])
        bboxes[:, 2:] -= bboxes[:, 0:2]
        scores = [d[-1] for d in detections]
        features = self.encoder(frame, bboxes)

        dets = [Detection(b, s, f) for b, s, f in zip(bboxes, scores, features)]
        self.tracker.predict()
        self.tracker.update(dets)
        self.update_tracks()

    def update_tracks(self):
        self.tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            self.tracks.append(Track(track.track_id, bbox))


# Inicializar
video_path = '/Users/nicolasgonzalez/Desktop/Model/data/d006sV2r06p0520211006.MOV'
cap = cv2.VideoCapture(video_path)
model = YOLO("/Users/nicolasgonzalez/Desktop/Model/e2v2.pt")
tracker = Tracker()
lost_vehicles = {}

FPS = 30
FRAME_THRESHOLD = FPS * 5  # 5 segundos = 150 frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []

    # Detectar autos
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf.item()
        cls = int(box.cls[0])
        label = model.names[cls]

        if label == "vehicle" or label == "car":
            detections.append([x1, y1, x2, y2, conf])

    # Actualizar tracker
    tracker.update(frame, detections)

    active_ids = set()

    # Dibujar tracks activos
    for track in tracker.tracks:
        x1, y1, x2, y2 = map(int, track.bbox)
        track_id = track.track_id
        active_ids.add(track_id)

        if track_id in lost_vehicles:
            del lost_vehicles[track_id]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Verificar vehículos que desaparecieron
    for prev_track in tracker.tracker.tracks:
        if not prev_track.is_confirmed() or prev_track.time_since_update <= 1:
            continue

        track_id = prev_track.track_id
        if track_id not in active_ids and track_id not in lost_vehicles:
            print(f"[ALERTA] El vehículo ID {track_id} ha desaparecido.")
            lost_vehicles[track_id] = (prev_track.to_tlbr(), 0)

    # Dibujar vehículos desaparecidos (amarillo o rojo)
    for track_id, (bbox, counter) in list(lost_vehicles.items()):
        x1, y1, x2, y2 = map(int, bbox)

        if counter < FRAME_THRESHOLD:
            color = (0, 255, 255)  # Amarillo
        else:
            color = (0, 0, 255)    # Rojo

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Desaparecido ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if counter > 300:  # Eliminar luego de 10 segundos (aprox.)
            del lost_vehicles[track_id]
        else:
            lost_vehicles[track_id] = (bbox, counter + 1)

    cv2.imshow("Tracking con YOLOv8 + Deep SORT", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
# import cv2
# from ultralytics import YOLO

# path = '/Users/nicolasgonzalez/Desktop/Model/data/d006sV2r06p0520211006.MOV'
# model = YOLO('yolov8n.pt')
# cap = cv2.VideoCapture(path)

# import cv2
# from ultralytics import YOLO

# model = YOLO("e2v2.pt", verbose=False) 

# # Abrir video grabado
# cap = cv2.VideoCapture(path)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Ejecutar detección
#     results = model(frame)

#     for result in results:
#         boxes = result.boxes
#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             conf = box.conf.item()
#             cls = int(box.cls[0])
#             label = model.names[cls]

#             # Filtrar solo autos (opcional)
#             if label == "vehicle":
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     # Mostrar resultado
#     cv2.imshow("Video con detección", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()
