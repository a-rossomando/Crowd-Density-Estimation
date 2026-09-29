"""Scorre il video e salva screenshot automatici quando trova allarmi o gruppi."""
import cv2

from crowd_density.config import PipelineConfig
from crowd_density.pipeline import CrowdDensityPipeline

VIDEO = r"c:\Users\alero\Desktop\unisi\ADIP\registrazioni_sala_statue\rec_20260925_110750.mp4"
BG_IMAGE = r"c:\Users\alero\Desktop\unisi\ADIP\progetto\Crowd-Density-Estimation\background_model.png"
OUT_DIR = r"c:\Users\alero\Desktop\unisi\ADIP\progetto\Crowd-Density-Estimation"

config = PipelineConfig(display_output=False)
pipeline = CrowdDensityPipeline(config=config)
pipeline.enable_background_filter_from_image(image_path=BG_IMAGE)

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {total} frame, {fps:.0f} fps, durata {total/fps:.0f}s")
print("Cerco frame con allarme assembramento...")

saved_alarm = 0
frame_id = 0

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = pipeline.process_frame(frame=frame_rgb, frame_id=frame_id, timestamp=frame_id / fps)

    n = len(result.head_detections)

    # Salva SOLO screenshot con allarme
    if result.alarms and saved_alarm < 3:
        overlay = frame_bgr.copy()
        for det in result.head_detections:
            b = det.bbox
            cv2.rectangle(overlay, (b.x, b.y), (b.x + b.width, b.y + b.height), (0, 255, 0), 2)
        h_img = overlay.shape[0]
        cv2.putText(overlay, f"Conteggio: {n}", (10, h_img - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(overlay, "ALLARME ASSEMBRAMENTO", (10, h_img - 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        path = f"{OUT_DIR}/screenshot_allarme_{saved_alarm + 1}.png"
        cv2.imwrite(path, overlay)
        print(f"Frame {frame_id} ({frame_id/fps:.0f}s): {n} persone, ALLARME! -> salvato")
        saved_alarm += 1

    if saved_alarm >= 1:
        break

    frame_id += 1
    if frame_id % 5000 == 0:
        print(f"  ... frame {frame_id}/{total} ({frame_id/fps:.0f}s), nessun allarme ancora")

cap.release()

if saved_alarm == 0:
    print("\nNessun allarme trovato nell'intero video.")
else:
    print(f"\nFatto. Salvati {saved_alarm} screenshot con allarme.")
