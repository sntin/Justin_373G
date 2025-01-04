from ultralytics import YOLO
import cv2


class VehicleDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame_queue, results_queue):
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                results = self.model(frame, conf=0.4, iou=0.8, verbose=False)[0]
                inf_msg = self.get_inference_data(results)
                results_queue.put(inf_msg)

    def get_inference_data(self, results):
        frame = results.orig_img
        annotated_frame = frame.copy()
        boxes = results.boxes
        for box in boxes:
            x1, x2, y1, y2 = box.xyxy[0].numpy()
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            c = int(box.cls)
            cv2.rectangle(
                annotated_frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2
            )
            text = f"{self.model.names[c]} | ID: {c} | conf: {box.conf}"
            text_size, _ = cv2.getTextSize(
                text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1
            )
            text_width, text_height = text_size
            cv2.rectangle(
                annotated_frame,
                (x2 + text_width, y1 - text_height - 5),
                (x2, y1),
                color=(0, 0, 255),
                thickness=-1,
            )
            cv2.putText(
                annotated_frame,
                text,
                (x2 + 2, y2 - 3),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=1,
            )

        confs = boxes.conf
        coords = boxes.xywhn
        inf_msg = (frame, annotated_frame, coords, boxes.cls, confs)
        return inf_msg
