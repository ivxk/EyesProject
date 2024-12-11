import os
import cv2
import time
import json
import numpy as np
from flask import Flask, request, render_template, url_for
from skimage.feature import hog
import joblib
from ultralytics import YOLO

app = Flask(__name__)

# Load metrics
with open('models/metrics.json', 'r') as f:
    metrics_data = json.load(f)

classification_acc = metrics_data.get('classification_acc', 0)
yolo_map = metrics_data.get('yolo_map', 0)
yolov8_seg_map = metrics_data.get('yolov8_seg_map', 0)

# Load Classification Models (ANN+HOG)
ann_hog = joblib.load('models/best_ann_hog_model.joblib')
hog_scaler = joblib.load('models/hog_scaler.joblib')
label_encoder = joblib.load('models/label_encoder.joblib')

# Load YOLO Detector (Cataract Detection) - YOLOv8
yolo_detector = YOLO('models/yolo_detector.pt')

# Load YOLOv8 Segmentation Model
segmentation_model = YOLO('runs/segment/train/weights/best.pt')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    task = request.form.get('task')
    file = request.files.get('image')

    if file:
        filename = file.filename
        filepath = os.path.join('static', filename)
        file.save(filepath)

        if task == 'classification':
            # Classification Inference
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (128,128)) / 255.0
            hog_features = hog(img_resized, orientations=9, pixels_per_cell=(16,16),
                                cells_per_block=(3,3), visualize=False, block_norm='L2-Hys')
            hog_features_scaled = hog_scaler.transform([hog_features])
            pred_probs = ann_hog.predict(hog_features_scaled)
            pred_class_idx = np.argmax(pred_probs)
            pred_class = label_encoder.inverse_transform([pred_class_idx])[0]

            return render_template('result.html',
                                   image_path=url_for('static', filename=filename),
                                   result=pred_class,
                                   classification_acc=classification_acc)

        else:
            # Detection + Segmentation Inference
            start_time = time.time()
            # Detection
            detection_results = yolo_detector.predict(source=filepath, save=False, conf=0.25)
            det_img = cv2.imread(filepath)
            boxes, classes_list, confidences_list = [], [], []
            if len(detection_results) > 0 and detection_results[0].boxes is not None:
                boxes_np = detection_results[0].boxes.xyxy.cpu().numpy()
                class_ids = detection_results[0].boxes.cls.cpu().numpy()
                confs = detection_results[0].boxes.conf.cpu().numpy()

                for i, box in enumerate(boxes_np):
                    x1, y1, x2, y2 = map(int, box)
                    boxes.append((x1, y1, x2, y2))
                    classes_list.append(int(class_ids[i]))
                    confidences_list.append(float(confs[i]))
                    cv2.rectangle(det_img, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(det_img, f'{confs[i]:.2f}', (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            det_filename = 'det_img.jpg'
            cv2.imwrite(os.path.join('static', det_filename), det_img)

            # Segmentation
            seg_results = segmentation_model.predict(source=filepath, save=False, conf=0.25)
            seg_img = seg_results[0].plot()
            seg_filename = 'seg_img.jpg'
            cv2.imwrite(os.path.join('static', seg_filename), seg_img)

            end_time = time.time()
            inference_time_ms = int((end_time - start_time)*1000)

            return render_template('result_det_seg.html',
                                   det_path=url_for('static', filename=det_filename),
                                   seg_path=url_for('static', filename=seg_filename),
                                   detections=zip(boxes, classes_list, confidences_list),
                                   yolo_map=yolo_map,
                                   yolov8_seg_map=yolov8_seg_map,
                                   inference_time=inference_time_ms)
    else:
        return "No file uploaded!"

if __name__ == '__main__':
    app.run(debug=True)
