import cv2
import dlib
import numpy as np
import os
import json
from datetime import datetime
from ultralytics import YOLO

class FaceRegistrar:
    def __init__(self):
        self.MODEL_DIR = "model"
        self.DATASET_PATH = "dataset"
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.DATASET_PATH, exist_ok=True)

        # Load models
        self.yolo = YOLO(os.path.join(self.MODEL_DIR, "yolov8n-face.pt"))
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            os.path.join(self.MODEL_DIR, "shape_predictor_68_face_landmarks.dat")
        )

        # Configuration
        self.REQUIRED_SAMPLES = 7
        self.CAPTURE_DELAY = 1.5

    def register_user(self, user_id, user_name):
        user_dir = os.path.join(self.DATASET_PATH, user_id)
        if os.path.exists(user_dir):
            return {'success': False, 'message': f"User ID {user_id} already exists"}

        cap = cv2.VideoCapture(0)
        samples = []
        
        try:
            while len(samples) < self.REQUIRED_SAMPLES:
                ret, frame = cap.read()
                if not ret:
                    continue

                # Person validation
                results = self.yolo(frame)
                person_boxes = [box for box in results[0].boxes if int(box.cls) == 0]
                if len(person_boxes) != 1:
                    continue

                # Face processing
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.detector(rgb)
                if len(faces) == 1:
                    shape = self.predictor(rgb, faces[0])
                    landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
                    
                    x1, y1, x2, y2 = faces[0].left(), faces[0].top(), faces[0].right(), faces[0].bottom()
                    samples.append({
                        'face_img': frame[y1:y2, x1:x2],
                        'landmarks': landmarks,
                        'bbox': [x1, y1, x2, y2],
                        'timestamp': datetime.now().isoformat()
                    })
                    cv2.waitKey(int(self.CAPTURE_DELAY * 1000))

            # Save data
            os.makedirs(user_dir)
            metadata = {
                'user_id': user_id,
                'user_name': user_name,
                'registration_date': datetime.now().isoformat(),
                'samples': len(samples)
            }
            
            for i, sample in enumerate(samples):
                cv2.imwrite(os.path.join(user_dir, f"face_{i}.jpg"), sample['face_img'])
                np.savez(
                    os.path.join(user_dir, f"sample_{i}.npz"),
                    landmarks=sample['landmarks'],
                    bbox=sample['bbox']
                )
            
            with open(os.path.join(user_dir, 'meta.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return {'success': True, 
                   'message': f"Registered {user_name} (ID: {user_id})",
                   'samples': len(samples)}
        
        finally:
            cap.release()