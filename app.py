from flask import Flask, render_template, request, jsonify, Response
import cv2
import dlib
import numpy as np
import os
import json
from datetime import datetime
from ultralytics import YOLO

app = Flask(__name__)
app.config['DATASET_PATH'] = 'dataset'
os.makedirs(app.config['DATASET_PATH'], exist_ok=True)

# Initialize models
MODEL_DIR = "model"
yolo = YOLO(os.path.join(MODEL_DIR, "yolov8n-face.pt"))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat"))

class FaceRegistrar:
    def __init__(self):
        self.REQUIRED_SAMPLES = 7
        self.EYE_AR_THRESH = 0.25

    def _eye_aspect_ratio(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)

    def register_user(self, user_id, user_name):
        user_dir = os.path.join(app.config['DATASET_PATH'], user_id)
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
                results = yolo(frame)
                person_boxes = [box for box in results[0].boxes if int(box.cls) == 0]
                if len(person_boxes) != 1:
                    continue

                # Face processing
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = detector(rgb)
                if len(faces) == 1:
                    shape = predictor(rgb, faces[0])
                    landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
                    
                    # Liveness check
                    left_eye = landmarks[42:48]
                    right_eye = landmarks[36:42]
                    ear = (self._eye_aspect_ratio(left_eye) + self._eye_aspect_ratio(right_eye)) / 2.0
                    
                    if ear >= self.EYE_AR_THRESH:
                        x1, y1, x2, y2 = faces[0].left(), faces[0].top(), faces[0].right(), faces[0].bottom()
                        samples.append({
                            'face_img': frame[y1:y2, x1:x2],
                            'landmarks': landmarks,
                            'bbox': [x1, y1, x2, y2],
                            'timestamp': datetime.now().isoformat()
                        })
                        cv2.waitKey(500)  # Small delay between captures

            # Save data
            if len(samples) == self.REQUIRED_SAMPLES:
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
            return {'success': False, 'message': 'Insufficient valid samples'}
                
        finally:
            cap.release()

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Person detection
        results = yolo(frame)
        person_count = sum(int(box.cls) == 0 for box in results[0].boxes)
        
        # Face detection and landmarks
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector(rgb)
        
        # Draw annotations
        if person_count == 1 and len(faces) == 1:
            shape = predictor(rgb, faces[0])
            landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
            
            # Draw YOLO box
            for box in results[0].boxes:
                if int(box.cls) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw landmarks and contours
            for (x, y) in landmarks:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
            
            contours = [
                landmarks[0:17],   # Jaw
                landmarks[17:22],  # Left eyebrow
                landmarks[22:27],  # Right eyebrow
                landmarks[27:31],  # Nose bridge
                landmarks[31:36],  # Lower nose
                landmarks[36:42],  # Left eye
                landmarks[42:48],  # Right eye
                landmarks[48:60],  # Outer lips
                landmarks[60:68]   # Inner lips
            ]
            
            for contour in contours:
                cv2.polylines(frame, [contour.astype(int)], False, (255, 0, 0), 1)
            
            cv2.putText(frame, "Ready to register", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "ERROR: Only one person allowed", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    user_id = request.form.get('user_id')
    user_name = request.form.get('user_name')
    
    if not user_id or not user_name:
        return jsonify({'error': 'Both ID and Name are required'}), 400
    
    try:
        registrar = FaceRegistrar()
        result = registrar.register_user(user_id, user_name)
        if result['success']:
            return jsonify(result), 200
        return jsonify({'error': result['message']}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)