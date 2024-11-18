from flask import Flask, request, jsonify
from PIL import Image
import cv2
import mediapipe as mp
import numpy as np
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)

# Initialize Mediapipe Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Function to calculate the average skin tone
def get_skin_tone(image):
    avg_color_per_row = np.average(image, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color

# Function to calculate eye shape and size
def get_eye_shape_and_size(landmarks, image):
    # Extract eye landmarks properly
    left_eye_inner = (landmarks.landmark[33].x, landmarks.landmark[133].y)  # Approx inner left eye landmark
    left_eye_outer = (landmarks.landmark[133].x, landmarks.landmark[362].y)  # Approx outer left eye landmark
    right_eye_inner = (landmarks.landmark[362].x, landmarks.landmark[263].y)  # Approx inner right eye landmark
    right_eye_outer = (landmarks.landmark[263].x, landmarks.landmark[33].y)  # Approx outer right eye landmark

    # Eye size: Calculate the area of the bounding box around the eyes
    left_eye_size = abs(left_eye_outer[0] - left_eye_inner[0]) * abs(left_eye_outer[1] - left_eye_inner[1])
    right_eye_size = abs(right_eye_outer[0] - right_eye_inner[0]) * abs(right_eye_outer[1] - right_eye_inner[1])
    
    return {
        "left_eye_size": left_eye_size,
        "right_eye_size": right_eye_size,
        "eye_shape": {
            "left_eye_shape": left_eye_outer[0] - left_eye_inner[0],
            "right_eye_shape": right_eye_outer[0] - right_eye_inner[0]
        }
    }

# Function to calculate eyebrow color and thinness
def get_eyebrow_color_and_thinness(landmarks, image):
    # Use more precise eyebrow landmarks, including both upper and lower regions
    left_eyebrow = np.array([(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in range(70, 85)])
    right_eyebrow = np.array([(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in range(55, 70)])

    # Calculate the bounding box of the eyebrow region for color extraction (including upper and lower parts)
    left_bounding_box = cv2.boundingRect(np.int32(left_eyebrow * image.shape[1]))
    right_bounding_box = cv2.boundingRect(np.int32(right_eyebrow * image.shape[1]))

    # Crop the eyebrows (expanded bounding box to capture both upper and lower parts)
    left_eyebrow_region = image[left_bounding_box[1]:left_bounding_box[1] + left_bounding_box[3], 
                                left_bounding_box[0]:left_bounding_box[0] + left_bounding_box[2]]
    right_eyebrow_region = image[right_bounding_box[1]:right_bounding_box[1] + right_bounding_box[3], 
                                 right_bounding_box[0]:right_bounding_box[0] + right_bounding_box[2]]

    # Calculate the average color of the eyebrows (RGB values)
    left_eyebrow_color = np.mean(left_eyebrow_region, axis=(0, 1))
    right_eyebrow_color = np.mean(right_eyebrow_region, axis=(0, 1))

    # Calculate eyebrow thinness: ratio of width to height of the bounding box
    left_eyebrow_thinness = left_bounding_box[2] / left_bounding_box[3]
    right_eyebrow_thinness = right_bounding_box[2] / right_bounding_box[3]

    return {
        "left_eyebrow_color": left_eyebrow_color.tolist(),
        "right_eyebrow_color": right_eyebrow_color.tolist(),
        "left_eyebrow_thinness": left_eyebrow_thinness,
        "right_eyebrow_thinness": right_eyebrow_thinness
    }

# Function to calculate lip color (Improved)
def get_lip_color(landmarks, image):
    # Use more precise landmarks for the lips
    lip_landmarks = np.array([(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in range(61, 81)])

    # Calculate the bounding box of the lip region
    lip_bounding_box = cv2.boundingRect(np.int32(lip_landmarks * image.shape[1]))

    # Expand the bounding box slightly to ensure full lip area is captured
    margin = 10  # Increase the margin to capture more of the lip area
    lip_bounding_box_expanded = [
        max(0, lip_bounding_box[0] - margin),
        max(0, lip_bounding_box[1] - margin),
        min(image.shape[1], lip_bounding_box[0] + lip_bounding_box[2] + margin),
        min(image.shape[0], lip_bounding_box[1] + lip_bounding_box[3] + margin)
    ]

    # Crop the lips region (with expanded bounding box)
    lip_region = image[lip_bounding_box_expanded[1]:lip_bounding_box_expanded[3], 
                       lip_bounding_box_expanded[0]:lip_bounding_box_expanded[2]]

    # Calculate the average color of the lips
    lip_color = np.mean(lip_region, axis=(0, 1))
    return lip_color.tolist()

# Function to calculate face shape
def get_face_shape(landmarks):
    # We will use a simple approach: Based on the relative positions of key landmarks to classify face shape
    jawline_width = np.linalg.norm(np.array([landmarks.landmark[0].x, landmarks.landmark[0].y]) - 
                                  np.array([landmarks.landmark[16].x, landmarks.landmark[16].y]))
    cheekbone_width = np.linalg.norm(np.array([landmarks.landmark[234].x, landmarks.landmark[234].y]) - 
                                     np.array([landmarks.landmark[454].x, landmarks.landmark[454].y]))
    face_length = np.linalg.norm(np.array([landmarks.landmark[10].x, landmarks.landmark[10].y]) - 
                                 np.array([landmarks.landmark[152].x, landmarks.landmark[152].y]))

    # Classify face shape based on proportions
    if cheekbone_width > jawline_width and face_length > cheekbone_width:
        return "Oval"
    elif jawline_width > cheekbone_width and face_length > jawline_width:
        return "Square"
    elif cheekbone_width > jawline_width and face_length < cheekbone_width:
        return "Round"
    else:
        return "Heart-shaped"

# Function to extract features from the face mesh
def analyze_image(image_path):
    try:
        # Read the image
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Mediapipe Face Detection
        face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        results = face_detection.process(rgb_image)

        if not results.detections:
            return {"error": "No face detected"}

        # For simplicity, process only the first detected face
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        h, w, _ = rgb_image.shape
        x1, y1, x2, y2 = int(bboxC.xmin * w), int(bboxC.ymin * h), int((bboxC.xmin + bboxC.width) * w), int((bboxC.ymin + bboxC.height) * h)

        # Crop the face for further analysis
        face_crop = rgb_image[y1:y2, x1:x2]

        # Mediapipe Face Mesh for landmarks
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
        face_mesh_results = face_mesh.process(rgb_image)

        if not face_mesh_results.multi_face_landmarks:
            return {"error": "No landmarks detected"}

        landmarks = face_mesh_results.multi_face_landmarks[0]

        # Analyze features based on regions
        left_eye = face_crop[int(0.3 * face_crop.shape[0]):int(0.4 * face_crop.shape[0]), 
                             int(0.3 * face_crop.shape[1]):int(0.4 * face_crop.shape[1])]
        right_eye = face_crop[int(0.3 * face_crop.shape[0]):int(0.4 * face_crop.shape[0]), 
                              int(0.6 * face_crop.shape[1]):int(0.7 * face_crop.shape[1])]

        hair_region = face_crop[:int(0.2 * face_crop.shape[0]), :]
        cheek_region = face_crop[int(0.6 * face_crop.shape[0]):int(0.8 * face_crop.shape[0]), 
                                 int(0.4 * face_crop.shape[1]):int(0.6 * face_crop.shape[1])]

        # Calculate colors
        left_eye_color = np.mean(left_eye, axis=(0, 1))
        right_eye_color = np.mean(right_eye, axis=(0, 1))
        hair_color = np.mean(hair_region, axis=(0, 1))
        skin_tone = get_skin_tone(cheek_region)

        # Extract other features
        eye_shape_and_size = get_eye_shape_and_size(landmarks, image)
        eyebrow_color_and_thinness = get_eyebrow_color_and_thinness(landmarks, image)
        lip_color = get_lip_color(landmarks, image)
        face_shape = get_face_shape(landmarks)

        return {
            "eye_color": {
                "left": left_eye_color.tolist(),
                "right": right_eye_color.tolist()
            },
            "hair_color": hair_color.tolist(),
            "skin_tone": skin_tone.tolist(),
            "eye_shape_and_size": eye_shape_and_size,
            "eyebrow_color_and_thinness": eyebrow_color_and_thinness,
            "lip_color": lip_color,
            "face_shape": face_shape
        }

    except Exception as e:
        return {"error": str(e)}

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image file found"}), 400

    image_file = request.files['image']
    image_path = "uploaded_image.jpg"
    image_file.save(image_path)

    # Analyze the image
    analysis_result = analyze_image(image_path)
    
    return jsonify(analysis_result)

if __name__ == '__main__':
    app.run(debug=True)
