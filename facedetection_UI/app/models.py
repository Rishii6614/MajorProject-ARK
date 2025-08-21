import cv2
import os
import numpy as np

DATASET_PATH = 'dataset/labeled_faces'
STATIC_OUTPUT_DIR = 'app/static/results'
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def train_lbph_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    label_map = {}
    current_id = 0

    for person in os.listdir(DATASET_PATH):
        person_folder = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(person_folder):
            continue
        label_map[current_id] = person
        for image_name in os.listdir(person_folder):
            if image_name.startswith('.'):
                continue
            image_path = os.path.join(person_folder, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(current_id)
        current_id += 1

    recognizer.train(faces, np.array(labels))
    return recognizer, label_map

def detect_and_recognize(image_path):
    recognizer, label_map = train_lbph_model()
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    results = []

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(roi)

        if confidence <= 60 and label in label_map:
            name = label_map[label]
        else:
            name = "Unknown"

        results.append(name)

        # Draw label and rectangle
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(image, name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Save annotated image
    os.makedirs(STATIC_OUTPUT_DIR, exist_ok=True)
    output_filename = os.path.basename(image_path)
    output_path = os.path.join(STATIC_OUTPUT_DIR, output_filename)
    cv2.imwrite(output_path, image)

    return output_filename, results
