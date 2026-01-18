import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from pathlib import Path
import random

base_options = python.BaseOptions(model_asset_path="detector.tflite")
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)


def load_known_faces(known_dir: str = "known") -> dict:
    """Return a dict mapping a person's name → face encoding (numpy array)."""
    known_dir_path = Path(known_dir)
    encoding_map = {}
    for known_file in known_dir_path.iterdir():
        if known_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        name = known_file.stem
        print("Known " + str(known_file))
        image = mp.Image.create_from_file(str(known_file))

        detection_result = detector.detect(image)

        if not detection_result:
            print(f"⚠️ No face found in {known_file.name}; skipping.")
            continue
        encoding_map[name] = (image, detection_result)
    return encoding_map


def load_image_from_webcam(output_path: str = "image.png") -> None:
    """Capture a single frame from the first webcam and save it."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")
    for i in range(30):
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
    cap.release()

    cv2.imwrite(output_path, frame)
    print(f"🖼️ Saved frame to {output_path}")


def encode_faces(image_path: str, known_encodings: dict) -> list:
    """Return a list of (name, bounding box) pairs for faces found in the image."""
    image = mp.Image.create_from_file(image_path)

    detection_result = detector.detect(image)

    if not detection_result:
        print("🔍 No faces detected.")
        return []
    face_locations = detection_result.detections
    results = []
    for face_location in face_locations:
        name = random.choice(list(known_encodings.keys()))
        meme_image, meme_face_detection = known_encodings[name]

        results.append((meme_image, meme_face_detection, image, face_location))
    return results


def draw_results(image_path: str, results: list, output_path: str = "annotated.jpg"):
    """Swap faces from the webcam image into the meme images."""
    for meme_image, meme_face_detection, webcam_image, face_detection in results:
        # Convert MediaPipe images to OpenCV format
        meme_cv = cv2.cvtColor(meme_image.numpy_view(), cv2.COLOR_RGB2BGR)
        webcam_cv = cv2.cvtColor(webcam_image.numpy_view(), cv2.COLOR_RGB2BGR)

        # Get bounding box from webcam face detection
        bb_webcam = face_detection.bounding_box
        webcam_left = bb_webcam.origin_x
        webcam_top = bb_webcam.origin_y
        webcam_right = bb_webcam.origin_x + bb_webcam.width
        webcam_bottom = bb_webcam.origin_y + bb_webcam.height

        # Get bounding box from meme face detection
        bb_meme = meme_face_detection.detections[0].bounding_box
        meme_left = bb_meme.origin_x
        meme_top = bb_meme.origin_y
        meme_right = bb_meme.origin_x + bb_meme.width
        meme_bottom = bb_meme.origin_y + bb_meme.height

        # Extract face from webcam image
        webcam_face = webcam_cv[webcam_top:webcam_bottom, webcam_left:webcam_right]

        # Resize webcam face to match meme face size
        meme_width = meme_right - meme_left
        meme_height = meme_bottom - meme_top
        resized_face = cv2.resize(webcam_face, (meme_width, meme_height))

        # Replace the face in the meme image
        meme_cv[meme_top:meme_bottom, meme_left:meme_right] = resized_face

        # Save the composite image
        cv2.imwrite(output_path, meme_cv)
        print(f"🖼️ Face-swapped image saved to {output_path}")


if __name__ == "__main__":
    known_encodings = load_known_faces("faces")
    if not known_encodings:
        raise RuntimeError("No known faces loaded – add images to the `known/` folder.")
    load_image_from_webcam()
    results = encode_faces("image.png", known_encodings)
    if results:
        draw_results("image.png", results, "output.jpg")
    else:
        print("❌ No recognizable face found.")