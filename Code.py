import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from pathlib import Path
import random
import requests


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
        # FIXME Just for the new one
        image = cv2.imread(str(known_file), cv2.IMREAD_COLOR)
        encoding_map[name] = (image, detection_result)
    return encoding_map


def load_image_from_pi(output_path: str = "image.png") -> None:
    """Capture a single frame from the first webcam and save it."""
    response = requests.get("http://127.0.0.1:5000/picture")
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    print(f"🖼️ Saved image from Pi to {output_path}")


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

        # FIXME Just for the new one
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
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

        import cv2


import numpy as np
import sys

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def create_face_landmarker():
    base_options = python.BaseOptions(model_asset_path="face_landmarker.task")

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    return vision.FaceLandmarker.create_from_options(options)


def get_landmarks(image, landmarker):
    """Return Nx2 array of face landmarks."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None

    h, w, _ = image.shape
    landmarks = []
    for lm in result.face_landmarks[0]:
        landmarks.append((int(lm.x * w), int(lm.y * h)))

    return np.array(landmarks)


def create_face_mask(image, landmarks):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(landmarks)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def warp_face(src_img, src_landmarks, dst_landmarks, dst_shape):
    src_pts = np.float32(src_landmarks)
    dst_pts = np.float32(dst_landmarks)

    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if M is None:
        raise RuntimeError("Could not estimate affine transform")

    warped = cv2.warpAffine(
        src_img,
        M,
        (dst_shape[1], dst_shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return warped


def draw_results2(image_path: str, results: list, output_path: str = "annotated.jpg"):
    landmarker = create_face_landmarker()
    for meme_image, meme_face_detection, webcam_image, face_detection in results:
        print("src_img:", type(webcam_image), webcam_image is None)
        print("dst_img:", type(meme_image), meme_image is None)

        src_landmarks = get_landmarks(webcam_image, landmarker)
        dst_landmarks = get_landmarks(meme_image, landmarker)

        dst_face_mask = create_face_mask(meme_image, dst_landmarks)
        warped_src_face = warp_face(
            webcam_image, src_landmarks, dst_landmarks, meme_image.shape
        )
        center = np.mean(dst_landmarks, axis=0).astype(int)
        center = (int(center[0]), int(center[1]))

        output = cv2.seamlessClone(
            warped_src_face, meme_image, dst_face_mask, center, cv2.NORMAL_CLONE
        )

        cv2.imwrite(output_path, output)
        print(f"🖼️ Face-swapped image saved to {output_path}")


if __name__ == "__main__":
    known_encodings = load_known_faces("faces")
    if not known_encodings:
        raise RuntimeError("No known faces loaded – add images to the `known/` folder.")
    # load_image_from_webcam()
    load_image_from_pi()
    results = encode_faces("image.png", known_encodings)
    if results:
        draw_results2("image.png", results, "output.jpg")
    else:
        print("❌ No recognizable face found.")
