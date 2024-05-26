import os
import cv2
import insightface
import core.globals
from core.config import get_face
from core.utils import rreplace
from core.enhancer import enhance_face
from scipy.spatial.distance import cosine

face_swapper = None


def get_face_swapper():
    global face_swapper
    if face_swapper is None:
        face_swapper = insightface.model_zoo.get_model(
            "inswapper_128_fp16.onnx", providers=core.globals.providers
        )
    return face_swapper


def process_video(source_img, frame_paths, face_analyser, reference_img=None):
    source_face = get_face(cv2.imread(source_img), face_analyser)
    reference_face = (
        get_face(cv2.imread(reference_img), face_analyser) if reference_img else None
    )
    if reference_img and reference_face is None:
        print(
            "\n[WARNING] No face detected in reference image. Please try with another one.\n"
        )
        return

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        try:
            # print percentage
            print(
                f"{frame_paths.index(frame_path) / len(frame_paths) * 100:.2f}%", end=""
            )
            faces = face_analyser.get(frame)
            for face in faces:
                if reference_face:
                    if match_faces(face, reference_face):
                        result = face_swapper.get(
                            frame, face, source_face, paste_back=True
                        )
                        enhanced_result = enhance_face(result)
                        cv2.imwrite(frame_path, enhanced_result)
                        print(".", end="")
                        break
                else:
                    result = face_swapper.get(frame, face, source_face, paste_back=True)
                    enhanced_result = enhance_face(result)
                    cv2.imwrite(frame_path, enhanced_result)
                    print(".", end="")
                    break
            else:
                print("S", end="")
        except Exception as e:
            print("E", end="")
            pass


def process_img(source_img, target_path, face_analyser, reference_img=None):
    frame = cv2.imread(target_path)
    faces = face_analyser.get(frame)
    source_face = get_face(cv2.imread(source_img), face_analyser)
    reference_face = (
        get_face(cv2.imread(reference_img), face_analyser) if reference_img else None
    )
    if reference_img and reference_face is None:
        print(
            "\n[WARNING] No face detected in reference image. Please try with another one.\n"
        )
        return target_path

    for face in faces:
        if reference_face:
            if match_faces(face, reference_face):
                result = face_swapper.get(frame, face, source_face, paste_back=True)
                break
        else:
            result = face_swapper.get(frame, face, source_face, paste_back=True)
            break
    enhanced_result = enhance_face(result)
    target_path = (
        rreplace(target_path, "/", "/swapped-", 1)
        if "/" in target_path
        else "swapped-" + target_path
    )
    print(target_path)
    cv2.imwrite(target_path, enhanced_result)
    return target_path


def match_faces(face1, face2, threshold=1):
    """
    Compare two faces based on their embeddings.

    :param face1: First face object with an embedding attribute.
    :param face2: Second face object with an embedding attribute.
    :param threshold: Distance threshold to consider the faces as matching.
    :return: True if faces match, False otherwise.
    """
    embedding1 = face1.embedding
    embedding2 = face2.embedding

    distance = cosine(embedding1, embedding2)

    return distance < threshold
