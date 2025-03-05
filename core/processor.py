import os
import cv2
import insightface
import core.globals
from core.config import get_face
from core.utils import rreplace
from core.enhancer import enhance_face
from scipy.spatial.distance import cosine
from typing import List, Tuple

face_swapper = None


def get_face_swapper():
    global face_swapper
    if face_swapper is None:
        face_swapper = insightface.model_zoo.get_model(
            "inswapper_128_fp16.onnx", providers=core.globals.providers
        )
    return face_swapper


def process_video(source_imgs: List[str], frame_paths: List[str], face_analyser, reference_imgs: List[str] = None):
    """
    Process video with multiple source-reference pairs
    
    Args:
        source_imgs: List of source image paths
        frame_paths: List of frame paths to process
        face_analyser: Face analyzer instance
        reference_imgs: List of reference image paths corresponding to source_imgs
    """
    # Prepare source and reference faces
    source_faces = []
    reference_faces = []
    
    for i, source_img in enumerate(source_imgs):
        source_face = get_face(cv2.imread(source_img), face_analyser)
        if source_face is None:
            print(f"\n[WARNING] No face detected in source image {i+1}. Skipping this pair.\n")
            continue
            
        reference_face = None
        if reference_imgs and i < len(reference_imgs):
            reference_face = get_face(cv2.imread(reference_imgs[i]), face_analyser)
            if reference_face is None:
                print(f"\n[WARNING] No face detected in reference image {i+1}. Skipping this pair.\n")
                continue
                
        source_faces.append(source_face)
        reference_faces.append(reference_face)

    if not source_faces:
        print("\n[WARNING] No valid source faces found.\n")
        return

    # Process each frame
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        try:
            print(f"{frame_paths.index(frame_path) / len(frame_paths) * 100:.2f}%", end="")
            faces = face_analyser.get(frame)
            result = frame.copy()
            
            for face in faces:
                # Try to match face with each reference-source pair
                for source_face, reference_face in zip(source_faces, reference_faces):
                    if reference_face:
                        if match_faces(face, reference_face):
                            result = face_swapper.get(result, face, source_face, paste_back=True)
                            print(".", end="")
                            break
                    else:
                        # If no reference, swap with first source face
                        result = face_swapper.get(result, face, source_faces[0], paste_back=True)
                        print(".", end="")
                        break
            
            # Enhance the final result after all swaps
            enhanced_result = enhance_face(result)
            cv2.imwrite(frame_path, enhanced_result)
        except Exception as e:
            print("E", end="")
            pass


def process_img(source_imgs: List[str], target_path: str, face_analyser, reference_imgs: List[str] = None):
    """
    Process image with multiple source-reference pairs
    
    Args:
        source_imgs: List of source image paths
        target_path: Path to target image
        face_analyser: Face analyzer instance
        reference_imgs: List of reference image paths corresponding to source_imgs
    """
    frame = cv2.imread(target_path)
    faces = face_analyser.get(frame)
    result = frame.copy()

    # Prepare source and reference faces
    source_faces = []
    reference_faces = []
    
    for i, source_img in enumerate(source_imgs):
        source_face = get_face(cv2.imread(source_img), face_analyser)
        if source_face is None:
            print(f"\n[WARNING] No face detected in source image {i+1}. Skipping this pair.\n")
            continue
            
        reference_face = None
        if reference_imgs and i < len(reference_imgs):
            reference_face = get_face(cv2.imread(reference_imgs[i]), face_analyser)
            if reference_face is None:
                print(f"\n[WARNING] No face detected in reference image {i+1}. Skipping this pair.\n")
                continue
                
        source_faces.append(source_face)
        reference_faces.append(reference_face)

    if not source_faces:
        print("\n[WARNING] No valid source faces found.\n")
        return target_path

    # Process each face in the target image
    for face in faces:
        # Try to match face with each reference-source pair
        for source_face, reference_face in zip(source_faces, reference_faces):
            if reference_face:
                if match_faces(face, reference_face):
                    result = face_swapper.get(result, face, source_face, paste_back=True)
                    break
            else:
                # If no reference, swap with first source face
                result = face_swapper.get(result, face, source_faces[0], paste_back=True)
                break

    # Enhance the final result after all swaps
    enhanced_result = enhance_face(result)

    # Modify target path to include the 'swapped-' prefix
    target_path = (
        rreplace(target_path, "/", "/swapped-", 1)
        if "/" in target_path
        else "swapped-" + target_path
    )
    print(target_path)
    
    # Save the final image with all face swaps
    cv2.imwrite(target_path, enhanced_result)
    
    return target_path


def match_faces(face1, face2, threshold=0.8):
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
