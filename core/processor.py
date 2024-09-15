import os
import cv2
import numpy as np
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

def extract_eye_regions(face, frame):
    """
    Extracts the left and right eye regions based on facial landmarks.
    
    :param face: Face object containing landmarks.
    :param frame: The image/frame where the face is detected.
    :return: Dictionary containing left and right eye regions and their corresponding masks.
    """
    # Assume face has 'kps' attribute with 5 or 68 landmarks
    landmarks = face.kps  # shape: (5, 2) or (68, 2) depending on the model
    
    if landmarks.shape[0] == 5:
        left_eye_center = landmarks[0]
        right_eye_center = landmarks[1]
        eye_radius = 15  # Adjust based on face size
        
        # Ensure coordinates are integers
        left_eye_center = (int(left_eye_center[0]), int(left_eye_center[1]))
        right_eye_center = (int(right_eye_center[0]), int(right_eye_center[1]))
        
        # Extract eye regions with boundary checks
        left_x1 = max(left_eye_center[0] - eye_radius, 0)
        left_y1 = max(left_eye_center[1] - eye_radius, 0)
        left_x2 = min(left_eye_center[0] + eye_radius, frame.shape[1])
        left_y2 = min(left_eye_center[1] + eye_radius, frame.shape[0])
        left_eye_region = frame[left_y1:left_y2, left_x1:left_x2]
        
        right_x1 = max(right_eye_center[0] - eye_radius, 0)
        right_y1 = max(right_eye_center[1] - eye_radius, 0)
        right_x2 = min(right_eye_center[0] + eye_radius, frame.shape[1])
        right_y2 = min(right_eye_center[1] + eye_radius, frame.shape[0])
        right_eye_region = frame[right_y1:right_y2, right_x1:right_x2]
        
        # Create mask
        mask = np.zeros_like(frame[:, :, 0])
        cv2.circle(mask, left_eye_center, eye_radius, 255, -1)
        cv2.circle(mask, right_eye_center, eye_radius, 255, -1)
        
        return {
            "left_eye": left_eye_region,
            "right_eye": right_eye_region,
            "mask": mask
        }
    elif landmarks.shape[0] >= 68:
        # For 68 landmarks
        left_eye_points = landmarks[36:42]
        right_eye_points = landmarks[42:48]
        
        # Compute bounding rectangles
        left_x, left_y, left_w, left_h = cv2.boundingRect(left_eye_points.astype(np.int32))
        right_x, right_y, right_w, right_h = cv2.boundingRect(right_eye_points.astype(np.int32))
        
        left_eye_region = frame[left_y:left_y+left_h, left_x:left_x+left_w]
        right_eye_region = frame[right_y:right_y+right_h, right_x:right_x+right_w]
        
        # Create mask
        mask = np.zeros_like(frame[:, :, 0])
        cv2.polylines(mask, [left_eye_points.astype(np.int32)], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_points.astype(np.int32)], 255)
        cv2.polylines(mask, [right_eye_points.astype(np.int32)], True, 255, 2)
        cv2.fillPoly(mask, [right_eye_points.astype(np.int32)], 255)
        
        return {
            "left_eye": left_eye_region,
            "right_eye": right_eye_region,
            "mask": mask
        }
    else:
        raise ValueError("Unsupported number of landmarks")

def warp_eye(source_eye, target_eye):
    """
    Warps the source eye to match the target eye size.
    
    :param source_eye: Source eye image.
    :param target_eye: Target eye image.
    :return: Warped eye image.
    """
    source_height, source_width = source_eye.shape[:2]
    target_height, target_width = target_eye.shape[:2]
    warped_eye = cv2.resize(source_eye, (target_width, target_height))
    return warped_eye

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def blend_eye(target_frame, warped_eye, eye_position, mask):
    """
    Blends the warped eye into the target frame at the specified position using the mask.
    
    :param target_frame: The original target image/frame.
    :param warped_eye: The warped eye image to be blended.
    :param eye_position: Tuple (x, y) indicating where to place the eye.
    :param mask: Mask defining the eye region.
    :return: Frame with the eye blended.
    """
    x, y = eye_position
    h, w = warped_eye.shape[:2]
    
    # Ensure coordinates are within image boundaries
    x = clamp(x, 0, target_frame.shape[1] - w)
    y = clamp(y, 0, target_frame.shape[0] - h)
    
    roi = target_frame[y:y+h, x:x+w]
    
    # Create mask for the warped eye
    eye_mask = mask[y:y+h, x:x+w]
    eye_mask = cv2.cvtColor(eye_mask, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels
    
    # Blend using mask
    blended = cv2.bitwise_and(roi, cv2.bitwise_not(eye_mask))
    blended = cv2.add(blended, warped_eye)
    
    target_frame[y:y+h, x:x+w] = blended
    return target_frame

def process_video(source_img, frame_paths, face_analyser, reference_img=None):
    source_frame = cv2.imread(source_img)
    source_faces = face_analyser.get(source_frame)
    if not source_faces:
        print("[ERROR] No face detected in source image.")
        return

    source_face = get_face(source_frame, face_analyser)
    if source_face is None:
        print("[ERROR] Unable to extract source face.")
        return

    # Extract source eyes
    source_eyes = extract_eye_regions(source_face, source_frame)
    if not source_eyes:
        print("[ERROR] Unable to extract eyes from source face.")
        return

    reference_face = (
        get_face(cv2.imread(reference_img), face_analyser) if reference_img else None
    )
    if reference_img and reference_face is None:
        print(
            "\n[WARNING] No face detected in reference image. Please try with another one.\n"
        )
        return

    for idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)
        try:
            # Print percentage
            print(
                f"{(idx + 1) / len(frame_paths) * 100:.2f}%", end=""
            )
            faces = face_analyser.get(frame)
            for face in faces:
                if reference_face:
                    if match_faces(face, reference_face):
                        # Extract target eyes
                        target_eyes = extract_eye_regions(face, frame)
                        if not target_eyes:
                            print("E", end="")
                            continue

                        # Warp source eyes to target
                        warped_left_eye = warp_eye(source_eyes["left_eye"], target_eyes["left_eye"])
                        warped_right_eye = warp_eye(source_eyes["right_eye"], target_eyes["right_eye"])

                        # Define positions (ensure integer values)
                        left_eye_pos = (
                            int(face.kps[0][0] - source_eyes["left_eye"].shape[1] // 2),
                            int(face.kps[0][1] - source_eyes["left_eye"].shape[0] // 2)
                        )
                        right_eye_pos = (
                            int(face.kps[1][0] - source_eyes["right_eye"].shape[1] // 2),
                            int(face.kps[1][1] - source_eyes["right_eye"].shape[0] // 2)
                        )

                        # Blend eyes into the frame
                        frame = blend_eye(frame, warped_left_eye, left_eye_pos, target_eyes["mask"])
                        frame = blend_eye(frame, warped_right_eye, right_eye_pos, target_eyes["mask"])

                        # Optionally enhance the frame
                        enhanced_result = enhance_face(frame)
                        cv2.imwrite(frame_path, enhanced_result)
                        print(".", end="")
                        break
                else:
                    # If no reference face, process all detected faces
                    target_eyes = extract_eye_regions(face, frame)
                    if not target_eyes:
                        print("E", end="")
                        continue

                    # Warp source eyes to target
                    warped_left_eye = warp_eye(source_eyes["left_eye"], target_eyes["left_eye"])
                    warped_right_eye = warp_eye(source_eyes["right_eye"], target_eyes["right_eye"])

                    # Define positions (ensure integer values)
                    left_eye_pos = (
                        int(face.kps[0][0] - source_eyes["left_eye"].shape[1] // 2),
                        int(face.kps[0][1] - source_eyes["left_eye"].shape[0] // 2)
                    )
                    right_eye_pos = (
                        int(face.kps[1][0] - source_eyes["right_eye"].shape[1] // 2),
                        int(face.kps[1][1] - source_eyes["right_eye"].shape[0] // 2)
                    )

                    # Blend eyes into the frame
                    frame = blend_eye(frame, warped_left_eye, left_eye_pos, target_eyes["mask"])
                    frame = blend_eye(frame, warped_right_eye, right_eye_pos, target_eyes["mask"])

                    # Optionally enhance the frame
                    enhanced_result = enhance_face(frame)
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

    # Extract source eyes
    source_eyes = extract_eye_regions(source_face, cv2.imread(source_img))
    if not source_eyes:
        print("[ERROR] Unable to extract eyes from source face.")
        return target_path

    result = frame.copy()

    for face in faces:
        if reference_face:
            if match_faces(face, reference_face):
                target_eyes = extract_eye_regions(face, result)
                if not target_eyes:
                    continue

                # Warp source eyes to target
                warped_left_eye = warp_eye(source_eyes["left_eye"], target_eyes["left_eye"])
                warped_right_eye = warp_eye(source_eyes["right_eye"], target_eyes["right_eye"])

                # Define positions (ensure integer values)
                left_eye_pos = (
                    int(face.kps[0][0] - source_eyes["left_eye"].shape[1] // 2),
                    int(face.kps[0][1] - source_eyes["left_eye"].shape[0] // 2)
                )
                right_eye_pos = (
                    int(face.kps[1][0] - source_eyes["right_eye"].shape[1] // 2),
                    int(face.kps[1][1] - source_eyes["right_eye"].shape[0] // 2)
                )

                # Blend eyes into the frame
                result = blend_eye(result, warped_left_eye, left_eye_pos, target_eyes["mask"])
                result = blend_eye(result, warped_right_eye, right_eye_pos, target_eyes["mask"])
        else:
            # If no reference face, process all detected faces
            target_eyes = extract_eye_regions(face, result)
            if not target_eyes:
                continue

            # Warp source eyes to target
            warped_left_eye = warp_eye(source_eyes["left_eye"], target_eyes["left_eye"])
            warped_right_eye = warp_eye(source_eyes["right_eye"], target_eyes["right_eye"])

            # Define positions (ensure integer values)
            left_eye_pos = (
                int(face.kps[0][0] - source_eyes["left_eye"].shape[1] // 2),
                int(face.kps[0][1] - source_eyes["left_eye"].shape[0] // 2)
            )
            right_eye_pos = (
                int(face.kps[1][0] - source_eyes["right_eye"].shape[1] // 2),
                int(face.kps[1][1] - source_eyes["right_eye"].shape[0] // 2)
            )

            # Blend eyes into the frame
            result = blend_eye(result, warped_left_eye, left_eye_pos, target_eyes["mask"])
            result = blend_eye(result, warped_right_eye, right_eye_pos, target_eyes["mask"])

    # Enhance the final result after all swaps
    enhanced_result = enhance_face(result)

    # Modify target path to include the 'swapped-' prefix
    target_path = (
        rreplace(target_path, "/", "/swapped-", 1)
        if "/" in target_path
        else "swapped-" + target_path
    )
    print(target_path)
    
    # Save the final image with all eye swaps
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
