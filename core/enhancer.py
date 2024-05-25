import threading
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import import_hook
from gfpgan import GFPGANer

FACE_ENHANCER = None
THREAD_LOCK = threading.Lock()


def get_face_enhancer():
    global FACE_ENHANCER
    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = "GFPGANv1.4.pth"
            FACE_ENHANCER = GFPGANer(
                model_path=model_path, upscale=1, arch="clean", channel_multiplier=2
            )
    return FACE_ENHANCER


def enhance_face(image):
    enhancer = get_face_enhancer()
    _, _, enhanced_image = enhancer.enhance(
        image, has_aligned=False, only_center_face=True, paste_back=True
    )
    return enhanced_image
