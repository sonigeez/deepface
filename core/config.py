def get_face(img_data, face_analyser):
    if img_data is None:
        return None
    analysed = face_analyser.get(img_data)
    try:
        return sorted(analysed, key=lambda x: x.bbox[0])[0]
    except IndexError:
        return None
