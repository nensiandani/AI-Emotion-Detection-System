from deepface import DeepFace

def detect_emotion(face_img):

    try:
        result = DeepFace.analyze(
            img_path=face_img,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend="retinaface"
        )

        emotions = result[0]["emotion"]
        emotion = max(emotions, key=emotions.get)

        return emotion, emotions

    except:
        return None, None