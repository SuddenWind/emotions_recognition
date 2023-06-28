import ultralytics
from preprocess import Preprocess
import cv2
from ultralytics import YOLO
import time



class Emotion_recognition:
    def __init__(self, GPU = True):
        self.cam = cv2.VideoCapture(0)
        if not self.cam.isOpened():
            print("Не удалось открыть камеру")
            raise Exception("Не удалось открыть камеру")
        else:
            print("Камера запущена")
            
        self.model = YOLO('yolo_m3.pt')

        
if __name__ == "__main__":
    recognition = Emotion_recognition()
    preprocessor = Preprocess(GPU=False)

    while True:
        ret, image = recognition.cam.read()
        cv2.imshow("facial emotion recognition", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    
        start = time.time()
        face_aligned = preprocessor.run(image)
        print('preprocessor ', time.time() - start)
        
        if face_aligned is not None:
            result = recognition.model.predict(face_aligned)
            emotion = result[0].probs.top1
            emotion = result[0].names[emotion]
            print(emotion)
            cv2.putText(face_aligned, emotion, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # display the output images
            cv2.imshow('Result', face_aligned)