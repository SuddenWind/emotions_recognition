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
        
    def predict(self, img):
        return self.model.predict(img)
    
def main():
    while True:
        ret, image = recognition.cam.read()
        cv2.imshow("facial emotion recognition", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    
        face_aligned = preprocessor.run(image)
        
        if face_aligned is not None:
            result = recognition.predict(face_aligned)
            emotion = result[0].probs.top1
            emotion = result[0].names[emotion]
            print(emotion)
            cv2.putText(face_aligned, emotion, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # display the output images
            cv2.imshow('Result', face_aligned)    
        
        
def main_wcounter():
    start_time = time.time()
    x = 1             # displays the frame rate every 1 second
    counter = 0
    fps = []

    while True:
        ret, image = recognition.cam.read()
        cv2.imshow("facial emotion recognition", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            print(fps)
            break
    
        start = time.time()
        face_aligned = preprocessor.run(image)
        print('preprocessor ', time.time() - start)
        
        if face_aligned is not None:
            result = recognition.predict(face_aligned)
            emotion = result[0].probs.top1
            emotion = result[0].names[emotion]
            print(emotion)
            cv2.putText(face_aligned, emotion, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # display the output images
            cv2.imshow('Result', face_aligned)
            
        counter += 1
        if (time.time() - start_time) > x :
            fps.append(counter / (time.time() - start_time))
            print("FPS: ", counter / (time.time() - start_time))
            counter = 0
            start_time = time.time()

        
if __name__ == "__main__":
    recognition = Emotion_recognition()
    preprocessor = Preprocess(GPU=False)
    main_wcounter()
    
    