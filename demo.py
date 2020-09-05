import cv2
import time
import argparse
import numpy as np
from keras.models import load_model

model = load_model('weights/saved_model.h5')    # load trained model
face_classifier = cv2.CascadeClassifier('weights/haarcascade_frontalface_default.xml')    # load haar cascade
expressions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']    # facial expression labels

def predict(img):
    '''Detects face, predicts facial expression
    and returns modified image.'''
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_coord = face_classifier.detectMultiScale(gray_img, 1.3, 5) 
    
    height = img.shape[0]
    if len(face_coord) > 0:
        for i in range(len(face_coord)):
            x,y,w,h = face_coord[i]
            cv2.rectangle(img, (x,y), (x+w,y+h), (225,0,0), 2)

            face_img = gray_img[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (48, 48))
            face_img = face_img.reshape(1,48,48,1)
            face_img = face_img/255

            pred = expressions[np.argmax(model.predict(face_img))]
            cv2.putText(img, pred, (x,y-int(height/60)), cv2.FONT_HERSHEY_SIMPLEX, 2, (53, 67, 255), 2, 2)     
    else:
        print('No face found!')
    
    return img


def image(inp_path):    # for image input

    out_path = 'data/outputs/output.jpg'
    img = cv2.imread(inp_path)
    img = predict(img)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    print('Output image saved to', out_path)
    cv2.imwrite(out_path, img)


def video(inp_path):    # for video input
    
    out_path = 'data/outputs/output.avi'
    vid = cv2.VideoCapture(inp_path)
    width, height = int(vid.get(3)), int(vid.get(4))  
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MJPG'), 30, (width, height)) 

    frame_count = 0
    time_sum = 0
    while(vid.isOpened()): 
        ret, frame = vid.read()
        frame_count += 1

        if ret == True:
            start = time.time()
            predict(frame)
            time_sum += time.time() - start
            fps = round(frame_count/time_sum, 2)
            cv2.putText(frame,'Avg fps - '+str(fps),(20,45),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,2)

            out.write(frame)
            cv2.imshow('frame', frame) 
            if cv2.waitKey(1) == 27: 
                break
        else:  
            break

    print('Output video saved to', out_path)
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/inputs/3.jpg', help="Path to input image/video")
    parser.add_argument('--video', action='store_true', help="For video input")
    args = parser.parse_args()
    
    print(args.video)
    if args.video:
        video(args.input)
    else:
        image(args.input)
