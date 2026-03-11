import cv2 
import joblib
import pandas as pd
import pyttsx3
import time
import warnings

warnings.filterwarnings('ignore')

cap = cv2.VideoCapture(0)

model = joblib.load("knn_digit_model.pkl")

engine = pyttsx3.init()

last_spoken_digit = None

last_spoken_time = 0
speak_delay = 2   # seconds

while True:
    ret,frame = cap.read()

    if not ret:
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

    x1,y1 = 200,150
    x2,y2 = 400,300
    roi = thresh[y1:y2,x1:x2]

    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.putText(frame,
                "Write Digit Here",
                (180,140),
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
                (0,255,0),
                2)
    



    # preprocess the ROI 
    roi_gussian_blur = cv2.GaussianBlur(roi,(5,5),0)
    roi_binary = cv2.adaptiveThreshold(roi_gussian_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

    countours,_ = cv2.findContours(
        roi_binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(countours) > 0:
        cnt = max(countours,key=cv2.contourArea)

        x,y,w,h = cv2.boundingRect(cnt)

        if w > 20 and h > 20:
            digit = roi_binary[y:y+h,x:x+w]
            
            cv2.rectangle(frame,(x+x1,y+y1),(x+x1+w,y+y1+h),(0,255,0),2)

            digit = cv2.copyMakeBorder(
                digit,
                20,20,20,20,
                cv2.BORDER_CONSTANT,
                value=0
            )

            digit = cv2.resize(digit,(28,28))

            digit = digit.flatten()

            digit = digit.reshape(1,-1)
            
            digit = pd.DataFrame(digit)

            prediction = model.predict(digit)

            prediction_digit = prediction[0]


            current_time = time.time()


            if current_time - last_spoken_time > speak_delay:
                engine.say(str(prediction_digit))
                engine.runAndWait()
                last_spoken_time = current_time
                time.sleep(0.5)

            cv2.putText(
                frame,
                f"Digit : {prediction_digit}",
                (50,50),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0,255,0),
                2
            )
            


    
    cv2.imshow("Digit Recognization",frame)

    cv2.imshow("ROI",roi)


    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
