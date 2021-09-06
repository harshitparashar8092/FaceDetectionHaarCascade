import cv2
smile_cascade=cv2.CascadeClassifier('haarcascade_smile.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')
def helper(gray,frame):
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        new_gray=gray[y:y+h,x:x+w]
        new_img=frame[y:y+h,x:x+w]
        eyes=eye_cascade.detectMultiScale(new_gray,1.1,3)
        for (ax,ay,aw,ah) in eyes:
            cv2.rectangle(new_img,(ax,ay),(ax+aw,ay+ah),(0,0,255),1)

        smiles=smile_cascade.detectMultiScale(new_gray,1.28,22)
        for (ex,ey,ew,eh) in smiles:
            cv2.rectangle(new_img,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
    return frame

vid=cv2.VideoCapture(0,cv2.CAP_DSHOW)

while True:
    _,frame=vid.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas=helper(gray,frame)
    cv2.imshow("Smile",canvas)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

vid.release()
cv2.destroyAllWindows()