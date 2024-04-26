model=load_model(r'C:\Users\Dhanjit Boro\Documents\Project\Mini_project\Project_local\trained_prot.h5')

faceDetect=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
labels_dict={0:'A', 1:'kalpojyoti', 2:'theophilus'}

frame = cv2.imread(r"C:\Users\Dhanjit Boro\Documents\Project\Mini_project\Project_local\dhanjitbaro0.jpg")

gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
faces= faceDetect.detectMultiScale(gray, 1.3, 3)

for x,y,w,h in faces:
    sub_face_img=gray[y:y+h, x:x+w]
    resized=cv2.resize(sub_face_img,(256,256))
    normalize = resized/255.0
    reshaped = np.stack((normalize,)*3,axis=-1)
    result=model.predict(np.expand_dims(reshaped, axis=0))
    label=np.argmax(result, axis=1)[0]
    print(label)
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
    cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
    cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,5,(255,255,255),10)

    os.chdir(r'C:\Users\Dhanjit Boro\Documents\Project\Mini_project\Project_local')
    cv2.imwrite('test.jpg',frame)
    img = mpimg.imread('test.jpg')
    imgplot = plt.imshow(img)
    k=cv2.waitKey(0)