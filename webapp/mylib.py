def login():
    import cv2
    import numpy as np
    import os

    success = "no name"
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    newmodel=cv2.face_LBPHFaceRecognizer.create()
    newmodel.read("savedstate.xml")

 # Open Webcam
    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()
        #print(ret)
        #print(frame)
        #image, face = face_detector(frame)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if faces is ():
            image = frame
            face = []
        
        else :
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
                roi = frame[y:y+h, x:x+w]
                roi = cv2.resize(roi, (200, 200))
            image = frame
            face = roi

        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Pass face to prediction model
            # "results" comprises of a tuple containing the label and the confidence value
            results = newmodel.predict(face)
            print(results)
            if results[1] < 500:
                #print("i am in 1")
                confidence = int( 100 * (1 - (results[1])/400) )
                display_string = str(confidence) + '% Confident it is User'
            cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)

            if confidence > 80:
                #print("i am in 2")
                success = "Cyber Wizard"
                break
                #webbrowser.open('http://google.com/')

            else:
                #print("i am in 3")
                cv2.putText(image, "i dont know", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
                cv2.imshow('Face Recognition', image )

        except:
            #print("i am in 4")
            cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )
            pass

        if cv2.waitKey(1) == 13: #13 is the Enter Key
            #print("i am in 5")
            break
    #print("i am in 6")
    cap.release()
    cv2.destroyAllWindows()
    return success




def signup():
    import cv2
    import numpy as np

    # Load HAAR face classifier
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Load functions
    def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
        if faces is ():
            return None
    
        # Crop all faces found
        for (x,y,w,h) in faces:
            cropped_face = img[y:y+h, x:x+w]

        return cropped_face

    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    count = 0

    # Collect 100 samples of your face from webcam input
    while True:

        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count += 1
            face = cv2.resize(face_extractor(frame), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Save file in specified directory with unique name
            file_name_path = '/webapp/uploads/cyber' + str(count) + '.jpg'
            cv2.imwrite(file_name_path, face)

            print(count)

        else:
            print("Face not found")
            pass

        if cv2.waitKey(1) == 13 or count == 100: #13 is the Enter Key
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Collecting Samples Complete") 
    return "Signup Completed"





def signup_verify():
    import cv2
    import numpy as np
    from os import listdir
    from os.path import isfile, join
    print(cv2.__version__)
    # Get the training data we previously made
    data_path = '/webapp/uploads/'
    # a=listdir('d:/faces')
    # print(a)
    # """
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    # Create arrays for training data and labels
    Training_Data, Labels = [], []

    # Open training images in our datapath
    # Create a numpy array for training data
    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)

    # 
    # Create a numpy array for both training data and labels
    Labels = np.asarray(Labels, dtype=np.int32)
    model=cv2.face_LBPHFaceRecognizer.create()
    # Initialize facial recognizer
    # model = cv2.face_LBPHFaceRecognizer.create()
    # model=cv2.f
    # NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()

    # Let's train our model 
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    model.write("/webapp/savedstate.xml")
    print("Model trained successfully")
    return "Verification Successfully Competed"

