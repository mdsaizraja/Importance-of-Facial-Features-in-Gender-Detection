import numpy as np
import cv2
import dlib
import os, os.path



def func(image_path,index):
    #path to frontal face detector
    
    cascade_path = "/Users/steveash/Downloads/opencv-4.0.0-alpha/data/haarcascades/haarcascade_frontalface_default.xml"
    
    #path to landmarks predictor
    
    predictor_path= "/Users/steveash/Desktop/learning/shape_predictor_68_face_landmarks.dat"
    
    
    # Create the haar cascade
    
    faceCascade = cv2.CascadeClassifier(cascade_path)
    
    # create the landmark predictor
    
    predictor = dlib.shape_predictor(predictor_path)
    
    # Read the image
    
    image = cv2.imread(image_path)
    
    # convert the image to grayscale
    
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow("img",image)
    
    # Detect faces in the image
    
    faces = faceCascade.detectMultiScale( image, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    print("Found {0} faces!".format(len(faces)))
    
    #to break the code, if the face is not found
    if len(faces) != 1:
        return 0
    

    for (x, y, w, h) in faces:
        # Draw a rectangle around the faces
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Converting the OpenCV rectangle coordinates to Dlib rectangle for further use
        
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        
        print(dlib_rect)
        
        # extracting feature from the recognized frontal face
        detected_landmarks = predictor(image, dlib_rect).parts()
        
        
        # numpy matrix of 68 coordinates
        landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])
        
        
        # copying the image so we can see side-by-side
        image_copy = image.copy()
        
        
        for idx, point in enumerate(landmarks):
            # iterating over coordinates of each of the 68 facial landmarks
            
            pos = (point[0, 0], point[0, 1])
            
            # typecasting of tuple pos into list for classification, training-testing
            
            mylist[index].append(list(pos))
            
            # for showing the landmarks in the image, idx = 0 to 67
            
            cv2.putText(image_copy, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 255))
            
            # draw points on the landmark positions
            
            cv2.circle(image_copy, pos, 3, color=(0, 255, 255))






#print(landmarks)
#small1 = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
#small2 = cv2.resize(image_copy, (0,0), fx=0.5, fy=0.5)
#print(mylist)
    cv2.imshow("Faces found", image)

    cv2.imshow("Landmarks found", image_copy)
    cv2.waitKey(500)
    return 1

#increments the list of lists
def increment():
    mylist.append([])
    jawline.append([])
    righteyebrow.append([])
    lefteyebrow.append([])
    nose.append([])
    righteye.append([])
    lefteye.append([])
    mouthinner.append([])
    mouthouter.append([])

#extract the facial features from whole face a
def featurelist(index):
    jawline[index] = mylist[index][:17]
    righteyebrow[index] = mylist[index][17:22]
    lefteyebrow[index] = mylist[index][22:27]
    nose[index] = mylist[index][27:36]
    righteye[index] = mylist[index][36:42]
    lefteye[index] = mylist[index][42:48]
    mouthouter[index] = mylist[index][48:61]
    mouthinner[index] = mylist[index][61:]
    writeintofile(index)

#write into the text file
def writeintofile(index):
    jawlinefile.write("{},".format(jawline[index]))
    righteyebrowfile.write("{},".format(righteyebrow[index]))
    lefteyebrowfile.write("{},".format(lefteyebrow[index]))
    nosefile.write("{},".format(nose[index]))
    righteyefile.write("{},".format(righteye[index]))
    lefteyefile.write("{},".format(lefteye[index]))
    mouthinnerfile.write("{},".format(mouthinner[index]))
    mouthouterfile.write("{},".format(mouthouter[index]))
    facefile.write("{},".format(mylist[index]))







image_path_list = []

# for creating the list of paths to image files
for file in os.listdir("/Users/steveash/Desktop/Female_1"):
    extension = os.path.splitext(file)[1]
    if extension.lower() not in [".jpg",".png",".pgm",".tiff",".bmp"]:
        continue
    image_path_list.append(os.path.join("/Users/steveash/Desktop/Female_1", file))


mylist = [[]]
jawline = [[]]
righteyebrow = [[]]
lefteyebrow = [[]]
nose = [[]]
righteye = [[]]
lefteye = [[]]
mouthouter = [[]]
mouthinner = [[]]
face = [[]]

jawlinefile = open("fjawline.txt","w")
righteyebrowfile = open("frighteyebrow.txt","w")
lefteyebrowfile = open("flefteyebrow.txt","w")
nosefile = open("fnose.txt","w")
righteyefile = open("frighteye.txt","w")
lefteyefile = open("flefteye.txt","w")
mouthinnerfile = open("fmouthinner.txt","w")
mouthouterfile = open("fmouthouter.txt","w")
facefile = open("fface.txt","w")


print( "this is {}".format(mylist))

# iterating over each image path
i=0
for value in image_path_list:
    if func(value,i):
        featurelist(i)
        increment()
        i+=1



# closing the opened file in the code
jawlinefile.close()
righteyefile.close()
lefteyefile.close()
nosefile.close()
righteyebrowfile.close()
lefteyebrowfile.close()
mouthinnerfile.close()
mouthouterfile.close()
facefile.close()

'''
    print("\n\nfinal list {}".format(mylist))
    print("\n\njawline list {}".format(jawline))
    print("\n\nrighteye list {}".format(righteye))
    print("\n\nnose list {}".format(nose))
    
    '''



