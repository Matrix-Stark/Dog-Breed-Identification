import tensorflow as tf
import os
import numpy as np
import cv2

modelFile = "C:/Users/KIIT0001/Desktop/Python/Dog_Breed_Identification/DogBreed_Identification/dogs.h5"
model = tf.keras.models.load_model(modelFile)

inputShape= (331,331)

allLables = np.load("C:/Users/KIIT0001/Desktop/Python/Dog_Breed_Identification/DogBreed_Identification/allDogsLables.npy")
categories = np.unique(allLables)

def prepareImage(img):
    resized = cv2.resize(img,inputShape , interpolation=cv2.INTER_AREA)
    imgResult = np.expand_dims(resized, axis=0)
    imgResult = imgResult / 255
    return imgResult

#load image
testImagePath = "C:/Users/KIIT0001/Desktop/Python/Dog_Breed_Identification/DogBreed_Identification/dog-breed-identification/train/0b6239db9b1649fe2f513357c82931aa.jpg"  
img = cv2.imread(testImagePath)
imageForModel = prepareImage(img)

#prediction
resultArray = model.predict(imageForModel, verbose=1)
answers = np.argmax(resultArray, axis=1)

text = categories[answers[0]]

font = cv2.FONT_HERSHEY_COMPLEX

cv2.putText(img, text, (0, 20), font, 1, (0, 0, 255), 2)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()