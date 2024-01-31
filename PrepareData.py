import numpy as np 
import cv2

Image_Size = (331,331)
Image_Full_Size = (331,331,3)

trainMyImageFolder = "C:/Users/KIIT0001/Desktop/Python/Dog_Breed_Identification/DogBreed_Identification/dog-breed-identification/train"
#load the csv file
import pandas as pd

df = pd.read_csv("C:/Users/KIIT0001/Desktop/Python/Dog_Breed_Identification/DogBreed_Identification/dog-breed-identification/labels.csv")
print("Head of lables: ")
print("=================")

print(df.head())
print(df.describe())

print("Group by labels: ")
grouplables = df.groupby("breed")["id"].count()
print(grouplables.head(10))

# display one image

imgPath = "C:/Users/KIIT0001/Desktop/Python/Dog_Breed_Identification/DogBreed_Identification/dog-breed-identification/train/0a27d304c96918d440e79e6e9e245c3f.jpg"
img = cv2.imread(imgPath)


#Prepare all the images and lables as numpy arrays

allImages = []
allLables = []
import os

for ix , (image_name,breed) in enumerate(df[['id','breed']].values):
    img_dir = os.path.join(trainMyImageFolder, image_name + '.jpg')
    print(img_dir)

    img = cv2.imread(img_dir)
    resized = cv2.resize(img,Image_Size, interpolation= cv2.INTER_AREA)
    allImages.append(resized)
    allLables.append(breed)

print(len(allImages))
print(len(allLables))

print("save the data: ")
np.save("C:/Users/KIIT0001/Desktop/Python/Dog_Breed_Identification/DogBreed_Identification/allDogsImages.npy",allImages)
np.save("C:/Users/KIIT0001/Desktop/Python/Dog_Breed_Identification/DogBreed_Identification/allDogsLables.npy",allLables)
print("finish save the data")