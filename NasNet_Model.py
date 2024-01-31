import numpy as np

Image_Size = (331,331)
Image_Full_Size = (331,331,3)
batchsize = 8
allImages = np.load("C:/Users/KIIT0001/Desktop/Python/Dog_Breed_Identification/DogBreed_Identification/allDogsImages.npy")
allLables = np.load("C:/Users/KIIT0001/Desktop/Python/Dog_Breed_Identification/DogBreed_Identification/allDogsLables.npy")

print(allImages.shape)
print(allLables.shape)

#convert labels text to integers

print(allLables)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
integerLables = le.fit_transform(allLables)
print(integerLables)

#unique integer lables
numofcategories = len(np.unique(integerLables))
print(numofcategories)

#convert the integers lables to categorical -> prepare for the train
from tensorflow import keras
from keras.utils import to_categorical

allLalbelsForModel = to_categorical(integerLables,num_classes = numofcategories)
print(allLalbelsForModel)

#normalize the images from 0-255 to 0-1
allImagesForModal = allImages / 255.0

#create train and test data
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(allImagesForModal, allLalbelsForModel, test_size=0.3, random_state=42)

print("x_train , x_test , y_train , y_test -----> shapes: ")

print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)

del allImages
del allLables
del integerLables
del allImagesForModal


#build for model

from keras.layers import Dense , Flatten
from keras.applications.nasnet import NASNetLarge
from keras.models import Model

myModel = NASNetLarge(input_shape=Image_Full_Size, weights='imagenet', include_top=False)


#train existing layer
for layer in myModel.layers:
    layer.trainable = False
    print(layer.name)

#add Flatten layer
plusFlattenLayer = Flatten()(myModel.output)




#add the last dense layer with out 120 classes
prediction = Dense(numofcategories, activation='softmax')(plusFlattenLayer)

model = Model(inputs=myModel.input, outputs=prediction)
#print(model.summary())

from keras.optimizers import Adam
lr = 1e-4
opt = Adam(lr)

model.compile(
    loss= 'categorical_crossentropy',
    optimizer = opt,
    metrics = ['accuracy'])

stepsPerEpoch = np.ceil(len(x_train) / batchsize)
validationsteps = np.ceil(len(x_test) / batchsize)

#early stopping
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau , EarlyStopping

best_model_file = "C:/Users/KIIT0001/Desktop/Python/Dog_Breed_Identification/DogBreed_Identification/dogs.h5"

callbacks = [
    ModelCheckpoint(best_model_file, verbose=1 , save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss',patience=3, factor=0.1, verbose=1,min_lr=1e-6),
    EarlyStopping(monitor='val_accuracy',patience=7,verbose=1)]

#train the model (fit)

r = model.fit(
    x_train, y_train,
    validation_data=(x_test,y_test),
    epochs = 30,
    batch_size = batchsize,
    steps_per_epoch=stepsPerEpoch,
    validation_steps = validationsteps,
    callbacks = [callbacks]
)