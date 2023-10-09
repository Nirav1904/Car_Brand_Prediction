
# ML keras
from keras.layers import Input , Lambda , Dense , Flatten
# from keras.models import Model
import  keras.models

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator,load_img
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt



# resizing images
img_size = [ 224 , 224]
trainPath = "Datasets/train"
testPath = "Datasets/test"

resNet = ResNet50(input_shape= img_size + [3] , weights='imagenet' , include_top=False)
'''
imagesize = RGB channel
reuse weigths as imageNet that resnet is trained on
include_top = False ::: resNet is trained on Thousands of classes so output is also thousands of catagories so we are excluding top and last layer
it means it remove first and the last layer (input and output layers)
'''


# print( resNet.summary())


for layer in resNet.layers:
    layer.trainable = False # so the existing weights don't train again

folders = glob("Datasets/Train/*") # kind of like OS module

# flatten the output layers
x = Flatten()(resNet.output) # like

pred = Dense(len(folders) , activation='softmax')(x) # generate a dense layer
''' Dense layer : a simple layer of neurons in which each neurons receives
 input from all the neurons of previous layer'''

# model = Model(inputs = resNet.input , output = prediction)
model = keras.models.Model(inputs = resNet.input ,outputs = pred)
''' 
Now we have created a Neural network that takes 224 X 224 X 3  images and has dense
layer at the end
'''

#print model summary  to now   last layer has 3 nodes and i have only  3 output categories(model.summary()) # now last layer has 3 nodes as I have only 3 output catagories

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

''' Data augementation in training dataset'''
train_datagen = ImageDataGenerator( rescale= 1./255 , # just pixels are divided by 255 making them between 0 to 1
                                    shear_range=0.2,
                                    zoom_range= 0.2 ,
                                    horizontal_flip= True)
# so with this the same image will be taken and all the tilted and zoomed in out , rescalled images will be generated
# so that the purpose of the image will be preserved

test_datagen = ImageDataGenerator( rescale = 1./255)

trainingSet = train_datagen.flow_from_directory('Datasets/Train' ,
                                                target_size=(224,224),
                                                batch_size=32 ,
                                                class_mode='categorical') # for more than two classes catagorical use else binary is used

testSet = test_datagen.flow_from_directory("Datasets/Test",
                                           target_size=(224 ,224),
                                           batch_size=32,
                                           class_mode='categorical')

run = model.fit(
    trainingSet ,
    validation_data=testSet,
    epochs=50,
    steps_per_epoch=len(trainingSet),
    validation_steps=len(testSet)
)

#Loss graphs
plt.plot(run.history['loss'] , label = "Training Loss")
plt.plot(run.history['val_loss'] , label = "Validation Loss")
plt.legend()
plt.show()
plt.savefig("LossVal Loss")

# Accuracy graphs
plt.plot(run.history['accuracy'] , label = "Training accuracy")
plt.plot(run.history['val_accuracy'] , label = "Validation Accuracy")
plt.legend()
plt.show()
plt.savefig("AccVal Acc")

# saving the model with h5 file
from keras.models import  load_model

model.save('model_resnet50.h5')

''' ------------------Prediction-----------------------'''
y_pred = model.predict(testSet)

