
# importing libraries
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from keras.models import load_model

# Define path
# replace location placed below with file location in your device
training_path=r'C:\Users\Antarlin\Desktop\Data_Science\Deep Learning\CatDog\training_set\training_set'
# replace location placed below with file location in your device
testing_path=r'C:\Users\Antarlin\Desktop\Data_Science\Deep Learning\CatDog\test_set\test_set'

train_datagen=ImageDataGenerator(rescale=1/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1/255)

training_set=train_datagen.flow_from_directory(training_path,target_size=(64,64),batch_size=32,class_mode='binary',)

testing_set=test_datagen.flow_from_directory(testing_path,target_size=(64,64),batch_size=32,class_mode='binary',)

# lets see the data
training_data=next(training_set)

training_data[0].shape

plt.imshow(training_data[0][1])

# 1.model architecture
# 2.compile
# 3.fit

classifier=Sequential()
classifier.add(Conv2D(32,3,activation='relu',input_shape=(64,64,3)))
classifier.add(MaxPooling2D())
classifier.add(Conv2D(64,3,activation='relu'))
classifier.add(MaxPooling2D())
classifier.add(Conv2D(128,3,activation='relu'))
classifier.add(MaxPooling2D())

classifier.add(Flatten())
classifier.add(Dense(128,activation='relu'))
classifier.add(Dense(64,activation='relu'))
classifier.add(Dense(1,activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(training_set,epochs=10,validation_data=testing_set)

# saving the model
# replace location placed below with where u want to save the model in your device
classifier.save(r'C:\Users\Antarlin\Desktop\Data_Science\Deployment26July2021\CatsDogs\flask_app\Newfolder\catdog.h5')

# to check the classes

print(training_set.class_indices)

