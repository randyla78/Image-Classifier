import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

#get datasets from keras (in the format of two tuples)
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
#scale data down
training_images = training_images / 255
testing_images = testing_images / 255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

#display 16 images
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()

#take first 20k to train neural network
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]


#-=(+)=-
#building neural network
model = models.Sequential() #define neural network
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3))) #define input layer
#^to know more abt activation and neural network, waych intro to neural network vid by nerualnine
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu')) #filters for features in an image
model.add(layers.MaxPooling2D((2,2))) #MaxPooling reduces image to the essential information
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax')) #scales results so that probabilities add up to 1

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels)) #10 epochs mean model will see the same data 10 times

loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

#run with this line to save model (so that we dont have to train model from scratch every time we run the program
#model.save('image_classifier.model')

#-=(+)=-
#run with this line after saving to load the previously saved model
#model = models.load_model('image_classifier.model') #Make sure to comment everything between the two -=(+)=- before loading


#run code below after loading in order to test any image from google (make sure to convert image to 32x32)
img = cv.imread('horse.jpg')
img = cv.cvtcolor(img, cv.COLOR_BGR2GRB)
plt.imshow(img, cmap=plt.cm.binary)
prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f'Prediction is {class_names[index]}')