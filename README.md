﻿# Image-Classifier
Simple python script that builds and trains a convolutional neural network (CNN) using TensorFlow and Keras to classify images from the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes. Can take pictures from google of any of the following:
'Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
And the CNN should be able to classify what the image is with a 68% accuracy rate.

To use:
run the following commands:
1. git clone https://github.com/randyla78/Image-Classifier.git
2. cd Image-Classifier
3. pip install -r requirements.txt

Run the code once to train the CNN. To import your own photo for testing, download photo of any of the objects/animals listed above, import into project as a 32x32 image and replace 'horse.jpg' with actual image name. 
(Optional) Save the trained model for future use by uncommenting the line 'model.save('image_classifier.model') in the script.
