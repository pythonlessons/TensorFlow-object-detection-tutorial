# TensorFlow Object Detection step by step custom object detection tutorial

Welcome to part 5 of the TensorFlow Object Detection API tutorial series. In this part and few in future, we're going to cover how we can track and detect our own custom objects with this API.

I am doing this by using the pre-built model to add custom detection objects to it. That’s a decent jump from my findings, and it’s quite hard to locate any full step-by-step tutorials, so hopefully I can help you with that. Once you finish this tutorial you will have the ability to train for any custom object you can think of (and create data for) - that’s an awesome skill to have in my opinion.

Alright, so this is my overview of the steps needed to do in this tutorial:
*	Collect at least 500 images that contain your object - The bare minimum would be about 100, ideally more like 1000 or more, but, the more images you have, the more tedious step 2 will be.
*	Annotate/label the images. I am personally using LabelImg. This process is basically drawing boxes around your object(s) in an image. The label program automatically will create an XML file that describes the object(s) in the pictures.
*	Split this data into train/test samples. Training data should be around 80% and testing around 20%.
*	Generate TF Records from these splits.
*	Setup a .config file for the model of choice (you could train your own from scratch, but we'll be using transfer learning).
*	Train our model.
*	Export inference graph from new trained model.
*	Detect custom objects in real time.

TensorFlow needs hundreds of images of an object to train a good detection classifier, best would be at least 1000 pictures for one object. To train a robust classifier, the training images should have random objects in the image along with the desired objects, and should have a variety of backgrounds and lighting conditions. There should be some images where the desired object is partially obscured, overlapped with something else, or only halfway in the picture. 

In my classifier, I will use four different objects I want to detect (terrorist, terrorist head, counter-terrorist and counter-terrorist head). I used to play CSGO with harmless bots to gather hundreds of these images playing on few different game maps, also I was grabbing some pictures from google, to make them different from my game. 