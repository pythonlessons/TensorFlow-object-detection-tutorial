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

### Label Pictures

Here comes the best part. With all the pictures gathered, it’s time to label the desired objects in every picture. LabelImg is a great tool for labeling images, and its GitHub page has very clear instructions on how to install and use it.

[LabelImg GitHub link](https://github.com/tzutalin/labelImg)
[LabelImg download link](https://www.dropbox.com/s/tq7zfrcwl44vxan/windows_v1.6.0.zip?dl=1)

Download and install LabelImg, when running this, you should get a GUI window. From here, choose to open dir and pick the directory that you saved all of your images to. Now, you can begin to annotate images with the create rectbox button. Draw your box, add the name in, and hit ok. Save, hit next image, and repeat! You can press the w key to draw the box and do ctrl+s to save faster. For me it took in average 1 hour for 100 images, this depends on object quantity you have in image. Keep in mind, this will take a while!

LabelImg saves a .xml file containing the label data for each image. These .xml files will be used to generate TFRecords, which are one of the inputs to the TensorFlow trainer. Once you have labeled and saved each image, there will be one .xml file for each image in the ```\test``` and ```\train``` directories.

Once you have labeled your images, we're going to separate them into training and testing groups. To do this, just copy about 20% of your images and their annotation XML files to a new dir called test and then copy the remaining ones to a new dir called train.

With the images labeled, it’s time to generate the TFRecords that serve as input data to the TensorFlow training model. This tutorial uses the xml_to_csv.py and generate_tfrecord.py scripts from <b>EdjeElectronics</b> repository, with some slight modifications to work with our directory structure.
