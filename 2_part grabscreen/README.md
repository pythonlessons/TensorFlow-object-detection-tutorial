# Python grab screen tutorial
### Original text version of tutorial you can visit [here](http://pylessons/Tensorflow-object-detection-grab-screen/).

[![IMAGE ALT TEXT](https://github.com/pythonlessons/TensorFlow-object-detection-tutorial/blob/master/1_part%20grabscreen/1_YouTube.JPG)](https://www.youtube.com/watch?v=Wc2gFarBEE0 "TensorFlow object detection tutorial how to grab a screen")

In previous installation tutorial we mentioned that vision of this tutorial series is to create TensorFlow object detection model, that could detect CS:GO players. In this tutorial we’ll focus more on how to grab our monitor screen where we could detect objects. I must mention that we need to find the fastest way to grab screen, because later when we process images FPS drops, and if our screen grab FPS would be slow, this would affect our final frames per second.

At first we need to install all required libraries, so you can begin from installing opencv by writing this line: 
```
pip install opencv-python
```
then you can install mss library: 
```
python -m pip install --upgrade --user mss
```
If you don't have already, install numpy: 
```
pip install numpy
```
And at the end probably you will need pywin32 package, [download](https://github.com/Sentdex/pygta5/blob/master/grabscreen.py) it and install from wheel file. grabscreen.py file you can download from this GitHub repository. 
Now you should be ready to test grabscreen codes. So begin your code by importing libraries and setting variables that we'll use:

```
import time
import cv2
import mss
import numpy
from PIL import ImageGrab
from grabscreen import grab_screen

# title of our window
title = "FPS benchmark"
# set start time to current time
start_time = time.time()
# displays the frame rate every 2 second
display_time = 2
# Set primarry FPS to 0
fps = 0
# Load mss library as sct
sct = mss.mss()
# Set monitor size to capture to MSS
monitor = {"top": 40, "left": 0, "width": 800, "height": 640}
# Set monitor size to capture
mon = (0, 40, 800, 640)
```
We will begin with most basic and slowest PIL method. In this first code part I commented all lines, I wrote what it is done in each line and in other examples I simply copied PIL code and changed few lines of code, exactly that you can see on my youtube tutorial.
```
def screen_recordPIL():
    # set variables as global, that we could change them
    global fps, start_time
    # begin our loop
    while True:
        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.asarray(ImageGrab.grab(bbox=mon))
        # Display the picture
        cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # add one to fps
        fps+=1
        # calculate time difference
        TIME = time.time() - start_time
        # check if our 2 seconds passed
        if (TIME) >= display_time :
            print("FPS: ", fps / (TIME))
            # set fps again to zero
            fps = 0
            # set start time to current time again
            start_time = time.time()
        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
```
Here I used sentdex method of taking computer screen, it's much faster that PIL. But we must have grabscreen.py file in our local files to use it, so we move to final example.
```
def screen_grab():
    global fps, start_time
    while True:
        # Get raw pixels from the screen 
        img = grab_screen(region=mon)
        # Display the picture
        cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        fps+=1
        TIME = time.time() - start_time
        if (TIME) >= display_time :
            print("FPS: ", fps / (TIME))
            fps = 0
            start_time = time.time()
        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
```
This is the last example of taking computer screen, and I love it the most. Because for this method we don't need local files and most importantly, this method has more functionality. In their website you can find that it's possible to use this method to grab screen from different computer screens, must mention that this is impossible with previous methods. Moreover this method is as fast as second example.
```
def screen_recordMSS():
    global fps, start_time
    while True:
        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.array(sct.grab(monitor))
        # to get real color we do this:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        fps+=1
        TIME = time.time() - start_time
        if (TIME) >= display_time :
            print("FPS: ", fps / (TIME))
            fps = 0
            start_time = time.time()
        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
```
Here just simply uncomment function line, which you would like to test.
```
screen_recordMSS()
#screen_recordPIL()
#screen_grab()
```
In this short tutorial we learned 3 different ways to grab computer screen. It's sad that maximum performance we can get is around 20 FPS, but this is the best I found right now. If someone know better ways how to get more FPS, please let me know. So now we can move to another TensorFlow tutorials.
### Original text version of tutorial you can visit [here](http://pylessons/Tensorflow-object-detection-grab-screen/).
