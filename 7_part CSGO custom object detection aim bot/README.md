# TensorFlow CS:GO custom object detection aim bot
### Original text version of tutorial you can visit [here](http://pylessons.com/Tensorflow-object-detection-csgo-aim-bot).
### Use the same CSGO_frozen_inference_graph.pb download [link](https://drive.google.com/open?id=1U6JBcTKPEG9pxviCidVhkPe459XSJlXm).

Welcome to part 7 of our TensorFlow Object Detection API tutorial series. In this part, we're going to change our code, that we could find center of rectangles on our enemies, move our mouse to the center and shoot them.

<div align="center">
  <a href="https://www.youtube.com/watch?v=nJ3p96TevMw" target="_blank"><img src="https://github.com/pythonlessons/TensorFlow-object-detection-tutorial/blob/master/1_part%20images/6_YouTube.JPG" alt="TensorFlow object detection tutorial"></a>
</div>

In this tutorial are working with same files as we got in 6th [tutorial](https://github.com/pythonlessons/TensorFlow-object-detection-tutorial/tree/master/6_part%20actual%20CSGO%20object%20detection). To achieve desired goals for this tutorial we’ll need to add several lines to the code. At first we start with importing pyautogui library:
```
import pyautogui
```
This library will be used to move our mouse in game. But some games may not allow you to move mouse, then you will need to start python script with administrator rights, same as I am doing for CSGO in my YouTube tutorial video.

Next we are changing defined monitor size line to following. We are doing this because we will use our window width and height in other places to calculate right coordinates for our game. So to avoid mistakes and not to write same values in many places, we are defining our window size accordingly:
```
width = 800
height = 640
monitor = {"top": 80, "left": 0, "width": width, "height": height}
```

Before our main while loop we are defining new function, which we will use to aim and shoot enemies. As you can see in following function, we are calculating y differently from x. In my YouTube tutorial we’ll see that when we are calculating y in same way as x, we are shooting above the head. So we are removing that difference dividing our desired screen height by 9 and adding it to standard y height. 
```
def Shoot(mid_x, mid_y):
  x = int(mid_x*width)
  y = int(mid_y*height+height/9)
  pyautogui.moveTo(x,y)
  pyautogui.click()
```

Next, we are improving our code, while working in our main while loop. So we create following for loop. At first we initialize array_ch array, where we will place all our ch objects. Then we are going through boxes[0] array, and if we find our needed classes we search further. For example in our case classes[0][i] == 2 is equal to ch and if scores[0][i] >= 0.5 of this class is equal or more that 50 percent we detected our object. In this case we are taking boxes array numbers, where:<br>```boxes[0][i][0] – y axis upper start coordinates boxes[0][i][1] – x axis left start coordinates boxes[0][i][2] – y axis down start coordinates boxes[0][i][3] – x axis right start coordinates```

While subtracting same axis start coordinates and dividing them by two we receive center of two axis. This way we can calculate center of our detected rectangle.
And at the last line we are drawing a dot in a center.

```
array_ch = []
for i,b in enumerate(boxes[0]):
  if classes[0][i] == 2: # ch
    if scores[0][i] >= 0.5:
      mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
      mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
      array_ch.append([mid_x, mid_y])
      cv2.circle(image_np,(int(mid_x*width),int(mid_y*height)), 3, (0,0,255), -1)
```
These few line of code were only for one object, we do this for all four objects:
```
for i,b in enumerate(boxes[0]):
  if classes[0][i] == 2: # ch
    if scores[0][i] >= 0.5:
      mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
      mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
      array_ch.append([mid_x, mid_y])
      cv2.circle(image_np,(int(mid_x*width),int(mid_y*height)), 3, (0,0,255), -1)
  if classes[0][i] == 1: # c 
    if scores[0][i] >= 0.5:
      mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
      mid_y = boxes[0][i][0] + (boxes[0][i][2]-boxes[0][i][0])/6
      array_c.append([mid_x, mid_y])
      cv2.circle(image_np,(int(mid_x*width),int(mid_y*height)), 3, (50,150,255), -1)
  if classes[0][i] == 4: # th
    if scores[0][i] >= 0.5:
      mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
      mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
      array_th.append([mid_x, mid_y])
      cv2.circle(image_np,(int(mid_x*width),int(mid_y*height)), 3, (0,0,255), -1)
  if classes[0][i] == 3: # t
    if scores[0][i] >= 0.5:
      mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
      mid_y = boxes[0][i][0] + (boxes[0][i][2]-boxes[0][i][0])/6
      array_t.append([mid_x, mid_y])
      cv2.circle(image_np,(int(mid_x*width),int(mid_y*height)), 3, (50,150,255), -1)
```
After this we are making shooting function. So as a team = "t" we choose who we will be shooting at, at this case we are trying to shoot terrorists. So at first we check if we have detected terrorists heads, if we have detected at least one head, we call ```Shoot(mid_x, mid_y)``` function with needed coordinates. If we don't have any heads detected we check maybe we have detected terrorists bodies, if we did, we call the same shooting function.
```
team = "t"
if team == "c":
  if len(array_ch) > 0:
    Shoot(array_ch[0][0], array_ch[0][1])
  if len(array_ch) == 0 and len(array_c) > 0:
    Shoot(array_c[0][0], array_c[0][1])
if team == "t":
  if len(array_th) > 0:
    Shoot(array_th[0][0], array_th[0][1])
  if len(array_th) == 0 and len(array_t) > 0:
    Shoot(array_t[0][0], array_t[0][1])
```
If we would like to shoot to counter-terrorists we change "t" to "c". 

This was only a short explanation of code, full code you can download from above files.
In my YouTube video you can see how my CSGO aim bot model is working. For now, I am really disappointed about our FPS, because no one can play at these numbers... But I am glad that our bot can target to enemies quite accurate and shoot them. So maybe for next tutorial I will think what we could do to make it work faster for us.


