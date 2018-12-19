# Grab screen with multiprocessings
### Original text version of tutorial you can visit [here](https://pylessons.com/Tensorflow-object-detection-grab-screen-multiprocessing/).

Welcome everyone to part 9 of our TensorFlow object detection API series. This tutorial will be a little different from previous tutorials. 
UPDATE. I updated this tutorial, added grab screen code using multiprocessing pipes.

<div align="center">
  <a href="https://www.youtube.com/watch?v=3Yr1kYTIdV4" target="_blank"><img src="https://github.com/pythonlessons/TensorFlow-object-detection-tutorial/blob/master/1_part%20images/9_YouTube.jpg" alt="Grab screen with multiprocessings"></a>
</div><br>

In 8 part I told that I will be working with python multiprocessing to make code work in parallel with other processes. So I spent hours of learning how to use multiprocessing (was not using it before).

So I copied whole code from my second tutorial and removed ```screen_recordPIL``` and ```screen_grab``` functions. Left only to work with ```screen_recordMSS``` function. This function we can divide into two parts where we grab screen and where we show our grabbed screen. So this mean we will need to create two processes.

At first I divide whole code into two parts, first part we will call GRABMSS_screen. Next we need to put whole code into while loop, that it would run over and over. When we have our screen, we call ```q.put_nowait(img)``` command where we put our image into shared queue, and with following line ```q.join()``` we are saying wait since img will be copied to queue.
```
def GRABMSS_screen(q):
    while True:
        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.array(sct.grab(monitor))
        # To get real color we do this:
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        q.put_nowait(img)
        q.join()
```

Second function we will call SHOWMSS_screen. This function also will run in a while loop, and we always check if our queue is not empty. When we have something in queue we call ```q.get_nowait()``` command which takes everything from queue, and with ```q.task_done()``` we are locking the process, not to interrupt queue if we didn't finished picking up all data. After that we do same things as before, showing grabbed image and measuring FPS.
```
def SHOWMSS_screen(q):
    global fps, start_time
    while True:
        if not q.empty():
            img = q.get_nowait()
            q.task_done()
            # To get real color we do this:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Display the picture
            cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # Display the picture in grayscale
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

Right now, we have two different functions, we will use them in parallel processes.

If we want to run our code in multiprocessing we must begin our code with ```if __name__=="__main__":``` and we must run python script from command prompt elsewise if we'll run it from python shell, we won't get any prints, which we need here to measure FPS. So our full 3rd code part looks like this: 
```
if __name__=="__main__":
    # Queue
    q = multiprocessing.JoinableQueue()

    # creating new processes
    p1 = multiprocessing.Process(target=GRABMSS_screen, args=(q, ))
    p2 = multiprocessing.Process(target=SHOWMSS_screen, args=(q, ))

    # starting our processes
    p1.start()
    p2.start()
```

More about python multiprocessing and queues you can learn on this [link](https://docs.python.org/2/library/multiprocessing.html#multiprocessing.Queue.qsize). Short code explanation:
We begin with creating a chared queue:
```
# Queue
q = multiprocessing.JoinableQueue()
```
With following lines we are creating p1 and p2 processes which will run in background. p1 function will call GRABMSS_screen() function and p2 will call SHOWMSS_screen() function. As an argument for these functions we must give arguments, we give q there.
```
# creating new processes
p1 = multiprocessing.Process(target=GRABMSS_screen, args=(q, ))
p2 = multiprocessing.Process(target=SHOWMSS_screen, args=(q, ))
```
Final step is to start our processes, after these commands our grab screen function will run in background.
```
# starting our processes
p1.start()
p2.start()
```

For comparison I ran old code without multiprocessing and with multiprocessing. Here is results without multiprocessing:
<p align="center">
    <img src="https://github.com/pythonlessons/TensorFlow-object-detection-tutorial/blob/master/1_part%20images/09_FPS_slow.JPG"
</p><br>
We can see that average is about 19-20 FPS.
Here is results with multiprocessing:
<p align="center">
    <img src="https://github.com/pythonlessons/TensorFlow-object-detection-tutorial/blob/master/1_part%20images/09_FPS_fast.JPG"
</p><br>

We can see that average is about 32 FPS. So our final result is that our grab screen improved in around 50%. I would like like to impove it more, but for now I don't have ideas how to do that. Anyway results are much better than before !
  
### Original text version of tutorial you can visit [here](https://pylessons.com/Tensorflow-object-detection-grab-screen-multiprocessing/).
