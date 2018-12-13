import multiprocessing
import time
import cv2
import mss
import numpy

title = "FPS benchmark"
start_time = time.time()
display_time = 2 # displays the frame rate every 2 second
fps = 0
sct = mss.mss()
# Set monitor size to capture
monitor = {"top": 40, "left": 0, "width": 800, "height": 640}


def GRABMSS_screen(q):
    while True:
        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.array(sct.grab(monitor))
        # To get real color we do this:
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        q.put_nowait(img)
        q.join()

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
    
if __name__=="__main__":
    # Queue
    q = multiprocessing.JoinableQueue()

    # creating new processes
    p1 = multiprocessing.Process(target=GRABMSS_screen, args=(q, ))
    p2 = multiprocessing.Process(target=SHOWMSS_screen, args=(q, ))

    # starting our processes
    p1.start()
    p2.start()
