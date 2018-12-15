import multiprocessing
from queue import Empty
import time
import cv2
import mss
import numpy
import sys


def GRABMSS_screen(q):
    # Set monitor size to capture
    monitor = {"top": 40, "left": 0, "width": 800, "height": 640}

    with mss.mss() as sct:
        while True:
            # Get raw pixels from the screen, save it to a Numpy array
            img = numpy.array(sct.grab(monitor))
            q.put(img)

def SHOWMSS_screen(q):
    title = "FPS benchmark"
    start_time = time.time()
    display_time = 2 # displays the frame rate every 2 second
    fps = 0

    while True:
        try:
            img = q.get_nowait()
        except Empty:
            continue
            
        # Display the picture
        cv2.imshow(title, img)

        # Display the picture
        fps += 1
        TIME = time.time() - start_time
        if TIME >= display_time :
            print("FPS:", fps / TIME)
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
