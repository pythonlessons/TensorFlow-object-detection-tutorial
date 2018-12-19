import multiprocessing
from multiprocessing import Pipe
import time
import cv2
import mss
import numpy as np
import datetime

title = "FPS benchmark"
start_time = time.time()
display_time = 2 # displays the frame rate every 2 second
fps = 0
sct = mss.mss()
# Set monitor size to capture
monitor = {"top": 40, "left": 0, "width": 800, "height": 640}

def GRABMSS_screen(p_input):
    while True:
        #Grab screen image
        img = np.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Put image from pipe
        p_input.send(img)
    
def SHOWMSS_screen(p_output):
    global fps, start_time
    while True:
        # Get image from pipe
        img = p_output.recv()
        
        # Display the picture
        cv2.imshow(title, img)
        
        # Calculate FPS
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
    # Pipes
    p_output, p_input = Pipe()

    # creating new processes
    p1 = multiprocessing.Process(target=GRABMSS_screen, args=(p_input,))
    p2 = multiprocessing.Process(target=SHOWMSS_screen, args=(p_output,))

    # starting our processes
    p1.start()
    p2.start()


