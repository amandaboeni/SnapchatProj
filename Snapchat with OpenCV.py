#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import os
import numpy as np
import dlib
from math import hypot
  
class App:
    #constructor
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
 
        #open source
        self.vid = MyVideoCapture(self.video_source)
 
        #canvas to fit
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()
 
        #buttons
        self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=10, command=self.snapshot)
        self.btn_Cyan=tkinter.Button(window, text="Cyan", width=10, command=self.Cyan)
        self.btn_Purple=tkinter.Button(window, text="Purple", width=10, command=self.Purple)
        self.btn_Spooky=tkinter.Button(window, text="Spooky", width=10, command=self.Spooky)
        self.btn_Face=tkinter.Button(window, text="Face", width=10, command=self.Face)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)
        self.btn_Cyan.pack(anchor=tkinter.CENTER, expand=True)
        self.btn_Purple.pack(anchor=tkinter.CENTER, expand=True)
        self.btn_Spooky.pack(anchor=tkinter.CENTER, expand=True)
        self.btn_Face.pack(anchor=tkinter.CENTER, expand=True)
 
        #delay: milliseconds
        self.delay = 15
        self.update()
 
        self.window.mainloop()
 
    def snapshot(self):
        #frame
        ret, frame = self.vid.get_frame()
        path = "C:/Users/Public/Desktop/Snapchat"
 
        if ret:
           cv2.imwrite(os.path.join(path, "frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    def Face(self):
         #----------------------------------------------------------------------------------
        #get nose image and create mask
        pig_nose = cv2.imread("Desktop\\pig_nose.png")
        dog_nose = cv2.imread("Desktop\\dog_nose.png")
        bear_nose = cv2.imread("Desktop\\bear_nose.png")
        deer_nose = cv2.imread("Desktop\\deer_nose.png")
        noses = [dog_nose, pig_nose, bear_nose, deer_nose]
        nose = 0
        #get ear image and create mask
        dog_left = cv2.imread("Desktop\\dog_left.png")
        dog_right = cv2.imread("Desktop\\dog_right.png")
        pig_ear_left = cv2.imread("Desktop\\pig_left.png")
        pig_ear_right = cv2.imread("Desktop\\pig_right.png")
        bear_left = cv2.imread("Desktop\\bear_left.png")
        bear_right = cv2.imread("Desktop\\bear_right.png")
        deer_left = cv2.imread("Desktop\\deer_left.png")
        deer_right = cv2.imread("Desktop\\deer_right.png")
        ears = [(dog_left, dog_right), (pig_ear_left, pig_ear_right), (bear_left, bear_right), (deer_left, deer_right)]
        ear = 0

        #choose to show
        show_right = True
        show_left = True
        show_nose = True

        #gets rows&cols
        ret, frame = self.vid.get_frame()
        rows, cols, _ = frame.shape
        nose_mask = np.zeros((rows, cols), np.uint8)
        ear_mask_left = np.zeros((rows, cols), np.uint8)
        ear_mask_right = np.zeros((rows, cols), np.uint8)

        #load face detector
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("Desktop\\shape_predictor_68_face_landmarks.dat")


        def landmark_coords(num):
            return (landmarks.part(num).x, landmarks.part(num).y)

        def add_part(orig, width, height, new_pos, frame):
            if (new_pos[1] + height > rows
                or new_pos[0] + width > cols
                or new_pos[1] <= 0
                or new_pos[0] <= 0):
                return
            part = cv2.resize(orig, (width, height), interpolation = cv2.INTER_AREA)
            part_gray = cv2.cvtColor(part, cv2.COLOR_BGR2GRAY)
            _, part_mask = cv2.threshold(part_gray, 15, 255, cv2.THRESH_BINARY_INV)
            part_area = frame[new_pos[1]: new_pos[1] + height,
                        new_pos[0]: new_pos[0] + width]
            part_area_mask = cv2.bitwise_and(part_area, part_area, mask=part_mask)
            final_part = cv2.add(part_area_mask, part)

            frame[new_pos[1]: new_pos[1] + height,
                        new_pos[0]: new_pos[0] + width] = final_part

        if __name__ == '__main__':
            while True:
                ret, frame = self.vid.get_frame()
                nose_mask.fill(0)
                ear_mask_left.fill(0)
                ear_mask_right.fill(0)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(frame)
                for face in faces:
                    landmarks = predictor(gray_frame, face)

                    #-----------------------nose COORDINATES-----------------------
                    top_nose = landmark_coords(29)
                    center_nose = landmark_coords(30)
                    left_nose = landmark_coords(31)
                    right_nose = landmark_coords(35)
                    nose_width = int(hypot(left_nose[0] - right_nose[0],
                                    left_nose[1] - right_nose[1]) * 1.7)
                    nose_height = int(nose_width * 0.70)

                    #eyebrow coordinates
                    center_left_eyebrow = landmark_coords(19)
                    left_left_eyebrow = landmark_coords(17)
                    right_left_eyebrow = landmark_coords(21)

                    center_right_eyebrow = landmark_coords(24)
                    left_right_eyebrow = landmark_coords(22)
                    right_right_eyebrow = landmark_coords(26)

                    #eyebrow width and height
                    eyebrow_left_width = int(hypot(left_left_eyebrow[0] - right_left_eyebrow[0],
                                    left_left_eyebrow[1] - right_left_eyebrow[1]))
                    eyebrow_right_width = int(hypot(left_right_eyebrow[0] - right_right_eyebrow[0],
                                    left_right_eyebrow[1] - right_right_eyebrow[1]))
                    eyebrow_left_height = int(eyebrow_left_width * 1.1)
                    eyebrow_right_height = int(eyebrow_right_width * 1.1)

                    #-----------------------new nose position, box coords-----------------------
                    new_nose_pos = (int(center_nose[0] - nose_width / 2),
                                        int(center_nose[1] - nose_height / 2))

                    #new ear positions
                    new_left_ear = (int(center_left_eyebrow[0] - eyebrow_left_width*1.1),
                                        int(center_left_eyebrow[1] - eyebrow_left_height))
                    new_right_ear = (int(center_right_eyebrow[0] - eyebrow_right_width*0.),
                                         int(center_right_eyebrow[1] - eyebrow_right_height))

                    #-----------------------add the new NOSE-----------------------
                    if show_nose:
                        add_part(noses[nose], nose_width, nose_height, new_nose_pos, frame)
                    #-----------------------add the new EARS-----------------------
                    if show_left: add_part(ears[ear][0], eyebrow_left_width, eyebrow_left_height, new_left_ear, frame)
                    #right ear
                    if show_right: add_part(ears[ear][1], eyebrow_right_width, eyebrow_right_height, new_right_ear, frame)
                
                norm = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imshow("frame", norm)

                key = cv2.waitKey(1)
                if key == 27: break
                if key == ord('r'): show_right = not show_right
                if key == ord('l'): show_left = not show_left
                if key == ord('n'): show_nose = not show_nose
                if key == ord('d'): nose = (nose+1)%len(noses)
                if key == ord('e'): ear = (ear+1)%len(ears)
                    
    #Cyan cover and screenshot
    def Cyan(self):
        ret, frame = self.vid.get_frame() 
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = np.zeros_like(frame)
        result[...,0] = gray
        blank = np.zeros_like(gray)
        
        shaded = [
            cv2.merge([gray, gray, blank]),
        ]
        
        cv2.imshow('Cyan', np.hstack(shaded))
        path = "C:/Users/aboeni/Desktop/Snapchat"
 
        if ret:
           cv2.imwrite(os.path.join(path, "frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg"), np.hstack(shaded))
    
    #Purple cover and screenshot
    def Purple(self):
        ret, frame = self.vid.get_frame() 
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = np.zeros_like(frame)
        result[...,0] = gray
        blank = np.zeros_like(gray)
        
        shaded = [
            cv2.merge([gray, blank, gray]),
        ]
        
        cv2.imshow('Purple', np.hstack(shaded))
        path = "C:/Users/aboeni/Desktop/Snapchat"
 
        if ret:
           cv2.imwrite(os.path.join(path, "frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg"), np.hstack(shaded))

    #gray and white and screenshot
    def Spooky(self):
        ret, frame = self.vid.get_frame() 
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray[gray <= 75] = 60
        gray[gray >= 151] = 255
        
        cv2.imshow('Spooky', gray)
        path = "C:/Users/aboeni/Desktop/Snapchat"
 
        if ret:
           cv2.imwrite(os.path.join(path, "frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg"), gray)
    
    def update(self):
        ret, frame = self.vid.get_frame()
 
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
 
        self.window.after(self.delay, self.update)
 
 
class MyVideoCapture:
    def __init__(self, video_source=0):
        #source
        self.vid = cv2.VideoCapture(video_source)
        
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
 
        #get height and width
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
 
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                #convert
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)
 
    #X to close
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
 
    #Window and name
App(tkinter.Tk(), "OpenCV and Snapchat")

