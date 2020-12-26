# explore this
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_table_of_contents_calib3d/py_table_of_contents_calib3d.html


# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import os
import numpy as np
from skimage.draw import polygon
import json
import matplotlib.pyplot as plt
import time


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1, help="whether or not the Raspberry Pi camera should be used")
ap.add_argument("-d", "--delay", type=int, default=60, help="Time of video capture")
ap.add_argument("-f", "--video-file", default=None, help="provide video file to analyze")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

try:     
    # creating a folder named data 
    if not os.path.exists('data'): 
        os.makedirs('data') 
# if not created then raise error 
except OSError: 
    print ('Error: Creating directory of data') 


# frame 
currentframe = 0

# Green color in BGR 
color = (251, 105, 86) 

# Line thickness of 9 px 
thickness = 3

# font of text
font = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 0.5
fontcolor = (0, 0, 255)

BREAK_POINTS = [16, 21, 26, 35, 41, 47, ]

CLOSURE = {16: 15, 21: 20, 26: 25, 35: 30, 41: 36, 47: 42}

FACE_PARTS = {"cheek": [0, 17], 
              "l_eyebrow": [17, 21], 
              "r_eyebrow": [22, 26],
              "l_eye": [36, 42],
              "r_eye": [42, 48],
              "nose_bone": [27, 30],
              "nose_base": [31, 35],
              "lips": [48, 67]  
              }

# FACE_BOUNDARY = list(range(0, 16)) + [78, 76, 77] + list(range(69, 76)) + [79]
FACE_BOUNDARY = list(range(0, 16)) + [78, 74, 79, 73, 72, 80, 71, 70, 69, 68, 76, 75, 77]

ROI = {
       "lface_roi" : [48, 31, 27, 39, 40, 41, 36],
	   "rface_roi" : [54, 35, 27, 42, 47, 45, 46]
}

results = []

def mark_roi(rects, gray, frame):
    global currentframe, color, thickness, font, fontscale, fontcolor, BREAK_POINTS, CLOSURE
    point = 0
	# loop over the face detections
    for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # slight adjustment to the shape lip corner coordinates
        shape[31][0] -= 20  
        shape[35][0] += 20
        
        shape[48][0] -= 40  
        shape[54][0] += 40
        
        
        name = './data/frame' + str(currentframe) + '.jpg'
        
        # get the forehead coords
        
        xbl = shape[19][0]
        ybl = shape[19][1] - 10
        
        xbr = shape[24][0]
        ybr = shape[24][1] - 10
        
        xul = xbl
        yul = ybl - 50
        
        xur = xbr
        yur = yul

        y1, x1 = polygon(np.array([ybl, ybr, yur, yul]), np.array([xbl, xbr, xur, xul]))
        y2, x2 = polygon(shape[ROI["lface_roi"]][:, 1], shape[ROI["lface_roi"]][:, 0])
        y3, x3 = polygon(shape[ROI["rface_roi"]][:, 1], shape[ROI["rface_roi"]][:, 0])
        
        #print(points.shape)
        # # crop_face_part(shape[ROI["lface_roi"]], frame, name=name)
        crop_and_save((np.r_[y1, y2, y3], 
                       np.r_[x1, x2, x3]), frame, name=name)
		# # increasing counter so that it will 
        # show how many frames are created 
        currentframe += 1

def mark_boundaries(rects, gray, frame):
    global currentframe, color, thickness, font, fontscale, fontcolor, BREAK_POINTS, CLOSURE
    point = 0
	# loop over the face detections
    for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
        for (x, y) in shape:
            if point == 0:
                prevX = x
                prevY = y
                #cv2.putText(frame, str(point), (x-1, y-1), font, fontscale, fontcolor)
            else:
                if point in BREAK_POINTS:
                    start_point = (prevX, prevY)
                    end_point = (x, y)    
                    cv2.line(frame, start_point, end_point, color, thickness)
                        
                    start_point = (x, y)
                    #cv2.putText(frame, str(point), (x-1, y-1), font, fontscale, fontcolor)
                    try:
                        end_point = tuple(shape[CLOSURE[point]])
                        cv2.line(frame, start_point, end_point, color, thickness)
                    except:
                        pass
                else:
                    #cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                    if point - 1 in BREAK_POINTS:
                        #cv2.putText(frame, str(point), (x-1, y-1), font, fontscale, fontcolor)
                        pass
                    else:
                        #cv2.putText(frame, str(point), (x-1, y-1), font, fontscale, fontcolor)
                        start_point = (prevX, prevY)
                        end_point = (x, y)
                        cv2.line(frame, start_point, end_point, color, thickness)
                        if currentframe > 100 and currentframe % 1000 == 0:
                            name = './data/frame' + str(currentframe) + '.jpg'
                            cv2.imwrite (name, frame)
                            # increasing counter so that it will 
                            # show how many frames are created 
                        currentframe += 1
                prevX = x
                prevY = y
            point += 1

def mark_face_boundaries(rects, gray, frame):
    global currentframe, color, thickness, font, fontscale, fontcolor, BREAK_POINTS, CLOSURE
    point = 0
	# loop over the face detections
    for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        counter = 0
        #for (x, y) in shape:
        #   cv2.putText(frame, str(counter), (x-1, y-1), font, fontscale,  (0, 255, 0))
        for point in FACE_BOUNDARY:
            if counter == 0:
                prev = shape[point]
                pass
            else:
                cv2.line(frame, (prev[0], prev[1]), (shape[point][0], shape[point][1]), color, thickness)
                prev = shape[point]
            counter += 1

def mark_right_cheeks(rects, gray, frame):
    
    global currentframe, color, thickness, font, fontscale, fontcolor, BREAK_POINTS, CLOSURE, FACE_PARTS
 
    point = 0
     # loop over the face detections
    for rect in rects:
	    # determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
	    # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
        for (x, y) in shape[FACE_PARTS["r_eye"][0]+3: FACE_PARTS["r_eye"][1]]:
            if point == 0:
                pass
            else:
                # cv2.line(frame, (prevX, prevY), (x, y + 15), color, thickness)
                pass
            prevX = x
            prevY = y + 15
            point += 1

        xx, yy = tuple(shape[FACE_PARTS["r_eye"][0]])
        xx, yy = xx, yy + 15
        # cv2.line(frame, (prevX, prevY), (xx, yy), color, thickness)
        
        # now draw a line along the nose line
        slope = (shape[34][1] - shape[35][1]) / (shape[34][0] - shape[35][0])
        xn = shape[35][0] + 20
        yn = int(slope * (xn) + (shape[34][1] - slope * shape[34][0]))
        # cv2.line(frame, (xx, yy), (xn, yn), color, thickness)
        # cv2.fillPoly(frame, pts = [shape[FACE_PARTS["cheek"][0]: FACE_PARTS["cheek"][1]]], color=(255,255,255))
        
        eyeline = [[x, y + 20] for x, y in shape[FACE_PARTS["r_eye"][0]+3: FACE_PARTS["r_eye"][1]]]
        cheekline = [[x - 10, y] for x, y in shape[12: 16]]
        
        # add average of 42, 27
        xe = int((shape[42][0] + shape[27][0]) / 2)
        ye = int((shape[42][1] + shape[27][1]) / 2)
        
        # add a new point to capture cheekline
        boundary = np.r_[np.array([[shape[16][0] - 10, shape[16][1] - 10]]), np.array(eyeline),
                            np.array([[xx, yy]]),
                            np.array([[xn, yn]]),
                            np.array(cheekline),
                            ]
        #cv2.fillPoly(frame, pts = [boundary], color=(170,224,243), offset=(2, 2))
        image = cv2.polylines(frame, [boundary],  
                      isClosed=True, color=(255,224,243), thickness=1) 
        
        # add another point to close the loop
        len = 30
        ylen = 50
        
        x1 = shape[0][0] - 0.1 * (shape[29][0] - shape[0][0])
        y1 = shape[19][1] - 0.5 * (shape[57][1] - shape[19][1])        
        cv2.line(frame, (int(x1), int(y1)), (int(x1) + len, int(y1)), color, thickness)
        cv2.line(frame, (int(x1), int(y1)), (int(x1), int(y1) + ylen), color, thickness)
        
        x2 = x1 + 1.05 * (shape[16][0] - shape[0][0])
        y2 = y1
        cv2.line(frame, (int(x2), int(y2)), (int(x2) - len, int(y2)), color, thickness)
        cv2.line(frame, (int(x2), int(y2)), (int(x2), int(y2) + ylen), color, thickness)
        
        x3 = x1
        y3 = shape[8][1] - len
        cv2.line(frame, (int(x3), int(y3)), (int(x3) + len, int(y3)), color, thickness)
        cv2.line(frame, (int(x3), int(y3)), (int(x3), int(y3) - ylen), color, thickness)
        
        x4 = x2
        y4 = y3
        cv2.line(frame, (int(x4), int(y4)), (int(x4) - len, int(y4)), color, thickness)
        cv2.line(frame, (int(x4), int(y4)), (int(x4), int(y4) - ylen), color, thickness)
        
def mark_left_cheeks(rects, gray, frame):
    
    global currentframe, color, thickness, font, fontscale, fontcolor, BREAK_POINTS, CLOSURE, FACE_PARTS
 
    point = 0
    # loop over the face detections
    for rect in rects:
	    # determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
	    # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
		
        xx, yy = tuple(shape[FACE_PARTS["l_eye"][0]])
        xx, yy = xx, yy + 15
        
        # now draw a line along the nose line
        slope = (shape[31][1] - shape[32][1]) / (shape[31][0] - shape[32][0])
        xn = shape[31][0] - 20
        yn = int(slope * (xn) + (shape[31][1] - slope * shape[31][0]))
        # cv2.line(frame, (xx, yy), (xn, yn), color, thickness)
        # cv2.fillPoly(frame, pts = [shape[FACE_PARTS["cheek"][0]: FACE_PARTS["cheek"][1]]], color=(255,255,255))
        
        eyeline = [[x, y + 20] for x, y in shape[FACE_PARTS["l_eye"][0]+3: FACE_PARTS["l_eye"][1]]]
        cheekline = [[x + 10, y] for x, y in shape[0: 5]]
        
        # add a new point to capture cheekline
        boundary = np.r_[#np.array([[shape[0][0] + 10, shape[0][1] - 10]]), 
                            np.flip(np.array(eyeline), axis=0),
                            np.array([[xn, yn]]),
                            np.flip(np.array(cheekline), axis=0),
                            ]
        #cv2.fillPoly(frame, pts = [boundary], color=(170,224,243), offset=(2, 2))
        cv2.polylines(frame, [boundary],  
                      isClosed=True, color=(255,224,243), thickness=1) 
        
def mark_forehead(rects, gray, frame):
    
    global currentframe, color, thickness, font, fontscale, fontcolor, BREAK_POINTS, CLOSURE, FACE_PARTS

    # loop over the face detections
    for rect in rects:
	    # determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
	    # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
	
        xbl = shape[17][0] + 10
        ybl = shape[19][1]
        
        xbr = shape[26][0] - 10
        ybr = shape[24][1]
        
        xul = xbl
        yul = ybl - 50
        
        xur = xbr
        yur = yul

        boundary = np.r_[np.array([[xbl, ybl]]), np.array([[xbr, ybr]]), np.array([[xur, yur]]), np.array([[xul, yul]])]
        cv2.polylines(frame, [boundary], 
                      isClosed=True, color=(255,224,243), thickness=1) 
        
        #if currentframe > 100 and currentframe % 10 == 0:
        if currentframe > 0:
            name = './data/frame' + str(currentframe) + '.jpg'
            #cv2.imwrite (name, frame)      			
            get_face_patch(boundary, frame, name=name)
			# increasing counter so that it will 
            # show how many frames are created 
        currentframe += 1

def crop_and_save(points, image, savedir=None, prefix=None, pCounter=0, name=None):
    global results, a_file
    Y, X = points
    cropped_img = np.zeros(image.shape, dtype=np.uint8)
    cropped_img[points] = image[points]
    results.append((np.sum(image[points]) // points[1].shape)[0])
    if name == None:
       name = savedir + prefix + str(pCounter) + '.jpg'
    #cv2.imwrite(name, cropped_img)

def get_face_patch(vertices,image, savedir=None, prefix=None, pCounter=0, name=None):
    Y, X = polygon(vertices[:, 1], vertices[:, 0])
    cropped_img = np.zeros(image.shape, dtype=np.uint8)
    cropped_img[Y, X] = image[Y, X]
    if name == None:
       name = savedir + prefix + str(pCounter) + '.jpg'
    #cv2.imwrite(name, cropped_img)


# def crop_face_part(vertices, image, savedir=None, prefix=None, pCounter=0, name=None):
#     Y, X = polygon(vertices[:, 1], vertices[:, 0])
#     cropped_img = np.zeros(image.shape, dtype=np.uint8)
#     cropped_img[Y, X] = image[Y, X]
#     if name == None:
#        name = savedir + prefix + str(pCounter) + '.jpg'
#     cv2.imwrite(name, cropped_img)


vs = None
# loop over the frames from the video stream
def start_video_capture(args, detector):
    global vs
    # initialize the video stream and allow the cammera sensor to warmup
    print("[INFO] camera sensor warming up...")
    vs = VideoStream(usePiCamera=args["picamera"] > 0, resolution=(320, 240)).start()
    # time.sleep(2.0)

    while True:
        # grab the frame from the threaded video stream, resize it to
        # have a maximum width of 400 pixels, and convert it to
        # grayscale
        frame = vs.read()
        frame = imutils.resize(frame, width=1300)
        frame = cv2.flip(frame, 3)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)
        # mark_boundaries(rects, gray, frame)
        # mark_face_boundaries(rects, gray, frame)
        # mark_right_cheeks(rects, gray, frame)
        # mark_left_cheeks(rects, gray, frame)
        # mark_forehead(rects, gray, frame)
        mark_roi(rects, gray, frame)
        
        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        
        span = time.time() - start
        if span >= args["delay"]:
            break

def start_frame_capture(args, detector, mp4):
    print("[INFO] analysing frames of video file ... %s" % mp4)
    
    vidcap = cv2.VideoCapture(mp4)
    success, frame = vidcap.read()
    count = 0

    while success:
        # grab the frame from the threaded video stream, resize it to
        # have a maximum width of 400 pixels, and convert it to
        # grayscale
        frame = imutils.resize(frame, width=1300)
        frame = cv2.flip(frame, 3)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)
        # mark_boundaries(rects, gray, frame)
        # mark_face_boundaries(rects, gray, frame)
        # mark_right_cheeks(rects, gray, frame)
        # mark_left_cheeks(rects, gray, frame)
        # mark_forehead(rects, gray, frame)
        mark_roi(rects, gray, frame)
        
        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        success, frame = vidcap.read()

        
if args["video_file"] == None:
    start = time.time()
    print('Capture starts at: %s' % start)
    start_video_capture(args, detector)
    print('Capture ends at: %s' % time.time())
else:
    print("video file provided")
    start_frame_capture(args, detector, args["video_file"])


# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

plt.plot(range(len(results)), results, 'b')
plt.show()

a_file = open("D:\\Abhash\\Python\\Python_Practice\\signals.txt", "w")
print(results, file=a_file)
a_file.close()
