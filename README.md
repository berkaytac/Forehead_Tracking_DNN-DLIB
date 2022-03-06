# Face and Forehead Tracking with DNN+Dlib

The purpose of this side project is to extract the forehead information using OpenCV's DNN. Warping is applied using
DLIB face landmarks eye information to overcome head rotations. This program works also in real time for **webcam-based
Forehead Tracking Applications**. The forehead rectangle position is mostly centered, taking into account head
movements.

[![Forehead](https://media.giphy.com/media/fPtLnPU6xFWj9s8Fqe/giphy.gif)](https://youtu.be/7b_rWvnbYHk)

## Installation

- Install these dependencies (imutils, Numpy, Dlib, Opencv-Python):

```
pip install -r requirements.txt
```

> The Dlib library has four primary prerequisites: Boost, Boost.Python, CMake and X11/XQuartx. [Read this article](https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/) to know how to easily install them.

- From the Line 7-9 of example_foreheadtracking.py, change soure name to;
    - "0" for Real-Time Webcam Forehead Tracking,
    - Path of video file for Forehead Tracking from a video.

```
# Source video
source = 0  # For webcam
# OR
source = "source_vid.avi"
```

Run the Forehead Tracking file:

```
python example_foreheadtracking.py
```

"output_video.mp4" output file will be created in the source folder.

## Simple Demo

```python
import cv2
import imutils
from forehead_tracking import ForeheadTracking

source = 0  # Webcam
# source = "source_vid.avi" # video file

cap = cv2.VideoCapture(source)
video_file = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = round(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_out = cv2.VideoWriter(video_file, fourcc, fps, (w, h))

# initialize the tracker
forehead = ForeheadTracking()
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame = forehead.analyze(frame)
        video_out.write(frame)
        frame = imutils.resize(frame, width=640)  # this is only for the imshow window
        cv2.imshow("Forehead Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
video_out.release()
cap.release()
cv2.destroyAllWindows()
```

## Analyze Frame

```python
forehead.analyze(frame)
```

1. Face rectangle is found with Opencv DNN.
2. The forehead rectangle is subtracted by proportioning it to the face rectangle.
3. Using the DLIB face landmark, the coordinates of the two eye centers are found and the angle between them is calculated.
4. By using the eye angle and the center of the forehead rectangle found before, the translated and rotated corner coordinates of the forehead rectangle are found.
5. After the forehead coordinates are extracted, the forehead values are revealed by warping the forehead picture.

### You want to help?
Your suggestions, bugs reports and pull requests are welcome and appreciated. You can also star ⭐️ the project!

