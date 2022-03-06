# Face and Forehead Tracking with DNN+Dlib

The purpose of this side project is to extract the forehead information using OpenCV's DNN. Warping is applied using DLIB face landmarks eye information to overcome head rotations.
This program works also in real time for **webcam-based Forehead Tracking Applications**. The forehead rectangle position is mostly centered, taking into account head movements.

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

## Documentation

In the following examples, `forehead` refers to an instance of the `ForheadTracking` class.

### Refresh the frame

```python
gaze.refresh(frame)
```

Pass the frame to analyze (numpy.ndarray). If you want to work with a video stream, you need to put this instruction in a loop, like the example above.

### Position of the left pupil

```python
gaze.pupil_left_coords()
```

Returns the coordinates (x,y) of the left pupil.

### Position of the right pupil

```python
gaze.pupil_right_coords()
```

Returns the coordinates (x,y) of the right pupil.

### Looking to the left

```python
gaze.is_left()
```

Returns `True` if the user is looking to the left.

### Looking to the right

```python
gaze.is_right()
```

Returns `True` if the user is looking to the right.

### Looking at the center

```python
gaze.is_center()
```

Returns `True` if the user is looking at the center.

### Horizontal direction of the gaze

```python
ratio = gaze.horizontal_ratio()
```

Returns a number between 0.0 and 1.0 that indicates the horizontal direction of the gaze. The extreme right is 0.0, the center is 0.5 and the extreme left is 1.0.

### Vertical direction of the gaze

```python
ratio = gaze.vertical_ratio()
```

Returns a number between 0.0 and 1.0 that indicates the vertical direction of the gaze. The extreme top is 0.0, the center is 0.5 and the extreme bottom is 1.0.

### Blinking

```python
gaze.is_blinking()
```

Returns `True` if the user's eyes are closed.

### Webcam frame

```python
frame = gaze.annotated_frame()
```

Returns the main frame with pupils highlighted.

## You want to help?

Your suggestions, bugs reports and pull requests are welcome and appreciated. You can also starring ⭐️ the project!

If the detection of your pupils is not completely optimal, you can send me a video sample of you looking in different directions. I would use it to improve the algorithm.

## Licensing

This project is released by Antoine Lamé under the terms of the MIT Open Source License. View LICENSE for more information.
