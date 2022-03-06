"""
This is demonstration of the Forehead Tracking.
"""
import cv2
import imutils
from forehead_tracking import ForeheadTracking

# Source video
# source = 0  # For webcam
source = "source_vid.avi"

cap = cv2.VideoCapture(source)

# output file
video_file = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')

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
    # Break the loop
    else:
        break
# release the video capture and video write objects
video_out.release()
cap.release()
# Closes all the frames
cv2.destroyAllWindows()
