# Import VisionConfig
import VisionConfig
# Import GRIP pipeline
from grip import GripPipeline
# Import numpy, mainly for arrays
import numpy as np
# import cscore
import cscore
# import netowrktables
from networktables import NetworkTables
# import logging for network tables messages
import logging

# set this thing
logging.basicConfig(level=logging.DEBUG)

# create the usb cam
cam = cscore.UsbCamera("picam", 0)

# the resolution is 320x240 at 30 FPS
cam.setVideoMode(cscore.VideoMode.PixelFormat.kMJPEG, VisionConfig.resolution[0],
                 VisionConfig.resolution[1], VisionConfig.framerate)

# create a cv sink, which will grab images from the camera
cvsink = cscore.CvSink("cvsink")
cvsink.setSource(cam)

# create Pipeline Object
pipeline = GripPipeline()

# preallocate memory for images so that we dont allocate it every loop
img = np.zeros(shape=(240, 320, 3), dtype=np.uint8)


# set up mjpeg server, the ip for this is 0.0.0.0:8081
mjpegServer = cscore.MjpegServer("httpserver", 8081)
mjpegServer.setSource(cam)

# initialize the netowrktable
NetworkTables.initialize(server=VisionConfig.roboRIOIP)

# loop forever
while True:

    # grab the frame from the sink, call it img
    # this resets img, so it is not drawn on anymore
    time, img = cvsink.grabFrame(img)

    # If there's an error or no frame, lets skip this loop fam
    if time == 0:
        # skip the rest of this iteration (no point in processing an image that doesnt exist)
        continue

    # Process image through pipeline
    pipeline.process(img)