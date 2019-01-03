import VisionConfig
from grip import GripPipeline
import numpy as np
import cscore
from networktables import NetworkTables
import logging
import threading

# set logging level
logging.basicConfig(level=logging.DEBUG)

# create the Pi Cam
cam = cscore.UsbCamera("picam", 0)

# the resolution is set in VisionConfig
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
# Comment this out before competition, or change port to allowed port number
mjpegServer = cscore.MjpegServer("httpserver", 8081)
mjpegServer.setSource(cam)

# initialize the netowrktable and wait for connection
cond = threading.Condition()
notified = [False]


def connectionListener(connected, info):
    print(info, '; Connected=%s' % connected)
    with cond:
        notified[0] = True
        cond.notify()


NetworkTables.initialize(server=VisionConfig.roboRIOIP)
NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)

with cond:
    print("Waiting")
    if not notified[0]:
        cond.wait()
print("Connected!")
# loop forever
table = NetworkTables.getTable('VisionData')
while True:

    # grab the frame from the sink, call it img
    time, img = cvsink.grabFrame(img)

    # If there's an error or no frame, lets skip this loop iteration
    if time == 0:
        # skip the rest of this iteration (no point in processing an image that doesnt exist)
        continue

    # Process image through pipeline
    pipeline.process(img)

    for blob in pipeline.find_blobs_output:
        x, y = blob.pt
        print("Blob Found. Data: (" + str(x) + ", " + str(y) + ")")
        table.putNumberArray("coord", blob.pt)