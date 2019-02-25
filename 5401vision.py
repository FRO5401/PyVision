import VisionConfig
from grip import GripPipeline
import numpy as np
import cscore
from networktables import NetworkTables
import logging
import threading
import itertools

# set logging level
# this is needed to get NetworkTables information
logging.basicConfig(level=logging.DEBUG)

# create the camera objects
drivecam = cscore.UsbCamera("drivecamcam", 0)
aimcam = cscore.UsbCamera("aimcam", 1)


# set video modes as determined in VisionConfig.py
drivecam.setVideoMode(cscore.VideoMode.PixelFormat.kMJPEG, VisionConfig.drive_resolution[0],
                      VisionConfig.drive_resolution[1], VisionConfig.drive_framerate)
aimcam.setVideoMode(cscore.VideoMode.PixelFormat.kMJPEG, VisionConfig.aim_resolution[0],
                    VisionConfig.aim_resolution[1], VisionConfig.aim_framerate)

# create a cv sink, which will grab images from the camera
cvsink = cscore.CvSink("cvsink")
cvsink.setSource(aimcam)

# create Pipeline Object
pipeline = GripPipeline()

# preallocate memory for images so that we don't allocate it every loop
img = np.zeros(shape=(VisionConfig.aim_resolution[1], VisionConfig.aim_resolution[0], 3), dtype=np.uint8)

# set up mjpeg server, the ip for this is 0.0.0.0:1180 and 0.0.0.0:1181
# These are FMX approved port numbers
mjpegServerDrive = cscore.MjpegServer("httpserver", 1181)
mjpegServerDrive.setSource(drivecam)
mjpegServerAim = cscore.MjpegServer("httpserver", 1182)
mjpegServerAim.setSource(aimcam)

# initialize the networktable and wait for connection
cond = threading.Condition()
notified = [False]


def connectionlistener(connected, info):
    print(info, '; Connected=%s' % connected)
    with cond:
        notified[0] = True
        cond.notify()


NetworkTables.initialize(server=VisionConfig.roboRIOIP)
NetworkTables.addConnectionListener(connectionlistener, immediateNotify=True)

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


    try:
        # Process image through pipeline
        pipeline.process(img)

        # Find the difference in X values for every combination of lines, if they are positive and within 1 pixel of error
        # then continue
        for x, y in itertools.product(range(0, pipeline.filter_lines_0_output.__len__() - 1),
                                      range(0, pipeline.filter_lines_1_output.__len__() - 1)):
            diffX1 = pipeline.filter_lines_1_output[y].x1 - pipeline.filter_lines_0_output[x].x1
            diffX2 = pipeline.filter_lines_1_output[y].x2 - pipeline.filter_lines_0_output[x].x2
            if abs(diffX2 - diffX1) <= 1 and diffX1 > 0:
                diffX = (diffX1 + diffX2) / 2
                xLeft = pipeline.filter_lines_0_output[x].x1
                break

        # Find the center of the target and the distance from the center of the image, then push to NT
        targetCenter = (diffX / 2) + xLeft
        centerDistance = (img.shape[1] / 2) - targetCenter
        table.putNumber("distance", centerDistance)
    except:
        # if lines don't exist, skip loop iteration
        continue



