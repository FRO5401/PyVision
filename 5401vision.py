import VisionConfig
from grip import GripPipeline
import numpy as np
import cscore
from networktables import NetworkTables
import logging
import threading

# set logging level
# this is needed to get NetworkTables information
logging.basicConfig(level=logging.DEBUG)

# create the camera objects
picam = cscore.UsbCamera("picam", 0)
usbcam = cscore.UsbCamera("usbcam", 1)


# set video modes as determined in VisionConfig.py
picam.setVideoMode(cscore.VideoMode.PixelFormat.kMJPEG, VisionConfig.pi_resolution[0],
                   VisionConfig.pi_resolution[1], VisionConfig.pi_framerate)
usbcam.setVideoMode(cscore.VideoMode.PixelFormat.kMJPEG, VisionConfig.usb_resolution[0],
                    VisionConfig.usb_resolution[1], VisionConfig.usb_framerate)

# create a cv sink, which will grab images from the camera
cvsink = cscore.CvSink("cvsink")
cvsink.setSource(picam)

# create Pipeline Object
pipeline = GripPipeline()

# preallocate memory for images so that we don't allocate it every loop
img = np.zeros(shape=(VisionConfig.pi_resolution[1], VisionConfig.pi_resolution[0], 3), dtype=np.uint8)

# set up mjpeg server, the ip for this is 0.0.0.0:1180 and 0.0.0.0:1181
# These are FMX approved port numbers
mjpegServerPi = cscore.MjpegServer("httpserver", 1180)
mjpegServerPi.setSource(picam)
mjpegServerUsb = cscore.MjpegServer("httpserver", 1181)
mjpegServerUsb.setSource(usbcam)

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

    # Process image through pipeline
    pipeline.process(img)

    # Get all center coordinates for blobs and put them in a list
    blobs = []
    for x in range(0, pipeline.find_blobs_output.__len__()):
        blobs.append(pipeline.find_blobs_output[x].pt)
    blobs.sort()

    # get the difference in X values for the 2 first leftmost blobs if they exist
    try:
        diffx1 = blobs[1][0] - blobs[0][0]
    except IndexError:
        # if they dont exist, skip loop iteration
        continue
    # get the difference in X values for the 2nd and 3rd blobs if they exist
    try:
        diffx2 = blobs[2][0] - blobs[1][0]
    except IndexError:
        # if the 3rd blob doesnt exist then continue anyways
        diffx2 = False
        pass
    # make sure we drive between targets with a bigger difference
    if diffx1 > diffx2:
        # find the center between the two blobs
        blobcenter = diffx1 / 2 + blobs[0][0]
        # find the distance from the center of the image
        distance = (img.shape[1] / 2) - blobcenter
        # put that distance in the NetworkTable
        table.putnumber("distance", distance)
    # if the difference isn't correct do this
    if diffx2 > diffx1:
        blobcenter = diffx2 / 2 + blobs[0][0]
        distance = (img.shape[1] / 2) - blobcenter
        table.putnumber("distance", distance)
    else:
        # if no blobs found, keep running
        continue
