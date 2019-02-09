import VisionConfig
from grip import GripPipeline
import numpy as np
from networktables import NetworkTables
import logging
import threading
import cv2

# set logging level
logging.basicConfig(level=logging.DEBUG)

# create Pipeline Object
pipeline = GripPipeline()

# preallocate memory for images so that we dont allocate it every loop
img = np.zeros(shape=(240, 320, 3), dtype=np.uint8)

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

table = NetworkTables.getTable('VisionData')
while True:

    # grab the frame from the sink, call it img
    img = cv2.imread("field-tape-green.png")


    # Process image through pipeline
    pipeline.process(img)

    blobs = []
    for x in range(0, pipeline.find_blobs_output.__len__()):
        blobs.append(pipeline.find_blobs_output[x].pt)
    blobs.sort()
    diffx = ((blobs[1][0] - blobs[0][0]) / 2) + blobs[0][0]

    if diffx != (img.shape[1] / 2):
        distance = (img.shape[1] / 2) - diffx

    table.putnumber("distance", distance)
