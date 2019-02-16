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
    img = cv2.imread("field-tape-green-crop2.png")


    # Process image through pipeline
    pipeline.process(img)

    blobs = []
    for x in range(0, pipeline.find_blobs_output.__len__()):
        blobs.append(pipeline.find_blobs_output[x].pt)
    blobs.sort()

    diffx = blobs[1][0] - blobs[0][0]
    if diffx >= 200:
        blobcenter = diffx / 2 + blobs[0][0]
        distance = (img.shape[1] / 2) - blobcenter
        #table.putnumber("distance", distance)
        print("Distance: " + str(distance))
        print("Diffx: " + str(diffx))
    if 200 > diffx >= 190:
        try:
            diffx = blobs[2][0] - blobs[1][0]
        except IndexError:
            print("Failed Test")
            continue
        if diffx < 200:
            print("Failed Test")
            continue
        blobcenter = diffx / 2 + blobs[0][0]
        distance = (img.shape[1] / 2) - blobcenter
        # table.putnumber("distance", distance)
        print("ALT Distance:" + str(distance))
        print("ALT Diffx: " + str(diffx))
    else:
        continue
