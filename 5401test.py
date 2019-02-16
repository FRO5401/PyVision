import VisionConfig
from grip import GripPipeline
import numpy as np
from networktables import NetworkTables
import logging
import threading
import cv2

# set logging level
# this is needed to get NetworkTables information
logging.basicConfig(level=logging.DEBUG)

# create Pipeline Object
pipeline = GripPipeline()

# preallocate memory for images so that we don't allocate it every loop
img = np.zeros(shape=(240, 320, 3), dtype=np.uint8)

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

table = NetworkTables.getTable('VisionData')
while True:

    # get image data from the file
    img = cv2.imread("field-tape-green-crop.png")

    # Process image through pipeline
    pipeline.process(img)

    # Get all center coordinates for blobs and put them in a list
    blobs = []
    for x in range(0, pipeline.find_blobs_output.__len__()):
        blobs.append(pipeline.find_blobs_output[x].pt)
    blobs.sort()

    # get the difference in X values for the 2 first leftmost blobs
    diffx = blobs[1][0] - blobs[0][0]
    # make sure the difference between the blobs is correct for field use
    if diffx >= 200:
        # find the center between the two blobs
        blobcenter = diffx / 2 + blobs[0][0]
        # find the distance from the center of the image
        distance = (img.shape[1] / 2) - blobcenter
        # table.putnumber("distance", distance)
        # print the data
        print("Distance: " + str(distance))
        print("Diffx: " + str(diffx))
    # if the difference isn't correct do this
    if 200 > diffx >= 190:
        # try to use the next 2 coordinates
        # if blobs don't exist cleanly fail
        try:
            diffx = blobs[2][0] - blobs[1][0]
        except IndexError:
            print("Failed Test. Only 2 blobs exist in image.")
            continue
        if diffx < 200:
            print("Failed Test. Difference between targets is invalid.")
            continue
        blobcenter = diffx / 2 + blobs[0][0]
        distance = (img.shape[1] / 2) - blobcenter
        # table.putnumber("distance", distance)
        print("ALT Distance:" + str(distance))
        print("ALT Diffx: " + str(diffx))
    else:
        continue
