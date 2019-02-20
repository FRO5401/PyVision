import VisionConfig
from grip import GripPipeline
from networktables import NetworkTables
import logging
import threading
import cv2

# set logging level
# this is needed to get NetworkTables information
logging.basicConfig(level=logging.DEBUG)

# create Pipeline Object
pipeline = GripPipeline()

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
    img = cv2.imread("field-tape-green.png")

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
        print("ERROR: Either only 1 blob exists or no blobs found.")
        continue
    # get the difference in X values for the 2nd and 3rd blobs if they exist
    try:
        diffx2 = blobs[2][0] - blobs[1][0]
    except IndexError:
        # if the 3rd blob doesnt exist then continue anyways
        print("WARN: 3rd blob does not exist.")
        diffx2 = False
        pass
    # make sure we drive between targets with a bigger difference
    if diffx1 > diffx2:
        # find the center between the two blobs
        blobcenter = diffx1 / 2 + blobs[0][0]
        # find the distance from the center of the image
        distance = (img.shape[1] / 2) - blobcenter
        # table.putnumber("distance", distance)
        # print the data
        print("INFO: Distance: " + str(distance))
        print("INFO: Diffx: " + str(diffx1))
    # if the difference isn't correct do this
    if diffx2 > diffx1:
        print("INFO: Distance between first 2 blobs invalid. Trying next two.")
        blobcenter = diffx2 / 2 + blobs[0][0]
        distance = (img.shape[1] / 2) - blobcenter
        # table.putnumber("distance", distance)
        print("INFO: ALT Distance:" + str(distance))
        print("INFO: ALT Diffx: " + str(diffx2))
    else:
        continue
