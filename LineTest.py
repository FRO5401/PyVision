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
img = cv2.imread("field-tape-green.png")
while True:

    pipeline.process(img)

    try:
        centerXLeft = (pipeline.filter_lines_0_output[0].x1 + pipeline.filter_lines_0_output[0].x2) / 2
        centerXRight = (pipeline.filter_lines_1_output[0].x1 + pipeline.filter_lines_1_output[0].x2) / 2
    except IndexError:
        print("ERROR: No Matching Pair of Vision Targets Found. Continuing Loop")
        continue

    diffX = centerXRight - centerXLeft
    targetCenter = (diffX / 2) + centerXLeft
    distance = (img.shape[1] / 2) - targetCenter
    print("INFO: Distance: " + str(distance))
    print("INFO: DiffX: " + str(diffX))
