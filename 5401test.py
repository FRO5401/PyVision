import VisionConfig
from grip import GripPipeline
from networktables import NetworkTables
import logging
import threading
import cv2
import itertools

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

    for x, y in itertools.product(range(0, pipeline.filter_lines_0_output.__len__() - 1),
                                  range(0, pipeline.filter_lines_1_output.__len__() - 1)):
        diffX1 = pipeline.filter_lines_1_output[y].x1 - pipeline.filter_lines_0_output[x].x1
        diffX2 = pipeline.filter_lines_1_output[y].x2 - pipeline.filter_lines_0_output[x].x2
        if abs(diffX2 - diffX1) <= 1 and diffX1 > 0:
            diffX = (diffX1 + diffX2) / 2
            xLeft = pipeline.filter_lines_0_output[x].x1
            break

    targetCenter = (diffX / 2) + xLeft
    distance = (img.shape[1] / 2) - targetCenter
    # table.putnumber("distance", distance)
    print("INFO: Distance: " + str(distance))
    print("INFO: DiffX: " + str(diffX))
    print("INFO: targetCenter: " + str(targetCenter))
