import cv2

import numpy as np
import time
import collections

np.random.seed(20)


class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()

        self.classesList.insert(0, '__Background__')

        self.colorList = np.random.uniform(
            low=0, high=255, size=(len(self.classesList), 3))

        # print(self.classesList)

    def onVideo(self):
        t0 = time.time()
        cap = cv2.VideoCapture(self.videoPath)
        filePath = f"output/v3/{self.videoPath[6:-4]}.txt"
        file = open(filePath, "w")
        frameNum = 0

        if (cap.isOpened() == False):
            print("Error opening video file")
            return
        (success, image) = cap.read()

        startTime = 0
        avgFps = 0

        while success:
            if frameNum == 500:
                t0 = time.time()
                avgFps = 0
            currentTime = time.time()
            fps = 1/(currentTime-startTime)
            avgFps += fps
            startTime = currentTime
            classLabelIDs, confidences, bboxs = self.net.detect(
                image, confThreshold=0.5)

            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))

            bboxIdx = cv2.dnn.NMSBoxes(
                bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)

            numPeople = 0
            tempStr = ""

            if (len(bboxIdx) != 0):
                for i in range(0, len(bboxIdx)):
                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(
                        classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelID]
                    classColor = [int(c) for c in self.colorList[classLabelID]]

                    displayText = "{}:{:.2f}".format(
                        classLabel, classConfidence)

                    x, y, w, h = bbox

                    # num objects, type of object, confidence, x,y,w,h

                    # only grabbing people because thats all we care about
                    if (classLabel == 'person'):
                        # handle large box across the center of the screen, maybe needs tweaked to ensure people dont fall into this
                        if (w < 400):
                            numPeople += 1
                            tempStr = tempStr + f"{x},{y},{x+w},{y+h},|"
                            #print(bboxs, confidences)
                            #print(bbox, classLabelID)
                            cv2.rectangle(image, (x, y), (x+w, y+h),
                                          color=classColor, thickness=1)
                            cv2.putText(image, displayText, (x, y-10),
                                        cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

                            lineWidth = min(int(w * 0.3), int(h * 0.3))
                            # upper left corner
                            cv2.line(image, (x, y), (x + lineWidth, y),
                                     classColor, thickness=5)
                            cv2.line(image, (x, y), (x, y + lineWidth),
                                     classColor, thickness=5)
                            # upper right corner
                            cv2.line(image, (x + w, y), (x + w -
                                     lineWidth, y), classColor, thickness=5)
                            cv2.line(image, (x + w, y), (x + w, y +
                                     lineWidth), classColor, thickness=5)
                            # lower left corner
                            cv2.line(image, (x, y + h), (x + lineWidth,
                                     y + h), classColor, thickness=5)
                            cv2.line(image, (x, y + h), (x, y + h -
                                     lineWidth), classColor, thickness=5)
                            # lower right corner
                            cv2.line(image, (x + w, y + h), (x + w -
                                     lineWidth, y + h), classColor, thickness=5)
                            cv2.line(image, (x + w, y + h), (x + w, y +
                                     h - lineWidth), classColor, thickness=5)
            frameResult = f"frame{frameNum}|{numPeople}|" + tempStr + "\n"
            file.write(frameResult)
            frameNum += 1

            cv2.putText(image, "FPS: " + str(int(fps)), (20, 70),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.imshow("Result", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if frameNum == 1000:
                break

            (success, image) = cap.read()
        file.close()
        cv2.destroyAllWindows()
        t1 = time.time() - t0
        print(f"Time to run - {t1}")
        print(f"Average fps - {avgFps / 500}")
