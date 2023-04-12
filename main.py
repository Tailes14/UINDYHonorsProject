from yolov3.Detector import *
from yolov5.Detection import *
import os
import torch
import cv2
import re

numObjects = 0
posList = []


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    global numObjects
    global posList
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        #print(x, ' ', y)
        posList.append(x)
        posList.append(y)
        numObjects += 1

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        #print(x, ',', y)
        posList.append(x)
        posList.append(y)
        numObjects += 1


def getGT(filePath, imgPath):
    global numObjects
    global posList
    numObjects = 0
    posList = []

    file = open(filePath, 'a')
    img = cv2.imread(imgPath)
    cv2.imshow(imgPath[12:], img)
    cv2.setMouseCallback(imgPath[12:], click_event)

    cv2.waitKey(0)
    frameInfo = f"{imgPath[20:]}|{int(numObjects / 2)}|"
    i = 0
    for coord in posList:
        frameInfo = frameInfo + str(coord) + ","
        i += 1
        if i == 4:
            frameInfo = frameInfo + "|"
            i = 0
    frameInfo = frameInfo + "\n"
    print(frameInfo)
    file.write(frameInfo)
    cv2.destroyAllWindows()
    file.close()


def getFrames(numFrames, videoPath):
    cap = cv2.VideoCapture(videoPath)
    count = 0
    savePath = f"frameImages/{videoPath[6:-4]}"
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    os.chdir(savePath)
    while count != numFrames:
        ret, frame = cap.read()

        cv2.imshow('window-name', frame)
        cv2.imwrite(f"frame{count}.jpg", frame)
        count = count + 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()  # destroy all opened windows


def handleCoordsText(filePath):
    dataList = []
    with open(filePath) as file:
        for line in file:
            data = re.split(r'\|+', line)
            del data[-1]
            temp = data[:2]
            for val in data[2:]:
                temp.append(val[:-1])
            dataList.append(temp)
    return dataList


def getCoords(arr):
    coords = []
    for val in arr[2:]:
        temp = {}
        pos = val.split(",")
        temp["coords"] = [int(pos[0]), int(pos[1]), int(pos[2]), int(pos[3])]
        temp["flag"] = "f"
        coords.append(temp)
    return coords


def getIoU(gt, yolo):
    #print(gt, yolo)
    xA = np.maximum(gt[0], yolo[0])
    yA = np.maximum(gt[1], yolo[1])
    xB = np.minimum(gt[0] + gt[2], yolo[0] + yolo[2])
    yB = np.minimum(gt[1] + gt[3], yolo[1] + yolo[3])

    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)

    gtArea = (gt[2] + 1) * (gt[3] + 1)
    yoloArea = (yolo[2] + 1) * (yolo[3] + 1)

    iou = interArea / float(gtArea + yoloArea - interArea)

    return iou


def confusionMatrix(gt, pred, model):
    tp = 0
    fp = 0
    fn = 0

    count = 0
    while count < 500:
        gtCoords = getCoords(gt[count])
        yoloCoords = getCoords(pred[count])

        for gtSet in gtCoords:
            for yoloSet in yoloCoords:
                if (gtSet["flag"] != "t" and yoloSet["flag"] != "t"):
                    iou = getIoU(gtSet["coords"], yoloSet["coords"])
                    if iou > .6:
                        gtSet["flag"] = "t"
                        yoloSet["flag"] = "t"
                        tp += 1
                        '''
                        img = cv2.rectangle(img, (int(gtSet["coords"][0]), int(gtSet["coords"][1])), (int(gtSet["coords"][2]), int(gtSet["coords"][3])), (255,0,0), 2)
                        img = cv2.rectangle(img, (int(yoloSet["coords"][0]), int(yoloSet["coords"][1])),
                                            (int(yoloSet["coords"][2]), int(yoloSet["coords"][3])), (255, 0, 0), 2)
                        cv2.imshow("tp", img)
                        cv2.waitKey(0)
                        '''

        for gtSet in gtCoords:
            if gtSet["flag"] == "f":
                fn += 1
                '''
                img = cv2.rectangle(img, (int(gtSet["coords"][0]), int(gtSet["coords"][1])),
                                    (int(gtSet["coords"][2]), int(gtSet["coords"][3])), (0, 255, 0), 2)
                cv2.imshow("fn", img)
                cv2.waitKey(0)
                '''

        for yoloSet in yoloCoords:
            if yoloSet["flag"] == "f":
                fp += 1
                '''
                img = cv2.rectangle(img, (int(yoloSet["coords"][0]), int(yoloSet["coords"][1])),
                                    (int(yoloSet["coords"][2]), int(yoloSet["coords"][3])), (0, 0, 255), 2)
                cv2.imshow("fp", img)
                cv2.waitKey(0)
                '''
        count += 1

    print(f"{model}: recall - {tp / (tp + fn)}, precision - {tp / (tp + fp)}")
    print(f"tp - {tp}, fn - {fn}, fp - {fp}")


def main():

    modelType = "img"
    videoPath = "input/1feb3pm.MP4"

    if modelType == "v3":
        configPath = os.path.join("yolov3", "ssd_mobilenet.pbtxt")
        modelPath = os.path.join("yolov3", "frozen_inference_graph.pb")
        classesPath = os.path.join("yolov3", "coco.names")

        detector = Detector(videoPath, configPath, modelPath, classesPath)
        detector.onVideo()
    elif modelType == "v5":
        detector = V5Detection(videoPath)
        detector()
    elif modelType == "gt":
        #getFrames(500, videoPath)
        textPath = f"groundTruthFiles/{videoPath[6:-4]}.txt"
        count = 0
        if os.path.exists(textPath):
            print("getting the last frame that has data")
            with open(textPath, "r") as f:
                lastFrameData = f.readlines()[-1]
                lastFrameData = lastFrameData.split("|", 1)[0]
                lastFrameNum = ""
                for x in lastFrameData:
                    if x.isdigit():
                        lastFrameNum = lastFrameNum + x
                count = int(lastFrameNum) + 1
                print(
                    f"The last frame that was entered was frame #{count - 1}")
        else:
            print(f"creating ground truth file for {videoPath[6:-4]}")
            file = open(f"groundTruthFiles/{videoPath[6:-4]}.txt", 'w')
            file.close()

        while count < 500:
            getGT(textPath, f"frameImages/{videoPath[6:-4]}/frame{count}.jpg")
            count += 1
    elif modelType == "eval":
        v3Data = handleCoordsText(f"output/v3/{videoPath[6:-4]}.txt")
        v5Data = handleCoordsText(f"output/v5/{videoPath[6:-4]}.txt")
        gtData = handleCoordsText(f"groundTruthFiles/{videoPath[6:-4]}.txt")

        confusionMatrix(gtData, v5Data, "v3")

        gtData = handleCoordsText(f"groundTruthFiles/{videoPath[6:-4]}.txt")

        confusionMatrix(gtData, v5Data, "v5")
    elif modelType == "img":
        v3Data = handleCoordsText(f"output/v3/{videoPath[6:-4]}.txt")
        v5Data = handleCoordsText(f"output/v5/{videoPath[6:-4]}.txt")
        gtData = handleCoordsText(f"groundTruthFiles/{videoPath[6:-4]}.txt")
        temp = f"frameImages/{videoPath[6:-4]}/frame250.jpg"
        img = cv2.imread(temp)
        cv2.imshow("frame 250", img)

        gtCoords = getCoords(gtData[250])
        yoloCoords = getCoords(v3Data[250])

        for gtSet in gtCoords:
            for yoloSet in yoloCoords:
                if (gtSet["flag"] != "t" and yoloSet["flag"] != "t"):
                    iou = getIoU(gtSet["coords"], yoloSet["coords"])
                    if iou > .6:
                        gtSet["flag"] = "t"
                        yoloSet["flag"] = "t"

                        img = cv2.rectangle(img, (int(gtSet["coords"][0]), int(gtSet["coords"][1])), (int(
                            gtSet["coords"][2]), int(gtSet["coords"][3])), (0, 255, 0), 2)
                        img = cv2.rectangle(img, (int(yoloSet["coords"][0]), int(yoloSet["coords"][1])),
                                            (int(yoloSet["coords"][2]), int(yoloSet["coords"][3])), (255, 0, 0), 2)
                        cv2.imshow("tp", img)
                        cv2.waitKey(0)

        for gtSet in gtCoords:
            if gtSet["flag"] == "f":

                img = cv2.rectangle(img, (int(gtSet["coords"][0]), int(gtSet["coords"][1])),
                                    (int(gtSet["coords"][2]), int(gtSet["coords"][3])), (0, 255, 0), 2)
                cv2.imshow("fn", img)
                cv2.waitKey(0)

        for yoloSet in yoloCoords:
            if yoloSet["flag"] == "f":

                img = cv2.rectangle(img, (int(yoloSet["coords"][0]), int(yoloSet["coords"][1])),
                                    (int(yoloSet["coords"][2]), int(yoloSet["coords"][3])), (255, 0, 0), 2)
                cv2.imshow("fp", img)
                cv2.waitKey(0)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print(f"getting number of frames for {videoPath}")
        count = 0
        cap = cv2.VideoCapture(videoPath)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("img", frame)
            count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if count % 500 == 0:
                print(f"frame update - {count}")

        print(f"total frames: {count}")


if __name__ == '__main__':
    main()
