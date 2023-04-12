import torch
import numpy as np
import cv2
import time


class V5Detection:

    def __init__(self, videoPath):
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cpu'
        self.videoPath = videoPath
        print("Model loaded")

    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5',
                               'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        numPeople = 0
        tempStr = ""
        for i in range(n):
            # only showing people
            label = self.class_to_label(labels[i])
            if label == "person":
                numPeople += 1
                row = cord[i]
                if row[4] >= 0.2:
                    x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                        row[3] * y_shape)
                    tempStr = tempStr + f"{x1},{y1},{x2},{y2},|"
                    bgr = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                    cv2.putText(frame, label, (x1, y1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        finalStr = str(numPeople) + "|" + tempStr + "\n"
        return frame, finalStr

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        t0 = time.time()
        filePath = f"output/v5/{self.videoPath[6:-4]}.txt"
        file = open(filePath, "w")
        cap = cv2.VideoCapture(self.videoPath)
        frameNum = 0
        avgFps = 0
        while cap.isOpened():
            if frameNum == 500:
                t0 = time.time()
                avgFps = 0
            start_time = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break
            results = self.score_frame(frame)
            frame, frameData = self.plot_boxes(results, frame)
            end_time = time.perf_counter()
            fps = 1 / np.round(end_time - start_time, 3)
            avgFps += fps
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow("img", frame)
            frameResult = f"frame{frameNum}|" + frameData
            file.write(frameResult)
            frameNum += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if frameNum == 1000:
                break

        file.close()
        t1 = time.time() - t0
        print(f"Time to run - {t1}")
        print(f"Average fps - {avgFps / 500}")
