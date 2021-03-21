# USAGE
# conda activate opencv-env
# CUDA_VISIBLE_DEVICES="0" python detect_person_yolo.py --input videos/street.mp4 --output output/street_output.avi --yolo yolo-coco --time 0.3

from .deep_sort.tools import generate_detections as gdet
from .deep_sort.tracker import Tracker
from .deep_sort.detection import Detection
from .deep_sort import nn_matching
from collections import deque
from .models import Group
# import qi
import sys
# import Image
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import datetime
import calendar
# import argparse
import os

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", default="videos/street.mp4",
#                 help="path to input video")
# ap.add_argument("-o", "--output", default="output/street_output.avi",
#                 help="path to output video")
# ap.add_argument("-y", "--yolo", default="yolo-coco",
#                 help="base path to YOLO directory")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
#                 help="minimum probability to filter weak detections")
# ap.add_argument("-t", "--threshold", type=float, default=0.3,
#                 help="threshold when applying non-maxima suppression")
# ap.add_argument("-s", "--time", type=float, default=2.0,
#                 help="threshold time in minutes between people count measurements")
# ap.add_argument("--ip", type=str, default="127.0.0.1",
#                 help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
# ap.add_argument("--port", type=int, default=9559,
#                 help="Naoqi port number")
args = {"input": "pages/videos/street.mp4", "output": "pages/output/street_output.avi",
        "yolo": "pages/yolo-coco", "confidence": 0.5, "threshold": 0.3, "time": 2.0, "ip": "127.0.0.1", "port": 9559}

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize deep sort object
model_filename = 'pages/model/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric(
    "cosine", 0.5, None)
tracker = Tracker(metric)


cap = cv2.VideoCapture(args['input'])

ret, frame = cap.read()
(h, w) = frame.shape[:2]
# the output will be written to output.aviq
out = cv2.VideoWriter(
    args['output'],
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (416, 416))

ids = []
tracked_ids = []

data = [{'date': [], 'time': [], 'count': []}]
groups = [{'time': [], 'count': []}]
folder = 'pages/data/'
track_centers = {}

start_point = (70, 500)
end_point = (w-180, h-100)

people_crossed = 0
total = 0


def main_vid():
    check_data(folder+'data.csv')
    frame_count = 0
    while(cap.isOpened()):
        start_time = time.time()
        frame_count += 1

        ret, frame = cap.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif not ret:
            break

        boxes, confidence, label, features = get_detections(frame)

        detections = [Detection(bbox, score, class_name, feature) for bbox,
                      score, class_name, feature in zip(boxes, confidence, label, features)]

        frame = perform_tracking(detections, frame)

        FPS = 1.0 / (time.time() - start_time)

        record_data(FPS, frame_count)

        # cv2.imshow("image", frame)
        imge = cv2.resize(frame, (416, 416))

        # Write the output video
        out.write(imge.astype('uint8'))
    ids.sort()
    load_graph(folder+'data.csv')

    cap.release()
    # and release the output
    out.release()
    cv2.destroyAllWindows()


def main_nao():
    session = qi.Session()
    try:
        session.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) + ".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)

    """
    First get an image, then show it on the screen with PIL.
    """
    # Get the service ALVideoDevice.

    video_service = session.service("ALVideoDevice")
    resolution = 2    # VGA
    colorSpace = 11   # RGB

    videoClient = video_service.subscribe(
        "python_client", resolution, colorSpace, 5)

    # t0 = time.time()
    frame_count = 0
    # Get a camera image.
    while True:
        start_time = time.time()
        # image[6] contains the image data passed as an array of ASCII chars.
        naoImage = video_service.getImageRemote(videoClient)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # t1 = time.time()

        # Time the image transfer.
        # print ("acquisition delay ", t1 - t0)

        # Now we work with the image returned and save it as a PNG  using ImageDraw
        # package.

        # Get the image size and pixel array.
        imageWidth = naoImage[0]
        imageHeight = naoImage[1]
        frame = naoImage[6]
        image_string = str(bytearray(frame))

        boxes, confidence, label, features = get_detections(frame)

        detections = [Detection(bbox, score, class_name, feature) for bbox,
                      score, class_name, feature in zip(boxes, confidence, label, features)]

        frame = perform_tracking(detections, frame)

        FPS = 1.0 / (time.time() - start_time)

        record_data(FPS, frame_count)

        cv2.imshow("image", frame)
        imge = cv2.resize(frame, (416, 416))

        # Write the output video
        out.write(imge.astype('uint8'))

        # Create a PIL Image from our pixel array.
        im = Image.fromstring("RGB", (imageWidth, imageHeight), image_string)
        # Save the image.
        im.save(folder+"robot_image.png", "PNG")
    video_service.unsubscribe(videoClient)
    out.release()
    cv2.destroyAllWindows()
    # ids.sort()
    load_graph(folder+'data.csv')



def get_detections(frame):
    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes, confidence, label = [], [], []
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            prob = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if prob > args["confidence"] and LABELS[classID] == 'person':
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidence.append(float(prob))
                label.append(classID)

    # apply non-maxima suppression to suppress wieak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidence, args["confidence"],
                            args["threshold"])
    bbox = []
    if len(idxs) != 0:
        for i in idxs.flatten():
            bbox.append(boxes[i])

    # # ensure at least one detection exists
    # if len(idxs) > 0:
    #     # loop over the indexes we are keeping
    #     for i in idxs.flatten():
    #         # extract the bounding box coordinates
    #         (x, y) = (boxes[i][0], boxes[i][1])
    #         (w, h) = (boxes[i][2], boxes[i][3])

    # # draw a bounding box rectangle and label on the frame
    # color = [qint(c) for c in COLORS[label[i]]]
    # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    # text = "{}: {:.4f}".format(LABELS[label[i]],
    #                            confidence[i])
    # cv2.putText(frame, text, (x, y - 5),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return (np.array(bbox), np.array(confidence), np.array(label), np.array(encoder(frame, boxes)))


def perform_tracking(detections, frame):
    global start_point
    global end_point

    tracker.predict()
    tracker.update(detections)

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        class_name = track.get_class()
        bbox = track.to_tlbr().tolist()

        # track_bboxes.append(bbox.tolist()+[tracking_id, class_name])
        color = [int(c) for c in COLORS[class_name]]
        (x1, y1) = (int(bbox[0]), int(bbox[1]))
        (x2, y2) = (int(bbox[2]), int(bbox[3]))
        (cx, cy) = (x1+int((x2-x1)/2), y1+int((y2-y1)/2))

        tracking_id = track.track_id
        if tracking_id not in track_centers.keys():
            track_centers[tracking_id] = deque()
        else:
            if len(track_centers[tracking_id]) > 6:
                track_centers[tracking_id].pop()
        track_centers[tracking_id].append((cx, cy))
        if tracking_id not in ids:
            ids.append(tracking_id)
        if check_y(track_centers[tracking_id][-1][0]) < track_centers[tracking_id][-1][1] and check_y(track_centers[tracking_id][0][0]) > track_centers[tracking_id][0][1] and tracking_id not in tracked_ids:
            tracked_ids.append(tracking_id)

        # cv2.rectangle(frame, (x1, y1), (x2, y2), color)
        cv2.circle(frame, (cx, cy), 5, color, 2)
        cv2.putText(frame, str(tracking_id), (x1, y1+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, "Total:" + str(len(ids)), (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, "Current:" + str(len(tracked_ids)), (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.line(frame, start_point, end_point, color, 1)
    return frame


def check_y(x):
    global start_point
    global end_point
    grad = (end_point[1]-start_point[1])/(end_point[0]-start_point[0])
    if x is not None:
        return int(grad * x + start_point[1])
    return None


def findDay(date):
    day = datetime.datetime.strptime(date, '%Y:%m:%d').weekday()
    return calendar.day_name[day]


def load_graph(path):
    data = pd.read_csv(path)
    times = {}
    avg_days = {}
    days = ['Monday', 'Tuesday', 'Wednesday',
            'Thursday', 'Friday', 'Saturday', 'Sunday']

    for day in days:
        avg_days[day] = {}
        for j in range(24):
            if j < 10:
                avg_days[day]['0' + str(j)] = []
            else:
                avg_days[day][str(j)] = []

    for dat in data['date'].unique():

        day = findDay(str(dat))
        for i in range(len(data)):

            hr = str(data.iloc[i]['time'][0:2])
            if str(data.iloc[i]['date']) == str(dat):
                avg_days[day][hr].append(
                    data.iloc[i]['count'])
    for dat in data['date'].unique():
        day = findDay(str(dat))
        for time in avg_days[day].keys():
            if isinstance(avg_days[day][time], list):
                if len(avg_days[day][time]) > 0:
                    avg_days[day][time] = sum(
                        avg_days[day][time])/len(avg_days[day][time])
                else:
                    avg_days[day][time] = 0
    for dat in data['date'].unique():
        print(dat)
        day = findDay(str(dat))
        fig, ax = plt.subplots()
        ax.bar(avg_days[day].keys(),
               avg_days[day].values())
        ax.tick_params(axis='x', which='major', labelsize=5)
        ax.tick_params(axis='y', which='minor', labelsize=8)
        plt.suptitle(day+' Visitor Flow')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of People Detected')
        plt.savefig(folder+day+'_plot.pdf', bbox_inches='tight')


def check_data(path):
    try:
        with open(path, 'r') as csv_file:
            csv_dict = [row for row in csv.DictReader(csv_file)]
        if len(csv_dict) == 0:
            print('CSV file is empty')
            with open(folder+'data.csv', 'w') as csv_file:
                fieldnames = ['date', 'time', 'count']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
    except:
        print('File not loaded')
        with open(path, 'w') as csv_file:
            fieldnames = ['date', 'time', 'count']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()


def record_data(FPS, frame_count):
    global people_crossed
    global total
    global tracked_ids
    global groups
    if frame_count % int((args["time"]*60)/(4*FPS)) == 0:
        print("FPS: ", FPS)
        people_crossed += len(tracked_ids)
        if people_crossed != 0:
            print('{}:{}'.format(datetime.datetime.now().strftime(
                "%H:%M:%S"), len(tracked_ids)))
            group = Group(people=people_crossed)
            group.save()
            groups.append(
                {'count': people_crossed, 'time': datetime.datetime.now().strftime("%H:%M:%S")})
            people_crossed = 0
            tracked_ids = []

    if frame_count % int((args["time"]*60)/FPS) == 0:
        print("total FPS: ", FPS)
        data = pd.read_csv(folder+'data.csv')
        total = len(ids) - total
        data = data.append(
            {'date': datetime.datetime.now().strftime('%Y:%m:%d'), 'time': datetime.datetime.now().strftime(
                "%H:%M:%S"), 'count': total}, ignore_index=True)
        with open(folder+'data.csv', 'w') as csv_file:
            fieldnames = ['date', 'time', 'count']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(data)):
                writer.writerow(
                    {'date': data['date'].iloc[i], 'time': data['time'].iloc[i], 'count': data['count'].iloc[i]})
        total = len(ids)


def get_groups():
    global groups
    return groups


if __name__ == '__main__':
    main_vid()
    # main_nao()
    load_graph(folder+'data.csv')
