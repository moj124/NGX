# USAGE
# conda activate opencv-env
# CUDA_VISIBLE_DEVICES="0" python3 detect_person_yolov5.py --input videos/street.mp4  --time 5.0


import argparse
import time
import numpy as np
# from numpy.compat.py3k import open_latin1
import pandas as pd
import matplotlib.pyplot as plt
import csv
import datetime
import calendar
import sys
import math

from pathlib import Path

# import qi
# import Image
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from .deep_sort.tools import generate_detections as gdet
from .deep_sort.tracker import Tracker
from .deep_sort.detection import Detection
from .deep_sort import nn_matching
from collections import deque
from .models import Group


class People_Counter:
    """
    A class used to represent an people detector and tracking algorithm

    Attributes
    ----------
    opt : dictionary
        contains specified parameters for paths of files, inner-paramters of algorithm and output/input paths
    ids : integer
        the array of all detected people ids
    tracked_ids : int
        an array keeps track of already discovered people ids
    data : array of dictionary
        an array that records the time and count of people in the sample
    groups : array of dictionary
        an array that records the time and count of the people in the sampled group
    folder : str
        a string determining the recording path of data
    static_path : str
        a string specifying the output path for the visitor flow graphs
    track_centers : dictionary
        a dictionary that records a series of centriods coordinates of each person detected
    people_crossed : int
        a integer that determines the current number of people who crossed the line trigger
    total : int
        a integer that measures the total number of detected people
    encoder : Encoder
        a encoder object that aggregates the current image with the box coordinates of detections
    tracker : Tracker
        a tracker that performs the tracking of the coordinates of the detected people

    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makes

    main_vid()
        runs the whole detection and tracking algorithm given a video or image streams

    main_nao()
        runs the whole detection and tracking algoirthm using NAOqi's API to retrieve image streams

    perform_tracking(detections, frame)
        given a set of detections and frame image returns the annotated frame, while managing the ids and tracked_ids

    check_y(x)
        given x will return a specified y value of the line otherwise returns None

    findDay(date)
        given a date will return the corresponding day of the week

    load_graph(path)
        given a path determine load the data from path and then update the data with new time-sensitive metric data and save to output paths

    check_data(path)
        given a path will check if the file exists and correctly formats the file for use given its not found or already exists

    record_data(FPS,frame_count)
        given the current FPS rate and frame count of the data stream, sample the current metrics and update the data file, whilst adding group object to webpage

    """

    def __init__(self, opt):
        """
        Parameters
        ----------
        opt : dictionary
            contains specified parameters for paths of files, inner-paramters of algorithm and output/input paths
        """

        # check that all dependencies are loaded
        check_requirements()

        # setup models for tracking
        self.opt = opt
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.5, None)
        self.encoder = gdet.create_box_encoder(
            self.opt['encoder_model'], batch_size=1)
        self.tracker = Tracker(metric)

        # assign parameters to the attributes
        self.ids = []
        self.tracked_ids = []
        self.data = [{'date': [], 'time': [], 'count': []}]
        self.groups = [{'time': [], 'count': []}]
        self.track_centers = {}
        self.people_crossed = 0
        self.cumulative_count = 0
        self.counted = []
        self.total = 0

        # set path for data and output
        self.folder = 'data/'
        self.static_path = 'pages/static/'

    def main_vid(self):
        """Perfroms the detection and tracking of people in the data streams

        All models for tracking and detection are loaded, inference is run on the data stream
        and detections are inputted into the tracker. The tracker manages the ids to detections
        and data is recorded and saved to data folder.
        """

        # check and format data file
        self.check_data('data/'+'data.csv')

        # initialise inner parameters
        frame_count = 0
        source, weights,  view_img, save_txt, imgsz = self.opt['source'], self.opt[
            'weights'], self.opt['view-img'], self.opt['save-txt'], self.opt['img-size']
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://'))

        # Directories
        save_dir = Path(increment_path(Path(self.opt['project']) / self.opt['name'],
                                       exist_ok=self.opt['exist_ok']))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                              exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(self.opt['device'])
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load(
                'weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            save_img = True
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
                next(model.parameters())))  # run once
        t0 = time.time()

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=self.opt['augment'])[0]

            # Apply NMS
            pred = non_max_suppression(
                pred, self.opt['conf'], self.opt['iou'], classes=self.opt['classes'], agnostic=self.opt['agnostic-nms'])
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)
                pred1 = apply_classifier(pred1, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                start_time = time.time()
                frame_count += 1

                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
                    ), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(
                        dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                # img.txt
                txt_path = str(save_dir / 'labels' / p.stem) + \
                    ('' if dataset.mode == 'image' else f'_{frame}')
                s += '%gx%g ' % img.shape[2:]  # print string
                # normalization gain whwh
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                det = torch.tensor([list(map(float, a))
                                    for a in det if names[int(a[-1])] == "person"])
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        # add to string
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                    # Write results to arrays
                    box = []
                    ccls = []
                    confd = []
                    for *xyxy, conf, cls in reversed(det):
                        dims = list(map(int, xyxy))

                        # add each attribut from detection onto the arrays
                        box.append([dims[0], dims[1], dims[2] -
                                    dims[0], dims[3]-dims[1]])
                        ccls.append(int(cls))
                        confd.append(float(conf))

                        # print("Data", list(map(int, xyxy)),
                        #       float(conf), names[int(cls)])

                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                              ) / gn).view(-1).tolist()  # normalized xywh
                            # label format
                            line = (
                                cls, *xywh, conf) if self.opt['save-conf'] else (cls, *xywh)
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() %
                                        line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label,
                                         color=colors[int(cls)], line_thickness=3)

                # Print time (inference + NMS)

                print(f'{s}Done. ({t2 - t1:.3f}s)')

                # retrieve the attributes of each detection from arrays
                boxes, confidence, label, features = (np.array(box),
                                                      np.array(
                    confd),
                    np.array(ccls),
                    np.array(self.encoder(im0, box)))

                # print(boxes)

                # compile detection attributes into detection objects within array
                detections = [Detection(bbox, score, class_name, feature) for bbox,
                              score, class_name, feature in zip(boxes, confidence, label, features)]

                # perform tracking and retrieve annotated image frame for viewing
                frame = self.perform_tracking(detections, im0)

                # measure FPS of each frame
                FPS = 1.0 / (time.time() - start_time)

                # based off FPS and frame_count sample accordingly
                self.record_data(FPS, frame_count)

                # Stream results
                if view_img:
                    cv2.imshow(str(p), frame)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fourcc = 'mp4v'  # output video codec
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(
                                save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        vid_writer.write(im0)

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')
        # self.ids.sort()

        # save graph plots
        self.load_graph(self.folder+'data.csv')

    def main_nao(self):
        """Perfroms the detection and tracking of people from the NAO robot

        All models for tracking and detection are loaded, inference is run on the extracted image from NAO robot
        and detections are inputted into the tracker. The tracker manages the ids to detections
        and data is recorded and saved to data folder.
        """

        source, weights, view_img, save_txt, imgsz = self.opt['source'], self.opt[
            'weights'], self.opt['view-img'], self.opt['save-txt'], self.opt['img-size']
        session = qi.Session()

        # Directories
        save_dir = Path(increment_path(Path(self.opt['project']) / self.opt['name'],
                                       exist_ok=self.opt['exist_ok']))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                              exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(self.opt['device'])
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()  # to FP16

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        txt_path = str(save_dir / 'labels' / p.stem) + \
            ('' if dataset.mode == 'image' else f'_{frame}')

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
                next(model.parameters())))  # run once
        t0 = time.time()

        try:
            session.connect(
                "tcp://" + self.opt['ip'] + ":" + str(self.opt['port']))
        except RuntimeError:
            print("Can't connect to Naoqi at ip \"" + self.opt['ip'] + "\" on port " + str(self.opt['port']) + ".\n"
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

            # Get the image size and pixel array.
            imageWidth = naoImage[0]
            imageHeight = naoImage[1]
            frame = naoImage[6]
            image_string = str(bytearray(frame))

            img = torch.from_numpy(frame).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=self.opt['augment'])[0]

            # Apply NMS
            pred = non_max_suppression(
                pred, self.opt['conf'], self.opt['iou'], classes=self.opt['classes'], agnostic=self.opt['agnostic-nms'])
            t2 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                start_time = time.time()
                frame_count += 1

                gn = torch.tensor(img.shape)[[1, 0, 1, 0]]
                det = torch.tensor([list(map(float, a))
                                    for a in det if names[int(a[-1])] == "person"])
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], img.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        # add to string
                        # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                    # Write results to arrays
                    box = []
                    ccls = []
                    confd = []
                    for *xyxy, conf, cls in reversed(det):
                        dims = list(map(int, xyxy))

                        # add detection attributes to array
                        box.append([dims[0], dims[1], dims[2] -
                                    dims[0], dims[3]-dims[1]])
                        ccls.append(int(cls))
                        confd.append(float(conf))

                        # print("Data", list(map(int, xyxy)),
                        #       float(conf), names[int(cls)])

                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                              ) / gn).view(-1).tolist()  # normalized xywh
                            # label format
                            line = (
                                cls, *xywh, conf) if self.opt['save-conf'] else (cls, *xywh)
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() %
                                        line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, img, label=label,
                                         color=colors[int(cls)], line_thickness=3)

                # Print time (inference + NMS)

                print(f'{s}Done. ({t2 - t1:.3f}s)')

            # compile attributes of each detection
            boxes, confidence, label, features = (np.array(box),
                                                  np.array(
                confd),
                np.array(ccls),
                np.array(self.encoder(img, box)))

            # retrieve the detection objects from the arrays of detection attributes
            detections = [Detection(bbox, score, class_name, feature) for bbox,
                          score, class_name, feature in zip(boxes, confidence, label, features)]

            # perform tracking and return annotated frame image
            frame = self.perform_tracking(detections, frame)

            # measure the FPS of processing each of the image frames
            FPS = 1.0 / (time.time() - start_time)

            # based on FPS and frame_count sample the people metrics
            self.record_data(FPS, frame_count)

            # view image
            cv2.imshow("image", frame)

            # Create a PIL Image from our pixel array.
            im = Image.fromstring(
                "RGB", (imageWidth, imageHeight), image_string)
            # Save the image.
            im.save(self.folder+"robot_image.png", "PNG")

        # release subscription
        video_service.unsubscribe(videoClient)
        cv2.destroyAllWindows()
        # self.ids.sort()

        # save graph plots to output paths
        self.load_graph(self.folder+'data.csv')

    def perform_tracking(self, detections, frame):
        """Manages the tracking and assignment of ids to each detection and return the annotated frame image

        Parameters
        ----------
        detections : array of Detections
            An array of detections for each detected person

        frame : image
            An image from the data stream given current iteration
        """

        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(1, 3),
                                   dtype="uint8")

        # perfrom tracking step
        self.tracker.predict()
        self.tracker.update(detections)

        for track in self.tracker.tracks:  # per tracked object

            # given track is valid
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            # retrieve bounding box and class label
            class_name = track.get_class()
            bbox = track.to_tlbr().tolist()

            # track_bboxes.append(bbox.tolist()+[tracking_id, class_name])

            # calculate positions and color for annotation purposes
            color = [int(c) for c in COLORS[class_name]]
            (x1, y1) = (int(bbox[0]), int(bbox[1]))
            (x2, y2) = (int(bbox[2]), int(bbox[3]))

            # used for event line trigger
            (cx, cy) = (x1+int((x2-x1)/2), y1+int((y2-y1)/2))

            # add new ids and track the past 4 centroid position of the object
            tracking_id = track.track_id
            if tracking_id not in self.track_centers.keys():
                self.track_centers[tracking_id] = deque()
            else:
                if len(self.track_centers[tracking_id]) > 4:
                    self.track_centers[tracking_id].pop()

            # add most recent centroid to the array
            self.track_centers[tracking_id].append((cx, cy))

            # add new ids to list of encountered ids
            if tracking_id not in self.ids:
                self.ids.append(tracking_id)

            # directional based line event trigger
            # check if person has crossed the line and is not seen before

            if tracking_id not in self.counted and tracking_id not in self.tracked_ids:
                if self.opt['axis'].lower() == 'horizontal':
                    # PERSON IS SEEN BELOW THE LINE #################################
                    if self.opt['line-side'].lower() == 'left':
                        if self.check_y(self.track_centers[tracking_id][-1][0]) < self.track_centers[tracking_id][-1][1] and self.check_y(self.track_centers[tracking_id][0][0]) > self.track_centers[tracking_id][0][1]:
                            self.tracked_ids.append(tracking_id)
                            self.people_crossed += 1

                    # PERSON IS SEEN ABOVE THE LINE #################################
                    elif self.opt['line-side'].lower() == 'right':
                        if self.check_y(self.track_centers[tracking_id][-1][0]) > self.track_centers[tracking_id][-1][1] and self.check_y(self.track_centers[tracking_id][0][0]) < self.track_centers[tracking_id][0][1]:
                            self.tracked_ids.append(tracking_id)
                            self.people_crossed += 1
                elif self.opt['axis'].lower() == 'vertical':
                    # PERSON IS SEEN BELOW THE LINE #################################
                    if self.opt['line-side'].lower() == 'left':
                        if self.opt['start'][0] < self.track_centers[tracking_id][-1][0] and self.opt['start'][0] > self.track_centers[tracking_id][0][0]:
                            self.tracked_ids.append(tracking_id)
                            self.people_crossed += 1
                    # PERSON IS SEEN ABOVE THE LINE #################################
                    elif self.opt['line-side'].lower() == 'right':
                        if self.opt['start'][0] > self.track_centers[tracking_id][-1][0] and self.opt['start'][0] < self.track_centers[tracking_id][0][0]:
                            self.tracked_ids.append(tracking_id)
                            self.people_crossed += 1

                    # cv2.rectangle(frame, (x1, y1), (x2, y2), color)

                    # annotated frame image to view tracking elements
            cv2.circle(frame, (cx, cy), 5, color, 2)
            cv2.putText(frame, str(tracking_id), (x1, y1+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, "Total:" + str(len(self.ids)), (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, "Current:" + str(self.tracked_ids), (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.line(frame, self.opt['start'], self.opt['end'], color, 1)

        return frame

    def check_x(self, y):
        """calculates the y-value of a given x value based off the gradient of the line, otherwise return None if x is empty

        Parameters
        ----------
        x : float
            a given x-value

        Raises
        ------
        None
            If no x-value is set.
        """

        # gradient of the event line trigger
        if self.opt['start'][0] == self.opt['end'][0]:
            grad = 0
        else:
            grad = (self.opt['end'][1]-self.opt['start'][1]) / \
                (self.opt['end'][0]-self.opt['start'][0])

        # where x is a valid value return the corresponding y-value
        if y is not None:
            if grad == 0:
                return int(self.opt['start'][0])
            return int((y - self.opt['start'][1])/grad)

        # return None if input is not valid
        return None

    def check_y(self, x):
        """calculates the y-value of a given x value based off the gradient of the line, otherwise return None if x is empty

        Parameters
        ----------
        x : float
            a given x-value

        Raises
        ------
        None
            If no x-value is set.
        """

        # gradient of the event line trigger
        if self.opt['start'][0] == self.opt['end'][0]:
            grad = 0
        else:
            grad = (self.opt['end'][1]-self.opt['start'][1]) / \
                (self.opt['end'][0]-self.opt['start'][0])

        # where x is a valid value return the corresponding y-value
        if x is not None:
            return int(grad * x + self.opt['start'][1])

        # return None if input is not valid
        return None

    def findDay(self, date):
        """retrieves the day of the week given a date

        Parameters
        ----------
        date : str
            An string given in Y:m:d specifying the date
        """

        # retrieve the int day of the week
        day = datetime.datetime.strptime(date, '%Y:%m:%d').weekday()

        # return the day name of the week
        return calendar.day_name[day]

    def load_graph(self, path):
        """load and update the graph plots of visitor flows

        Parameters
        ----------
        path : string
            An string of the output path to save and load graph plots

        """

        # load data from path
        self.data = pd.read_csv(path)

        # setup dictionary with each day each hour
        avg_days = {}
        days = ['Monday', 'Tuesday', 'Wednesday',
                'Thursday', 'Friday', 'Saturday', 'Sunday']

        for day in days:  # every day
            avg_days[day] = {}

            for j in range(24):  # every hour
                if j < 10:
                    avg_days[day]['0' + str(j)] = []
                else:
                    avg_days[day][str(j)] = []

        # retrieve each record from data and add the count to the respective day and hour of the data dictionary
        for dat in self.data['date'].unique():
            day = self.findDay(str(dat))
            for i in range(len(self.data)):

                hr = str(self.data.iloc[i]['time'][0:2])

                # where date matches each other
                if str(self.data.iloc[i]['date']) == str(dat):
                    avg_days[day][hr].append(
                        self.data.iloc[i]['count'])

        # append the average of the counts from data dictionary
        for dat in self.data['date'].unique():
            day = self.findDay(str(dat))
            for time in avg_days[day].keys():

                # where data is in correct format
                if isinstance(avg_days[day][time], list):
                    if len(avg_days[day][time]) > 0:
                        avg_days[day][time] = sum(
                            avg_days[day][time])/len(avg_days[day][time])
                    else:
                        avg_days[day][time] = 0

        # for each day create a plot of average visitor flow
        for dat in self.data['date'].unique():
            print(dat)
            day = self.findDay(str(dat))
            fig, ax = plt.subplots()
            ax.bar(avg_days[day].keys(),
                   avg_days[day].values())
            ax.tick_params(axis='x', which='major', labelsize=5)
            ax.tick_params(axis='y', which='minor', labelsize=8)
            plt.suptitle(day+' Visitor Flow')
            plt.xlabel('Hour of Day')
            plt.ylabel('Average Occurence of People')

            # save to the respective folder paths
            plt.savefig(self.static_path+day+'_plot.pdf', bbox_inches='tight')
            plt.savefig(self.folder+day+'_plot.pdf', bbox_inches='tight')
            plt.savefig(self.static_path+day+'_plot.png', bbox_inches='tight')

    def check_data(self, path):
        """determine if the data file is correctly formated otherwise overwrite with a new file

        Parameters
        ----------
        path : string
            An string of the output path to save and load data file

        """

        # try to access the data csv file
        try:

            # open file
            with open(path, 'r') as csv_file:
                csv_dict = [row for row in csv.DictReader(csv_file)]

            # check if file is empty
            if len(csv_dict) == 0:
                print('CSV file is empty')

                # create new file
                with open(self.folder+'data.csv', 'w') as csv_file:
                    fieldnames = ['date', 'time', 'count']
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
        except:
            # when error occurs write a new file to path
            print('File not loaded')
            with open(path, 'w') as csv_file:
                fieldnames = ['date', 'time', 'count']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

    def record_data(self, FPS, frame_count):
        """update and manage people metrics within the data file, whilst add new group objects to the webpage

        Parameters
        ----------
        FPS : float
            An float that measures the number of image frames processed per second

        frame_count : int
            an integer that determines the current number of images processed

        """

        # sample group data based on the time parameter and FPS rate 4 times per total data
        print( 'tESTINHG',math.ceil((self.opt["time"]*60)/(4*FPS)), frame_count)
        if (frame_count % math.ceil((self.opt["time"]*60)/(4*FPS))) == 0:
            print("FPS: ", FPS)
            # self.people_crossed = len(self.tracked_ids)

            # where there are people who have crossed the event line trigger
            if self.people_crossed != 0:
                print('{}:{}'.format(datetime.datetime.now().strftime(
                    "%H:%M:%S"), len(self.tracked_ids)))

                # append new group count to the webpage database
                group = Group(people=self.people_crossed)
                group.save()

                # update the groups attribute with new additions
                self.groups.append(
                    {'count': self.people_crossed, 'time': datetime.datetime.now().strftime("%H:%M:%S")})

                # reset the attributes
                self.people_crossed = 0
                self.counted = list(set(self.counted + self.tracked_ids))
                self.tracked_ids = []

        # sample total data based on time and FPS parameters
        if (frame_count % math.ceil((self.opt["time"]*60)/FPS)) == 0:
            print("total FPS: ", FPS)

            # read csv and update with new total
            self.data = pd.read_csv(self.folder+'data.csv')

            # get cumulative total difference
            self.total = len(self.ids) - self.total
            self.data = self.data.append(
                {'date': datetime.datetime.now().strftime('%Y:%m:%d'), 'time': datetime.datetime.now().strftime(
                    "%H:%M:%S"), 'count': self.total}, ignore_index=True)

            # write to the data path
            with open(self.folder+'data.csv', 'w') as csv_file:
                fieldnames = ['date', 'time', 'count']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for i in range(len(self.data)):
                    writer.writerow(
                        {'date': self.data['date'].iloc[i], 'time': self.data['time'].iloc[i], 'count': self.data['count'].iloc[i]})

            # update new total attribute
            self.total = len(self.ids)


def start(start, end, side, axis, source):
    """load a people counter and run the algorithm"""

    # parameters
    opt = {'start': start, 'end': end, 'port': 9559, 'ip': '127.0.0.1', 'time': 5.0, 'weights': 'yolov5s.pt',
           'source': source, 'conf': 0.6, 'iou': 0.6, 'view-img': True, 'save-txt': False, 'img-size': 640,
           'project': 'runs/detect', 'name': 'exp', 'exist_ok': False, 'agnostic-nms': False, 'classes': None, 'save-conf': True, 'device': '',
           'augment': False, 'encoder_model': 'pages/model/mars-small128.pb', 'line-side': side, 'axis': axis}

    # setup people_counter
    counter = People_Counter(opt)

    # run people_counter
    counter.main_vid()

    # update graph plots
    counter.load_graph(counter.folder+'data.csv')


def warning(message):
    """
    Uses the Dictionary TTS methods (English) to say a message from NAO robot

    Parameters
    ----------
    message : str
        An string that contains the message fro the NAO robot to say

    Raises
    ------
    RuntimeError
        If can't create session with Naoqi using IP and port.


    """
    args = {'ip': "127.0.0.1", 'port': 9559}

    session = qi.Session()
    try:
        session.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) + ".\n"
              "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)

    # Get the service ALTextToSpeech.
    tts = session.service("ALTextToSpeech")

    # set language to English
    tts.setLanguage("English")

    # Say message in English
    tts.say(message)


def p(x):
    """
    return the constraint line 

    Parameters
    ----------
    x : int
        An integer used to calcualte the y-value of the line
    """
    p_max = 1

    return (p_max / 10) * x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default=[13, 217], type=int, nargs=2,
                        help='start point coordinates from image for line event trigger')
    parser.add_argument('--end', default=[376, 405], type=int, nargs=2,
                        help='end point coordinates from image for line event trigger')
    parser.add_argument('--port', type=int, default=9559,
                        help='port of the NAO robot setup for communication channel.')
    parser.add_argument('--ip', type=str, default="127.0.0.1",
                        help='the ip address of the NAO robot')
    parser.add_argument('--time', type=float, default=2.0,
                        help='time interval of sampling occurences of people.')
    parser.add_argument('--weights', nargs='+', type=str,
                        default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--mask', type=str, default='best.pt',
                        help='mask model.pt path')
    parser.add_argument('--encoder_model', type=str,
                        default='pages/model/mars-small128.pb', help='encoder model path.')
    parser.add_argument('--line-side', type=str,
                        default='left', help='determine the direction of detection for the line.')
    parser.add_argument('--axis', type=str,
                        default='horizontal', help='determine the axis of line trigger.')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='data/images/street.mp4', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=True, action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    # opt = {'start': (13, 217), 'end': (376, 405), 'port': 9559, 'ip': '127.0.0.1', 'time': 2.0, 'weights': 'yolov5s.pt',
    #        'source': 'data/images/street.mp4', 'conf': 0.25, 'iou': 0.45, 'view-img': True, 'save-txt': False, 'img-size': 640,
    #        'project': 'runs/detect', 'name': 'exp', 'exist_ok': False, 'agnostic-nms': True, 'classes': None, 'save-conf': False,
    #        'device': '', 'augment': False,  'line-side':'right','axis': 'horizontal'}

    counter = People_Counter(opt)
    counter.main_vid()
    counter.load_graph(counter.folder+'data.csv')
