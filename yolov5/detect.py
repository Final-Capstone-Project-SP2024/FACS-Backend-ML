import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from numpy import random
import datetime as dt
import threading

class CameraThread(threading.Thread):
    def __init__(self, opt, source):
        threading.Thread.__init__(self)
        self.opt = argparse.Namespace(**vars(opt))
        self.source = source
        print(f"Initializing thread with source {self.source}")

    def run(self):
        print(f"Starting detection and recording for camera {self.source}")
        detect_and_record(self.source, self.opt)
        print(f"Thread for camera {self.source} has finished")

def detect_and_record(src, opt, save_img=False):
    out, weights, view_img, save_txt, imgsz, record_folder = \
        opt.output, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.record_folder
    webcam = src.isdigit()  # Check if the source is a digit (webcam)

    # Initialize
    device = select_device(opt.device)
    # if os.path.exists(out):
    #     shutil.rmtree(out)  # delete output folder
    os.makedirs(out, exist_ok=True)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if str(src).isdigit(): # 0, 1, 2, etc. for webcam
        view_img = True
        dataset = LoadStreams(int(src), img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(src, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Initialize recording variables
    recording = False
    start_time = 0
    record_duration = 10  # in seconds
    pre_fire_duration = 5  # in seconds

    # Initialize fire detection counter
    total_frames = 0
    fire_frames = 0

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # We're indexing cameras by number
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                window_name = f"camera_{src}"  # Static window name for each camera
                save_path = str(Path(out) / f"{window_name}.jpg")
                txt_path = str(Path(out) / window_name)
                cv2.imshow(window_name, im0)  # Use static window name
            else:
                p, s, im0 = path, '', im0s
                # Ensure p is a string representation of the path for non-webcam sources
                p_str = str(p)
                save_path = str(Path(out) / Path(p_str).name)  # For file paths
                txt_path = str(Path(out) / Path(p_str).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                # Convert p to string in case it is not already
                cv2.imshow(p_str, im0)

            # save_path = str(Path(out) / Path(p).name)
            # txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            # Increment total frames counter
            total_frames += 1

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Recording logic
                if any(c == 0 for c in det[:, -1].unique()):  # Check if class 0 (fire) is detected
                    if not recording:
                        recording = True
                        start_time = time.time()
                        print("Fire detected! Recording...")

                        # Set up video writer for recording
                        if vid_writer is not None:
                            vid_writer.release()
                        timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        record_name = f'record_{timestamp}.t'
                        record_path = os.path.join(record_folder, record_name)
                        size = (im0.shape[1], im0.shape[0])
                        vid_writer = cv2.VideoWriter(record_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, size)

                        # Increment fire frames counter
                        fire_frames += 1

                        # Set start time for pre-fire duration
                        pre_fire_start_time = start_time

                # Write results
                for *xyxy, conf, cls in det:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    # Add bbox to image
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                # If recording, write frame to the video file
                if recording:
                    if time.time() - pre_fire_start_time < pre_fire_duration:
                        # Save frames only for pre-fire duration
                        vid_writer.write(im0)

                    # Check if 10 seconds have elapsed since the start of recording
                    elif time.time() - start_time >= record_duration:
                        recording = False
                        vid_writer.release()
                        print("Recording stopped. Duration:", time.time() - start_time)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path

                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))

        # Check if 10 seconds have elapsed since the start of recording
        if recording and time.time() - start_time >= record_duration:
            recording = False
            vid_writer.release()
            print("Recording stopped. Duration:", time.time() - start_time)

    # Release video writer when done
    if vid_writer is not None:
        vid_writer.release()

    # Calculate and print the percentage of frames with fire
    if total_frames > 0:
        percentage_fire = (fire_frames / total_frames) * 100 * 100
        print(f"Percentage of frames with fire: {percentage_fire:.2f}%")

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform.system() == 'Darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


# ...

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    # parser.add_argument('--source', nargs='+', type=str, help='source')  # multiple file/folders, 0 for webcam
    parser.add_argument('--source', nargs='+', type=str, default=['0', '1'], help='sources list: 0 for internal camera, 1 for external, or path to video files/stream URLs')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--record-folder', type=str, default='inference/records', help='folder to save recorded videos')
    # Parse the arguments and make sure the record folder exists
    opt = parser.parse_args()
    os.makedirs(opt.record_folder, exist_ok=True)

    # Start the camera threads
    threads = []
    for source in opt.source:
        thread = CameraThread(opt, source)
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()