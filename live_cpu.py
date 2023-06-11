import argparse
import numpy as np
import cv2
import time
from datetime import datetime as dt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from utils import select_device, draw_gaze
from PIL import Image, ImageOps

from face_detection import RetinaFace
# from model import ML2CS180
from model import VRI_GazeNet
import os

import pickle

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--gpu',dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.', 
        default='models/ML2CS.pkl', type=str)
        
    parser.add_argument(
        '--video_source',dest='video_filename', help='Video to be captioned',
        default=None, type=str)
    parser.add_argument(
        '--video_output',dest='video_output', help='Video file output',
        default=None, type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True

    video_filename = args.video_filename

    batch_size = 1
    # cam = args.cam_id
    gpu = select_device(args.gpu_id, batch_size=batch_size)
    snapshot_path = args.snapshot

    transformations = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # model = ML2CS180()
    model = VRI_GazeNet(num_bins=181)
    print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path, map_location=torch.device('cpu'))
    model.load_state_dict(saved_state_dict)
    model.cpu()
    model.eval()

    softmax = nn.Softmax(dim=1)
    detector = RetinaFace(gpu_id=-1)

    # idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    idx_tensor = [idx for idx in range(model.num_bins)]
    idx_tensor = torch.FloatTensor(idx_tensor).cpu()


    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open video")

    # print('Processing video...')
    x_list = []
    y_list = []

    knn = pickle.load(open("knn.pkl", "rb"))

    with torch.no_grad():
        last_sec = dt.now().second
        retval, frame = cap.read() 
        frame_count = 0
        while retval:
            sec = dt.now().second
            if sec == last_sec:
                frame_count += 1
            else:
                last_sec = sec
                frame_count = 0

            faces = detector(frame)
            if faces is not None: 
                for box, landmarks, score in faces:
                    if score < .98:
                        continue
                    x_min=int(box[0])
                    if x_min < 0:
                        x_min = 0
                    y_min=int(box[1])
                    if y_min < 0:
                        y_min = 0
                    x_max=int(box[2])
                    y_max=int(box[3])
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.resize(img, (224, 224))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    img=transformations(im_pil)
                    # img  = Variable(img).cuda(gpu)
                    img  = Variable(img).cpu()
                    img  = img.unsqueeze(0) 
                    
                    # gaze prediction
                    gazes = model.angles(img)
                    yaw, pitch = gazes[0]
                    # print(yaw, pitch)
                    
                    yaw_predicted= yaw * np.pi/180.0
                    pitch_predicted= pitch * np.pi/180.0
                    
                    draw_gaze(x_min,y_min,bbox_width, bbox_height,frame,(yaw_predicted,pitch_predicted),color=(185, 240, 113), scale=1, thickness=4, size=x_max-x_min, bbox=((x_min, y_min), (x_max, y_max)))
                    cv2.putText(frame,str(frame_count),(10, 30),cv2.FONT_HERSHEY_PLAIN, 2,(100, 200, 150),2) 
                    cv2.putText(frame,f"{yaw, pitch}",(10, 50),cv2.FONT_HERSHEY_PLAIN, 2,(100, 200, 150),2) 
                    cv2.imshow("FRAME", frame)

                    pressed = cv2.waitKey(33) % 256
                    if chr(pressed) != "ÿ" and pressed != 27:
                        should_calculate = True
                        print(f"Pressed {chr(pressed)}, on {yaw,pitch}")
                        x_list.append((yaw,pitch,(x_min + x_max)//2, (y_min+y_max)//2))
                        y_list.append(chr(pressed))

                        array = np.array((yaw,pitch,(x_min + x_max)//2, (y_min+y_max)//2))
                        array = np.array([array])
                        print(knn.predict(array))


                    
                    elif pressed == 27:
                        x = np.array(x_list)
                        np.save("X_ARRAY.npy", x)

                        y = np.array(y_list)
                        np.save("y_ARRAY.npy", y)
                        print("SAVED.")
                        exit()


            retval, frame = cap.read() 