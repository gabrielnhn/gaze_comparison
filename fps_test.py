import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

import datasets
from utils import select_device, natural_keys, gazeto3d, angular, draw_gaze
from model import L2CS, VRI_GazeNet

from fvcore.nn import FlopCountAnalysis
import typing
import datetime
import cv2
from face_detection import RetinaFace
from torchvision.transforms import ToPILImage


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze estimation using L2CSNet .')
     # Gaze360
    parser.add_argument(
        '--gaze360image_dir_test', dest='gaze360image_dir_test', help='Directory path for gaze images.',
        default='../gaze360_test/Image', type=str)
    parser.add_argument(
        '--gaze360label_dir_test', dest='gaze360label_dir_test', help='Directory path for gaze labels.',
        default='../gaze360_test/Label', type=str)
   
    # Important args -------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    parser.add_argument(
        '--dataset', dest='dataset', help='gaze360, mpiigaze',
        default= "gaze360", type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path to the folder contains models.', 
        default='output/snapshots/L2CS-gaze360-_loader-180-4-lr', type=str)
    parser.add_argument(
        '--evalpath', dest='evalpath', help='path for the output evaluating gaze test.',
        default="evaluation/L2CS-gaze360-_loader-180-4-lr", type=str)
    parser.add_argument(
        '--gpu',dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=100, type=int)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, [ResNet50], ''ResNet101, ResNet152, Squeezenet_1_0, Squeezenet_1_1, MobileNetV2',
        default='ResNet50', type=str)
    parser.add_argument(
    '--angle', dest='angle', help='bruh', default=90, type=int)
    # ---------------------------------------------------------------------------------------------------------------------
    # Important args ------------------------------------------------------------------------------------------------------
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    gpu = select_device(args.gpu_id, batch_size=args.batch_size)
    batch_size=args.batch_size
    angle=args.angle

    # Transformation for the model
    transformations = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Transformation for saving original images
    transformations_original = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])

    vri = VRI_GazeNet()
    vri.name = "VRI"
    saved_state_dict = torch.load("../models/VRI-181-June-7-LR1e-05-DEC1e-06-drop0.3-BATCH8-augment-CROSSENTROPY-True-BETA-10-180d-11.24-90d-10.98-40d-9.16.pkl")
    vri.load_state_dict(saved_state_dict)


    # TEST
    folder = os.listdir(args.gaze360label_dir_test)
    folder.sort()
    testlabelpathombined = [os.path.join(args.gaze360label_dir_test, j) for j in folder]
    gaze_dataset_test = datasets.Gaze360(testlabelpathombined, args.gaze360image_dir_test, transformations, 360, vri.binwidth)

    test_loader = torch.utils.data.DataLoader(
        dataset=gaze_dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    # Create an unshuffled dataloader for saving original images
    test_loader_original = torch.utils.data.DataLoader(
        dataset=gaze_dataset_test,
        batch_size=batch_size,
        shuffle=False,  # Do not shuffle
        num_workers=4,
        pin_memory=True)

    save_counter = 0  # To keep track of saved images
    save_frequency = 10  # Save every 10th image


    to_pil = ToPILImage()


    for model in (vri,):
        model.cuda(gpu)
        model.eval()
        total = 0
        start = datetime.datetime.now()
        with torch.no_grad():
            for i, (images_gaze, labels_gaze, cont_labels_gaze, name) in enumerate(test_loader_original):
                images = Variable(images_gaze).cuda(gpu)
                gazes = model.angles_gpu(images, gpu)
                yaw_predicted, pitch_predicted = gazes[0]

                images_np = images_gaze[0].cpu().numpy()  # Access the first image from the batch and convert to NumPy array
                images_pil = to_pil(images_gaze[0])  # Access the first image from the batch and convert to PIL Image

                # Draw ground truth in yellow
                ground_truth_yaw = cont_labels_gaze[i, 0] * np.pi / 180.0
                ground_truth_pitch = cont_labels_gaze[i, 1] * np.pi / 180.0
                draw_gaze(0, 0, images_gaze.shape[3], images_gaze.shape[2], images_np, (ground_truth_yaw, ground_truth_pitch), color=(255, 255, 0), scale=1, thickness=4, size=images_gaze.shape[3], bbox=((0, 0), (images_gaze.shape[3], images_gaze.shape[2])))

                # Draw predicted gaze in blue
                images_normalized = transformations(images_gaze)  # Apply transformations to the original image
                images_normalized = images_normalized.unsqueeze(0).cuda(gpu)  # Add batch dimension and move to GPU
                gazes_model = model.angles_gpu(images_normalized, gpu)
                yaw_predicted_model, pitch_predicted_model = gazes_model[0]
                draw_gaze(0, 0, images_gaze.shape[3], images_gaze.shape[2], images_np, (yaw_predicted_model, pitch_predicted_model), color=(0, 0, 255), scale=1, thickness=4, size=images_gaze.shape[3], bbox=((0, 0), (images_gaze.shape[3], images_gaze.shape[2])))

                cv2.imwrite("gaze_" + name[0] + ".jpg", images_pil)
                save_counter += 1

                if save_counter >= 20:
                    print("SAVED ALL")
                    break  # Stop processing after saving 20 images


        end = datetime.datetime.now()
        duration = end - start
        seconds = duration.total_seconds()

        log = f"[{model.name}] Images: {total}. Duration: {seconds}. FPS = {total / seconds}"
        print(log)



