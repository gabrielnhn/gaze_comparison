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
from utils import select_device, natural_keys, gazeto3d, angular
from model import L2CS, ML2CS, ML2CS180

from fvcore.nn import FlopCountAnalysis
import typing


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze estimation using L2CSNet .')
     # Gaze360
    parser.add_argument(
        '--gaze360image_dir_test', dest='gaze360image_dir_test', help='Directory path for gaze images.',
        default='../gaze360_test/Label', type=str)
    parser.add_argument(
        '--gaze360label_dir_test', dest='gaze360label_dir_test', help='Directory path for gaze labels.',
        default='../gaze360_test/Label', type=str)
   
    parser.add_argument(
        '--gaze360image_dir_val', dest='gaze360image_dir_val', help='Directory path for gaze images.',
        default='../gaze360_val/Image', type=str)
    parser.add_argument(
        '--gaze360label_dir_val', dest='gaze360label_dir_val', help='Directory path for gaze labels.',
        default='../gaze360_val/Label', type=str)
   
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
    # arch=args.arch
    data_set=args.dataset
    evalpath =args.evalpath
    snapshot_path = args.snapshot
    # bins=args.bins
    angle=args.angle
    # bin_width=args.bin_width

    transformations = transforms.Compose([
        # transforms.Resize(448),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    binwidth = int(360/180)
    
    if data_set=="gaze360":
        
        # TEST
        folder = os.listdir(args.gaze360label_dir_test)
        folder.sort()
        testlabelpathombined = [os.path.join(args.gaze360label_dir_test, j) for j in folder]
        # gaze_dataset_test=datasets.Gaze360(testlabelpathombined,args.gaze360image_dir_test, transformations, 90, binwidth, train=False)
        gaze_dataset_test=datasets.Gaze360(testlabelpathombined,args.gaze360image_dir_test, transformations, angle, binwidth, train=True)
        
        test_loader = torch.utils.data.DataLoader(
            dataset=gaze_dataset_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True)

        # VALIDATION
        folder = os.listdir(args.gaze360label_dir_val)
        folder.sort()
        testlabelpathombined = [os.path.join(args.gaze360label_dir_val, j) for j in folder]
        # gaze_dataset_val=datasets.Gaze360(testlabelpathombined,args.gaze360image_dir_val, transformations, angle, binwidth, train=True)
        gaze_dataset_val=datasets.Gaze360(testlabelpathombined,args.gaze360image_dir_val, transformations, 180, binwidth)

        
        val_loader = torch.utils.data.DataLoader(
            dataset=gaze_dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True)
        

        if not os.path.exists(evalpath):
            os.makedirs(evalpath)


        # list all epochs for testing
        folder = os.listdir(snapshot_path)
        folder.sort(key=natural_keys)
        softmax = nn.Softmax(dim=1)
        with open(os.path.join(evalpath,data_set+".log"), 'w') as outfile:
            configuration = f"\ntest configuration = gpu_id={gpu}, batch_size={batch_size}-----\n"
            print(configuration)
            outfile.write(configuration)
            epoch_list=[]
            avg_yaw=[]
            avg_pitch=[]
            avg_MAE_test=[]
            avg_MAE_val=[]
            for epochs in folder:
                # Base network structure

                model = ML2CS180()
                saved_state_dict = torch.load(os.path.join(snapshot_path, epochs))
                model.load_state_dict(saved_state_dict)
                model.cuda(gpu)
                model.eval()

                bins = model.num_bins
                # binwidth = int(360/bins)
                idx_tensor = [idx for idx in range(bins)]
                idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
                
                ## TEST
                with torch.no_grad():           
                    total = 0
                    avg_error = .0
                    for j, (images, labels, cont_labels, name) in enumerate(test_loader):
                        images = Variable(images).cuda(gpu)
                        total += cont_labels.size(0)

                        label_yaw = cont_labels[:,0].float()*np.pi/180
                        label_pitch = cont_labels[:,1].float()*np.pi/180

                        gaze_yaw, gaze_pitch = model(images)
                        
                        # Binned predictions
                        _, pitch_bpred = torch.max(gaze_pitch.data, 1)
                        _, yaw_bpred = torch.max(gaze_yaw.data, 1)
                        
            
                        # Continuous predictions
                        pitch_predicted = softmax(gaze_pitch)
                        yaw_predicted = softmax(gaze_yaw)
                        
                        # mapping from binned (0 to 28) to angels (-180 to 180)  
                        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * binwidth - 180
                        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * binwidth - 180

                        pitch_predicted = pitch_predicted*np.pi/180
                        yaw_predicted = yaw_predicted*np.pi/180

                        for p,y,pl,yl in zip(pitch_predicted,yaw_predicted,label_pitch,label_yaw):
                            avg_error += angular(gazeto3d([p,y]), gazeto3d([pl,yl]))

                t = avg_error/total
                avg_MAE_test.append(t)

                ## VALIDATION        
                with torch.no_grad():
                    total = 0
                    # idx_tensor = [idx for idx in range(bins)]
                    # idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
                    avg_error = .0        
                    for j, (images, labels, cont_labels, name) in enumerate(val_loader):
                        images = Variable(images).cuda(gpu)
                        total += cont_labels.size(0)

                        label_yaw = cont_labels[:,0].float()*np.pi/180
                        label_pitch = cont_labels[:,1].float()*np.pi/180
                        

                        gaze_yaw, gaze_pitch = model(images)
            
                        # Continuous predictions
                        pitch_predicted = softmax(gaze_pitch)
                        yaw_predicted = softmax(gaze_yaw)
                        
                        # mapping from binned (0 to 28) to angels (-180 to 180)  
                        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * binwidth - 180
                        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * binwidth - 180

                        pitch_predicted = pitch_predicted*np.pi/180
                        yaw_predicted = yaw_predicted*np.pi/180

                        for p,y,pl,yl in zip(pitch_predicted,yaw_predicted,label_pitch,label_yaw):
                            avg_error += angular(gazeto3d([p,y]), gazeto3d([pl,yl]))
                        
                v = avg_error/total
                avg_MAE_val.append(v)
                
                # x = ''.join(filter(lambda i: i.isdigit(), epochs))
                loger = f"[{epochs}---{args.dataset}] Total Num:{total},MAE_V:{v}, MAE_T:{t}\n"
                outfile.write(loger)
                print(loger)
                # epoch_list.append(x)
        

        epoch_list = list(range(len(folder)))
        fig = plt.figure()        
        plt.xlabel('epoch')
        plt.ylabel('avg')
        plt.title('Gaze angular error')
        plt.plot(epoch_list, avg_MAE_test, color='b', label='test')
        plt.plot(epoch_list, avg_MAE_val, color='g', label='val')

        plt.legend()
        # plt.locator_params(axis='x', nbins=30)

        fig.savefig(os.path.join(evalpath,data_set+".png"), format='png')
        # plt.show()
