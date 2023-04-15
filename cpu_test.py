import os, argparse
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
from model import L2CS

from fvcore.nn import FlopCountAnalysis
import typing

## FX MODE
from torch.quantization import quantize_fx

## EAGER MODE
from torch.quantization import quantize_dynamic

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze estimation using L2CSNet .')
     # Gaze360
    parser.add_argument(
        '--gaze360image_dir', dest='gaze360image_dir', help='Directory path for gaze images.',
        default='datasets/Gaze360/Image', type=str)
    parser.add_argument(
        '--gaze360label_dir', dest='gaze360label_dir', help='Directory path for gaze labels.',
        default='datasets/Gaze360/Label/test.label', type=str)
    # mpiigaze
    parser.add_argument(
        '--gazeMpiimage_dir', dest='gazeMpiimage_dir', help='Directory path for gaze images.',
        default='datasets/MPIIFaceGaze/Image', type=str)
    parser.add_argument(
        '--gazeMpiilabel_dir', dest='gazeMpiilabel_dir', help='Directory path for gaze labels.',
        default='datasets/MPIIFaceGaze/Label', type=str)
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
        '--bins', dest='bins', help='bruh', default=90, type=int)
    parser.add_argument(
        '--angle', dest='angle', help='bruh', default=90, type=int)
    # ---------------------------------------------------------------------------------------------------------------------
    # Important args ------------------------------------------------------------------------------------------------------
    args = parser.parse_args()
    return args


def getArch(arch,bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS( torchvision.models.resnet.BasicBlock,[2, 2,  2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS( torchvision.models.resnet.BasicBlock,[3, 4,  6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], bins)
    return model

if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = False
    gpu = select_device(args.gpu_id, batch_size=args.batch_size)
    batch_size=args.batch_size
    arch=args.arch
    data_set=args.dataset
    evalpath =args.evalpath
    snapshot_path = args.snapshot
    bins=args.bins
    angle=args.angle
    # bin_width=args.bin_width

    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    if data_set=="gaze360":
        folder = os.listdir(args.gaze360label_dir)
        folder.sort()
        testlabelpathombined = [os.path.join(args.gaze360label_dir, j) for j in folder] 
        gaze_dataset=datasets.Gaze360(testlabelpathombined,args.gaze360image_dir, transformations, 180, 4, train=False)
        test_loader = torch.utils.data.DataLoader(
            dataset=gaze_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True)


        if not os.path.exists(evalpath):
            os.makedirs(evalpath)


        softmax = nn.Softmax(dim=1)
        with open(os.path.join(evalpath,data_set+".log"), 'w') as outfile:
            avg_yaw=[]
            avg_pitch=[]
            avg_MAE=[]
            # Base network structure
            original_model=getArch(arch, 90)
            saved_state_dict = torch.load(snapshot_path)
            original_model.load_state_dict(saved_state_dict)
            original_model.cpu()
            # original_model.cuda(gpu)
            original_model.eval()
            total = 0
            idx_tensor = [idx for idx in range(90)]
            # idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
            idx_tensor = torch.FloatTensor(idx_tensor).cpu()
            avg_error = .0

            model_dynamic = quantize_dynamic(
                model=original_model, qconfig_spec={nn.LSTM, nn.Linear}, dtype=torch.qint8, inplace=False
            )
            torch.save(model_dynamic, "../Dynamic.pkl")


            for images, labels, cont_labels, name in test_loader:
                # images = Variable(images).cuda(gpu)
                images = Variable(images).cpu()
                break

            qconfig_dict = {"": torch.quantization.default_dynamic_qconfig}  # An empty key denotes the default applied to all modules
            model_prepared = quantize_fx.prepare_fx(original_model, qconfig_dict, images)
            model_fx = quantize_fx.convert_fx(model_prepared)

            torch.save(model_dynamic, "../FX.pkl")


            for model_name in ("original_model", "model_dynamic", "model_fx"):

                log = f"For model {model_name}:\n"
                outfile.write(log)
                print(log)

                model = eval(model_name)

                total_flops = 0
                total_flops_by_operator = typing.Counter()
                total_flops_by_module = typing.Counter()
                            
                with torch.no_grad():           
                    for j, (images, labels, cont_labels, name) in enumerate(test_loader):
                        # images = Variable(images).cuda(gpu)
                        images = Variable(images).cpu()

                        total += cont_labels.size(0)

                        label_pitch = cont_labels[:,1].float()*np.pi/180
                        label_yaw = cont_labels[:,0].float()*np.pi/180

                        gaze_yaw, gaze_pitch = model(images)
                        
                        # Binned predictions
                        _, pitch_bpred = torch.max(gaze_pitch.data, 1)
                        _, yaw_bpred = torch.max(gaze_yaw.data, 1)
                        
            
                        # Continuous predictions
                        pitch_predicted = softmax(gaze_pitch)
                        yaw_predicted = softmax(gaze_yaw)
                        
                        # mapping from binned (0 to 90) to angles (-180 to 180)  
                        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 4 - 180
                        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 4 - 180

                        pitch_predicted = pitch_predicted*np.pi/180
                        yaw_predicted = yaw_predicted*np.pi/180

                        for p,y,pl,yl in zip(pitch_predicted,yaw_predicted,label_pitch,label_yaw):
                            avg_error += angular(gazeto3d([p,y]), gazeto3d([pl,yl]))
                        
                        # Calculate AGAIN, but calculate FLOPs
                        flops = FlopCountAnalysis(model, images)

                        total_flops += flops.total()
                        total_flops_by_operator.update(flops.by_operator())

            
                    avg_MAE.append(avg_error/total)
                    log = f"\tTotal Num:{total}, MAE:{avg_error/total}\n"
                    outfile.write(log)
                    print(log)

                    log = f"\tTotal Flops:{total_flops}\nByOperator:{total_flops_by_operator}\n"
                    outfile.write(log)
                    print(log)

                    log = f"\tTotal Flops/image:{total_flops/total}.\n"
                    outfile.write(log)
                    print(log)
            
