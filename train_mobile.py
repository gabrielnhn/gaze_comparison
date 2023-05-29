import os
import argparse
import time

import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

import datasets
# from model import ML2CS, ML2CS180
from model import VRI_GazeNet
from utils import select_device, natural_keys, gazeto3d, angular
import numpy as np

import matplotlib.pyplot as plt
from datetime import datetime
from calendar import month_name

from PIL import ImageOps


def parse_args():
    """Parse input arguments."""
    
    # DATASET ARGS
    parser = argparse.ArgumentParser(description='Gaze estimation using L2CSNet.')
    parser.add_argument(
        '--gaze360image_dir_train', dest='gaze360image_dir_train', help='Directory path for gaze images.',
        default='../gaze360_train/Image', type=str)
    parser.add_argument(
        '--gaze360label_dir_train', dest='gaze360label_dir_train', help='Directory path for gaze labels.',
        default='../gaze360_train/Label', type=str)   
    parser.add_argument(
        '--gaze360image_dir_val', dest='gaze360image_dir_val', help='Directory path for gaze images.',
        default='../gaze360_val/Image', type=str)
    parser.add_argument(
        '--gaze360label_dir_val', dest='gaze360label_dir_val', help='Directory path for gaze labels.',
        default='../gaze360_val/Label', type=str)
    parser.add_argument(
        '--dataset', dest='dataset', help='mpiigaze, rtgene, gaze360, ethgaze',
        default= "gaze360", type=str)
    parser.add_argument(
        '--output', dest='output', help='Path of output models.',
        default='output/snapshots/', type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='', type=str)
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0] or multiple 0,1,2,3',
        default='0', type=str)

    ## MODEL ARGS
    
    parser.add_argument(
        '--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
        default=60, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=1, type=int)
    parser.add_argument(
        '--alpha', dest='alpha', help='Regression loss coefficient.',
        default=1, type=float)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.00001, type=float)
    parser.add_argument(
        '--decay', dest='decay', help='Learning rate decay.',
        default=0.000001, type=float)
    parser.add_argument(
        '--reg_only', dest='reg_only', help='Only use regression loss.',
        default=False, type=bool)
    parser.add_argument(
        '--augment', dest='augment', help='If dataset should be augmented',
        default=False, type=bool)
    # ---------------------------------------------------------------------------------------------------------------------
    # Important args ------------------------------------------------------------------------------------------------------
    args = parser.parse_args()
    return args



# augmentation_transform = transforms.Compose([
#     transforms.RandomApply(torch.nn.ModuleList([
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
#     ]), p=0.5),
#     transforms.RandomApply(torch.nn.ModuleList([
#         transforms.GaussianBlur(kernel_size=3)  # Adjust kernel_size as desired
#     ]), p=0.5),
# ])



if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = select_device(args.gpu_id, batch_size=args.batch_size)
    data_set=args.dataset
    alpha = args.alpha
    output=args.output    
    
    val_transform = transforms.Compose([
        # transforms.Resize(448),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    train_transform = transforms.Compose([
        # transforms.Resize(448),
        transforms.Resize(224),
        transforms.RandomApply(torch.nn.ModuleList([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ]), p=0.5),
        # transforms.RandomApply(torch.nn.ModuleList([
        #     transforms.GaussianBlur(kernel_size=3)  # Adjust kernel_size as desired
        # ]), p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    if data_set=="gaze360":

        # model = ML2CS180()
        model = VRI_GazeNet()
        model.cuda(gpu)
        bins = model.num_bins

        binwidth = model.binwidth

        print("BINWIDTH", binwidth)

        # TRAIN
        folder = os.listdir(args.gaze360label_dir_train)
        folder.sort()
        testlabelpathombined = [os.path.join(args.gaze360label_dir_train, j) for j in folder] 

        if args.augment:
            dataset=datasets.Gaze360(testlabelpathombined, args.gaze360image_dir_train, train_transform, 180, binwidth, mirror=True)
        else:
            dataset=datasets.Gaze360(testlabelpathombined, args.gaze360image_dir_train, val_transform, 180, binwidth, mirror=True)
        print('Loading data.')
        train_loader_gaze = DataLoader(
            dataset=dataset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=0,
            pin_memory=True)

        # # VALIDATION
        folder = os.listdir(args.gaze360label_dir_val)
        folder.sort()
        testlabelpathombined = [os.path.join(args.gaze360label_dir_val, j) for j in folder]
        gaze_dataset_val=datasets.Gaze360(testlabelpathombined,args.gaze360image_dir_val, val_transform, 180, binwidth)
        # gaze_dataset_val=datasets.Gaze360(testlabelpathombined,args.gaze360image_dir_val, transformations, 180, binwidth)
        
        val_loader = torch.utils.data.DataLoader(
            dataset=gaze_dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True)
        

        # torch.backends.cudnn.benchmark = True

        day = datetime.now().day
        month = datetime.now().month
        summary_name = f"VRI-{bins}-{month_name[month]}-{day}-LR{args.lr}-DEC{args.decay}-drop0.3-BATCH{batch_size}-augment-CROSSENTROPY-{not args.reg_only}"
        output=os.path.join(output, summary_name)
        if not os.path.exists(output):
            os.makedirs(output)
        
        criterion = nn.CrossEntropyLoss().cuda(gpu)
        # reg_criterion = nn.MSELoss().cuda(gpu)
        reg_criterion = nn.L1Loss().cuda(gpu)
        # softmax = nn.Softmax(dim=1).cuda(gpu)

        idx_tensor = [idx for idx in range(model.num_bins)]
        idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
        
        optimizer_gaze = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr, weight_decay=args.decay)

        avg_MAE_train=[]
        avg_MAE_val=[]

        best_val_loss = float("inf")
        best_val_epoch = None
        best_val_model = None
        
        best_train_loss = float("inf")
        best_train_epoch = None
        best_train_model = None

        all_models = []
        for epoch in range(num_epochs):
            avg_error_train = 0.0
            total_train = 0

            sum_loss_pitch_gaze = 0
            sum_loss_yaw_gaze = 0
            iter_gaze = 0
            
            model.train()
            for i, (images_gaze, labels_gaze, cont_labels_gaze,name) in enumerate(train_loader_gaze):

                # total_train += cont_labels_gaze.size(0)
                total_train += 2 * cont_labels_gaze.size(0)
                images_gaze = Variable(images_gaze).cuda(gpu)
                
                # Binned labels
                label_yaw_gaze = Variable(labels_gaze[:, 0]).cuda(gpu)
                label_pitch_gaze = Variable(labels_gaze[:, 1]).cuda(gpu)
                # Continuous labels
                label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 0]).cuda(gpu)
                label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 1]).cuda(gpu)

                #Mirror
                mirror_image = images_gaze.detach().clone()
                for i in range(len(mirror_image)):
                    mirror_image[i] = torchvision.transforms.functional.hflip(mirror_image[i])

                # print(mirror_image)
                mirror_yaw_bin = [(model.num_bins -1 - binned_yaw) for binned_yaw in label_yaw_gaze]
                mirror_pitch_bin = [int(binned_pitch) for binned_pitch in label_pitch_gaze]
                mirror_pitch_cont = [pitch for pitch in label_pitch_cont_gaze]
                mirror_yaw_cont = [-yaw for yaw in label_yaw_cont_gaze]

                mirror_image = Variable(mirror_image).cuda(gpu)
                mirror_yaw_bin = Variable(torch.tensor(mirror_yaw_bin)).cuda(gpu)
                mirror_pitch_bin = Variable(torch.tensor(mirror_pitch_bin)).cuda(gpu)
                mirror_pitch_cont = Variable(torch.Tensor(mirror_pitch_cont)).cuda(gpu)
                mirror_yaw_cont = Variable(torch.Tensor(mirror_yaw_cont)).cuda(gpu)

                ##CALCULATE ORIGINAL
                # images_gaze = augmentation_transform(images_gaze)
                yaw_predicted, pitch_predicted = model(images_gaze)

                with torch.no_grad():
                    pitch_predicted_cpu = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * binwidth - 180
                    yaw_predicted_cpu = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * binwidth - 180
                    label_pitch_cpu = cont_labels_gaze[:,1].float()*np.pi/180
                    label_pitch_cpu = label_pitch_cpu.cpu()
                    label_yaw_cpu = cont_labels_gaze[:,0].float()*np.pi/180
                    label_yaw_cpu = label_yaw_cpu.cpu()

                pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * binwidth - 180
                yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * binwidth - 180

                loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont_gaze)
                loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont_gaze)

                # Total loss

                if args.reg_only:
                    loss_pitch_gaze += loss_reg_pitch
                    loss_yaw_gaze += loss_reg_yaw
                else:
                    loss_pitch_gaze += alpha * loss_reg_pitch
                    loss_yaw_gaze += alpha * loss_reg_yaw

                loss_seq = [loss_yaw_gaze, loss_pitch_gaze]
                grad_seq = [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]
                optimizer_gaze.zero_grad(set_to_none=True)
                torch.autograd.backward(loss_seq, grad_seq)
                optimizer_gaze.step()

                with torch.no_grad():
                    pitch_predicted_cpu = pitch_predicted_cpu*np.pi/180
                    yaw_predicted_cpu = yaw_predicted_cpu*np.pi/180 

                    for p,y,pl,yl in zip(pitch_predicted_cpu,yaw_predicted_cpu,label_pitch_cpu,label_yaw_cpu):
                        avg_error_train += angular(gazeto3d([p,y]), gazeto3d([pl,yl]))
 

                ####### CALCULATE MIRROR
                # mirror_image = augmentation_transform(mirror_image)
                yaw_predicted, pitch_predicted = model(mirror_image)

                # Cross entropy loss
                loss_pitch_gaze = criterion(pitch_predicted, mirror_pitch_bin)
                loss_yaw_gaze = criterion(yaw_predicted, mirror_yaw_bin)

                with torch.no_grad():
                    pitch_predicted_cpu = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * binwidth - 180
                    yaw_predicted_cpu = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * binwidth - 180

                    # label_pitch_cpu = cont_labels_gaze[:,1].float()*np.pi/180
                    label_pitch_cpu = mirror_pitch_cont.float()*np.pi/180
                    label_pitch_cpu = label_pitch_cpu.cpu()
                    # label_yaw_cpu = cont_labels_gaze[:,0].float()*np.pi/180
                    label_yaw_cpu = mirror_yaw_cont.float()*np.pi/180
                    label_yaw_cpu = label_yaw_cpu.cpu()

                pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * binwidth - 180
                yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * binwidth - 180

                loss_reg_pitch = reg_criterion(pitch_predicted, mirror_pitch_cont)
                loss_reg_yaw = reg_criterion(yaw_predicted, mirror_yaw_cont)

                # Total loss
                loss_pitch_gaze += alpha * loss_reg_pitch
                loss_yaw_gaze += alpha * loss_reg_yaw

                loss_seq = [loss_yaw_gaze, loss_pitch_gaze]
                grad_seq = [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]
                optimizer_gaze.zero_grad(set_to_none=True)
                torch.autograd.backward(loss_seq, grad_seq)
                optimizer_gaze.step()

                with torch.no_grad():
                    pitch_predicted_cpu = pitch_predicted_cpu*np.pi/180
                    yaw_predicted_cpu = yaw_predicted_cpu*np.pi/180 

                    for p,y,pl,yl in zip(pitch_predicted_cpu,yaw_predicted_cpu,label_pitch_cpu,label_yaw_cpu):
                        avg_error_train += angular(gazeto3d([p,y]), gazeto3d([pl,yl]))



            model.eval()
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
                    

                    yaw_predicted, pitch_predicted = model(images)
        
                    # Continuous predictions
                    # pitch_predicted = softmax(gaze_pitch)
                    # yaw_predicted = softmax(gaze_yaw)
                    
                    # mapping from binned (0 to 28) to angels (-180 to 180)  
                    pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * binwidth - 180
                    yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * binwidth - 180

                    pitch_predicted = pitch_predicted*np.pi/180
                    yaw_predicted = yaw_predicted*np.pi/180

                    for p,y,pl,yl in zip(pitch_predicted,yaw_predicted,label_pitch,label_yaw):
                        avg_error += angular(gazeto3d([p,y]), gazeto3d([pl,yl]))
                    
            val_loss = avg_error/total
                    
            # val_loss = (avg_error_val/total_val)
            train_loss = (avg_error_train/total_train)

            print(f"Epoch {epoch}: val:{val_loss} ; train:{train_loss}")

            avg_MAE_val.append(val_loss)
            avg_MAE_train.append(train_loss)

            torch.save(model.state_dict(), output +'/'+'epoch_' + str(epoch) + '.pkl')

          
            all_models.append((model.state_dict().copy(), val_loss, train_loss, epoch))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch
                best_val_state_dict = model.state_dict().copy()
            
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_train_epoch = epoch
                best_train_state_dict = model.state_dict().copy()
            

        print(F'BEST EPOCH (VAL): {best_val_epoch}')
        print(F'BEST LOSS (VAL): {best_val_loss}')
        print("Saving best model...")
        torch.save(best_val_state_dict, output +'/'+'_best_val_' + str(best_val_epoch) + '.pkl')
        print("Saved")

        print(F'BEST EPOCH (train): {best_train_epoch}')
        print(F'BEST LOSS (train): {best_train_loss}')
        print("Saving best model...")
        torch.save(best_train_state_dict, output +'/'+'_best_train_' + str(best_train_epoch) + '.pkl')
        print("Saved")
        
        print("Generating plot..")
        epoch_list = list(range(num_epochs))
        
        fig = plt.figure()        
        plt.xlabel('Epoch')
        plt.ylabel('MAE (degrees)')
        plt.title('Mean angular error')
        plt.plot(epoch_list, avg_MAE_train, color='b', label='train')
        plt.plot(epoch_list, avg_MAE_val, color='g', label='validation')

        plt.legend(loc="upper left")
        plt.locator_params(axis='x', nbins=num_epochs//3)
        fig.savefig(os.path.join(output,data_set+".png"), format='png')
        # plt.show()


        # # save top10 val
        # all_models.sort(key=lambda x: x[1])

        # for good_model in all_models[:10]:
        #     epoch = good_model[-1]
        #     state_dict = good_model[0]

        #     print(f"Saving state_dict from epoch {epoch}")
        #     if (epoch != best_val_epoch) and (epoch != best_train_epoch):
        #         torch.save(state_dict, output +'/'+'_epoch_' + str(epoch) + '.pkl')
            