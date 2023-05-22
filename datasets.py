import os
import numpy as np
import cv2


import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter, ImageFile, ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Gaze360(Dataset):
    def __init__(self, path, root, transform, angle, binwidth, train=True, mirror=False):
        self.transform = transform
        self.root = root
        self.orig_list_len = 0
        self.angle = angle
        self.mirror = mirror
        # if train==False:
        #   angle=90
        self.binwidth=binwidth
        self.lines = []
        if isinstance(path, list):
            for i in path:
                with open(i) as f:
                    print("here")
                    line = f.readlines()
                    line.pop(0)
                    self.lines.extend(line)
        else:
            with open(path) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len = len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[5]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0]*180/np.pi)) <= angle and abs((label[1]*180/np.pi)) <= angle:
                        self.lines.append(line)
                    
                        
        print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len-len(self.lines), angle))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        line = line.strip().split(" ")

        face = line[0]
        lefteye = line[1]
        righteye = line[2]
        name = line[3]
        gaze2d = line[5]
        label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        pitch = label[0]* 180 / np.pi
        yaw = label[1]* 180 / np.pi


            


        img = Image.open(os.path.join(self.root, face))

        if self.mirror:
            mirror_image = ImageOps.mirror(img)


        if self.transform:
            img = self.transform(img)
            if self.mirror:
                mirror_image = ImageOps.mirror(mirror_image)        
        

        
        
        # Bin values
        bins = np.array(range(-1*self.angle, self.angle, self.binwidth))
        binned_pose = np.digitize([pitch, yaw], bins) - 1

        labels = binned_pose
        cont_labels = torch.FloatTensor([pitch, yaw])
        
        if self.mirror:
            num_bins = int(360/self.binwidth)
            mirror_bin = (binned_pose + num_bins//2) % num_bins
            mirror_pitch = (pitch + 180) % (360)            
            mirror_yaw = (yaw + 180) % (360)
            mirror_cont = torch.FloatTensor([mirror_pitch, mirror_yaw])

            return (img, mirror_image), (labels, mirror_bin), (cont_labels, mirror_cont), (name)


        else:
            return img, labels, cont_labels, name
