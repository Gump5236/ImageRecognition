from __future__ import print_function
import torch
from torchvision.transforms import RandomCrop, Resize, Compose, ToTensor
import numpy as np
import time
import os
import cv2

imagesize = (224,224,3)
labeldict = {'0' : 'sea', '1' : 'ship'}

def predict(model, imgarr, device):
    inimage = cv2.cvtColor(imgarr, cv2.COLOR_BGR2RGB)
    inimage = cv2.resize(inimage, (imagesize[0], imagesize[1]))
    inimage = np.transpose(inimage, (2, 0, 1)) / 255
    inimage = np.expand_dims(inimage, axis=0)
    inimage = torch.tensor(inimage.astype('float32'))
    inimage = inimage.to(device)
    netout = model(torch.tensor(inimage))
    label = str(netout.argmax().item())
    return labeldict[label]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('runing device : {}'.format(device))
model = torch.load('./best#111.pb')
model.to(device)
model.eval()

testdir = '/content/drive/MyDrive/DataSet/test/sea'
for filename in os.listdir(testdir):
    filepath = os.path.join(testdir, filename)
    img = cv2.imread(filepath)
    #rec_c = cv2.cvtColor(rec_c, cv2.COLOR)
    # print(img.shape)
    sol = predict(model, img, device)
    print(filename +'prediction : {}'.format(sol))
