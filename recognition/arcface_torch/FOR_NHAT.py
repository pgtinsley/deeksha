import argparse

import cv2
import numpy as np
import torch

from backbones import get_model

import os
import glob
import pickle

# @torch.no_grad()
# def inference(weight, name, img):
#     if img is None:
#         img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
#     else:
#         img = cv2.imread(img)
#         img = cv2.resize(img, (112, 112))
# 
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = np.transpose(img, (2, 0, 1))
#     img = torch.from_numpy(img).unsqueeze(0).float()
#     img.div_(255).sub_(0.5).div_(0.5)
#     net = get_model(name, fp16=False)
#     net.load_state_dict(torch.load(weight))
#     net.eval()
#     feat = net(img).numpy()
#     print(feat)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r100', help='backbone network')
    parser.add_argument('--weight', type=str, default='/afs/crc.nd.edu/user/p/ptinsley/insightface/model_zoo/ms1mv3_arcface_r100_fp16/backbone.pth')
    parser.add_argument('--inputDir', type=str, required=True)
    parser.add_argument('--outFile', type=str, required=True)
    args = parser.parse_args()

#     inference(args.weight, args.network, args.img)

    net = get_model(args.network, fp16=False)
    net.load_state_dict(torch.load(args.weight))
    net.eval()

    fnames = glob.glob(os.path.join(args.inputDir,'*.jpeg'))
    print(len(fnames))
    print(fnames)
    
    features = {}
    for fname in fnames:
	print(fname)
        img = cv2.imread(fname)
        img = cv2.resize(img, (112, 112))
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        feat = net(img).detach().numpy()
        features[fname] = feat
        
    with open(args.outFile, 'wb') as f:
        pickle.dump(features, f)
