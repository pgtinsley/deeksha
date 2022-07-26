import os
import glob
import pickle
import argparse

# arcface
import cv2
import numpy as np
import torch
from backbones import get_model

# retinaface
import sys
import datetime
# sys.path.append('../../detection/retinaface/')
# from retinaface import RetinaFace

# from facenet_pytorch import MTCNN
from PIL import Image
import torchvision.transforms as T

import csv

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

# def run_fast_scandir(dir, ext):    # dir: str, ext: list
#     subfolders, files = [], []
# 
#     for f in os.scandir(dir):
#         if f.is_dir():
#             subfolders.append(f.path)
#         if f.is_file():
#             if os.path.splitext(f.name)[1].lower() in ext:
#                 files.append(f.path)
# 
# 
#     for dir in list(subfolders):
#         sf, f = run_fast_scandir(dir, ext)
#         subfolders.extend(sf)
#         files.extend(f)
#         
#     return subfolders, files

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r100', help='backbone network')
    parser.add_argument('--weight', type=str, default='/afs/crc.nd.edu/user/p/ptinsley/insightface/model_zoo/ms1mv3_arcface_r100_fp16/backbone.pth')
    parser.add_argument('--inputDir', type=str, required=True)
    parser.add_argument('--outFile', type=str, required=True)
    parser.add_argument('--i', type=int, default=0)
    args = parser.parse_args()

#     mtcnn = MTCNN(image_size=512, margin=288, device=device)

#     _, fnames = run_fast_scandir(args.inputDir, ['.jpg'])
    fnames = glob.glob(os.path.join(args.inputDir, '*.png'), recursive=True)
    print(len(fnames))
    
    idx = args.i
    
#     with open('/afs/crc.nd.edu/user/p/ptinsley/TWINS/fnames_twins.pkl','rb') as f:
#         fnames = pickle.load(f)
    
#     print(len(fnames))
#     print(fnames[:5])
#     exit()

#     exts = [fname.split('/')[-1] for fname in fnames]
#     print(np.unique(exts).shape)
#     exit()

#     for fname in fnames:
#         mtcnn(Image.open(fname), 
#               save_path=os.path.join('/scratch365/ptinsley/twins112/',fname.split('/')[-1]))
#     
#     print('Done writing to twins112...')
    
    net = get_model(args.network, fp16=False)
    net.load_state_dict(torch.load(args.weight, map_location=device))
#     net.load_state_dict(torch.load(args.weight, map_location='cpu'))
    net.eval()

#     fnames = glob.glob(os.path.join(args.inputDir,'*.jpeg')) # NHAT version
#     fnames = glob.glob(os.path.join(args.inputDir,'**/*.png')) # NHAT version
    
#     transform = T.ToPILImage()
    
    
    
    with open(args.outFile, 'w') as f:
        
        spamwriter = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    
#         features = {}
#         for fname in fnames[(idx-1)*30000:(idx)*30000]:
#         for fname in fnames[(idx-1)*30000:((idx-1)*30000)+5]:
        for fname in fnames:    

            print(fname)
    #         img_pil = Image.open(fname)
            
            img = cv2.imread(fname)
    #         img = cv2.resize(img, (112, 112))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img).unsqueeze(0).float()
            img.div_(255).sub_(0.5).div_(0.5)
            feat = net(img).detach().numpy()
    #         print(feat)
            spamwriter.writerow([fname,feat])
            
#             features[fname] = feat
#         

#     print(features)
#     with open(args.outFile, 'wb') as f:
#         pickle.dump(features, f)
