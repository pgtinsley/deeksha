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

from facenet_pytorch import MTCNN
from PIL import Image

from torchvision import transforms
import torchvision.models as models

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

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

# if __name__ == "__main__":

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

## ArcFace

@torch.no_grad()
def inference(weight, name, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    feat = net(img).numpy()
    with open('/afs/crc.nd.edu/user/p/ptinsley/stargan-v2/probe-aligned/probe_arc.pkl','wb') as f:
        pickle.dump(feat[0], f)
#     print(feat)

name = 'r100'
weight = '/afs/crc.nd.edu/user/p/ptinsley/insightface/model_zoo/ms1mv3_arcface_r100_fp16/backbone.pth'
fname = '/afs/crc.nd.edu/user/p/ptinsley/stargan-v2/probe-aligned/probe.png'

inference(weight, name, fname)

## VGG19

class FeatureExtractor(torch.nn.Module):
  def __init__(self, model):
    super(FeatureExtractor, self).__init__()
		# Extract VGG-16 Feature Layers
    self.features = list(model.features)
    self.features = torch.nn.Sequential(*self.features)
		# Extract VGG-16 Average Pooling Layer
    self.pooling = model.avgpool
		# Convert the image into one-dimensional vector
    self.flatten = torch.nn.Flatten()
		# Extract the first part of fully-connected layer from VGG16
    self.fc = model.classifier[0]
  
  def forward(self, x):
		# It will take the input 'x' until it returns the feature vector called 'out'
    out = self.features(x)
    out = self.pooling(out)
    out = self.flatten(out)
    out = self.fc(out)
    return out


model = models.vgg19(pretrained=True)
new_model = FeatureExtractor(model)

# print(model)
# print(new_model)


input_image = Image.open(fname)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

# if torch.cuda.is_available():
input_batch = input_batch.to('cuda')
model.to('cuda')

with torch.no_grad():
    output = new_model(input_batch)

# print(output.cpu().numpy())

with open('/afs/crc.nd.edu/user/p/ptinsley/stargan-v2/probe-aligned/probe_vgg.pkl','wb') as f:
    pickle.dump(output.cpu().numpy(), f)

    
# print(output.shape)

#     mtcnn = MTCNN(image_size=112, margin=14, device=device)
    
#     with open('/afs/crc.nd.edu/user/p/ptinsley/TWINS/fnames_twins.pkl','rb') as f:
#         fnames = pickle.load(f)
    
#     print(len(fnames))
#     print(fnames[:5])

#     exts = [fname.split('/')[-1] for fname in fnames]
#     print(np.unique(exts).shape)
#     exit()

#     for fname in fnames:
#         mtcnn(Image.open(fname), 
#               save_path=os.path.join('/scratch365/ptinsley/twins112/',fname.split('/')[-1]))
#     
#     print('Done writing to twins112...')
    
#     parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
#     parser.add_argument('--network', type=str, default='r100', help='backbone network')
#     parser.add_argument('--weight', type=str, default='/afs/crc.nd.edu/user/p/ptinsley/insightface/model_zoo/ms1mv3_arcface_r100_fp16/backbone.pth')
# #     parser.add_argument('--inputDir', type=str, required=True)
# #     parser.add_argument('--outFile', type=str, required=True)
#     args = parser.parse_args()
# 
#     net = get_model(args.network, fp16=False)
#     net.load_state_dict(torch.load(args.weight))
#     net.eval()
# 
# #     fnames = glob.glob(os.path.join(args.inputDir,'*.jpeg')) # NHAT version
#     fnames = glob.glob(os.path.join(args.inputDir,'*.png')) # NHAT version
#     
#     features = {}
#     for fname in fnames:
#         print(fname)
#         img = cv2.imread(fname)
#         img = cv2.resize(img, (112, 112))
#         
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = np.transpose(img, (2, 0, 1))
#         img = torch.from_numpy(img).unsqueeze(0).float()
#         img.div_(255).sub_(0.5).div_(0.5)
#         feat = net(img).detach().numpy()
#         features[fname] = feat
#         
#     with open(args.outFile, 'wb') as f:
#         pickle.dump(features, f)
