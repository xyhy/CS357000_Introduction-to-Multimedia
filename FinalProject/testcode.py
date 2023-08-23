import math
import io
import torch
from torchvision import transforms
import numpy as np
import cv2

from PIL import Image
from matplotlib import pyplot as plt

import matplotlib.pyplot as pltfrom
from model import ScaleHyperprior

device = 'cuda' if torch.cuda.is_available() else 'cpu'



checkpt = torch.load('/Users/yen/Desktop/YHY/college_course/Multimedia/Homework/FinalProject/checkpoint_best_loss.pth.tar')
net = ScaleHyperprior(192,128).from_state_dict(checkpt["state_dict"])
net.eval()

print(f'Parameters: {sum(p.numel() for p in net.parameters())}')

img = Image.open('/Users/yen/Desktop/YHY/college_course/Multimedia/Homework/FinalProject/Real/test/501__M_Left_little_finger.BMP')
# img = cv2.imread('/Users/yen/Desktop/YHY/college_course/Multimedia/Homework/FinalProject/Real/test/501__M_Left_index_finger.BMP')
train_transforms = transforms.Compose(
            [transforms.Resize((96,96)), transforms.Grayscale(1), transforms.ToTensor()]
        )
x = train_transforms(img).unsqueeze(0).to(device)

with torch.no_grad():
    out_net = net.forward(x)
out_net['x_hat'].clamp_(0, 1)

print(out_net['likelihoods']['y'].size(), out_net['likelihoods']['z'].size())

rec_net = transforms.ToPILImage()(out_net['x_hat'].squeeze().cpu())
# plt.imshow(rec_net)
# plt.show()
rec_net = rec_net.save('/Users/yen/Downloads/501__M_Left_little_finger.BMP')