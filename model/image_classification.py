import torch
from torch.autograd import Variable
from torchvision import models
from PIL import Image
import torch.nn.functional as F
import json
import io
import requests

use_gpu = torch.cuda.is_available()

def load_image(path, transform):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        img = transform(img)
        return img.unsqueeze(0)

def load_image_from_url(img_url, transform):
    response = requests.get(img_url)
    img = Image.open(io.BytesIO(response.content))
    img = img.convert('RGB')
    img = transform(img)
    return img.unsqueeze(0)

def load_resnet18():
    
    net = torchvision.models.resnet18(pretrained=True)
    
    if use_gpu:
        net = net.cuda()

    net.eval()
    
    return net

def load_resnet18_from_file(model_path):
    net = models.resnet.ResNet(models.resnet.BasicBlock, [2, 2, 2, 2])
    model_weights = torch.load(model_path)
    net.load_state_dict(model_weights)
    if use_gpu:
        net = net.cuda()

    net.eval()
    return net

def load_labels(label_path):
    label_map = json.load(open(label_path,'r'))
    label_dict = {}
    for k in label_map:
        label_dict[int(k)] = label_map[k]
    return label_dict

def predict(net, inputs):
    if use_gpu:
        inputs = inputs.cuda()
    inputs = Variable(inputs)
    outputs = net(inputs)
    data = outputs.data
    outputs = F.softmax(data).data
    return outputs
