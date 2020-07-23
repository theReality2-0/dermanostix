import random as rd
from torchvision.models import densenet121, vgg11, resnet50
import torch.nn as nn
import torch


class Lesions(object):
    lst = ["Actinic Keratosis",  "Basal Cell Carcinoma", "Seborrheic Keratosis",
           "Dermatofibroma", "Melanocytic Nevi",  "Melanoma", "Vascular Lesions"]

    def __init__(self):
        num_classes=7
        model_ft = densenet121(pretrained=True)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        model_ft.load_state_dict(torch.load('functioning.pth', map_location='cpu'), strict=True)
        model_ft.eval()


    def make_prediction(self, img):
        # order of greatest to least common: sebhorreic, mel nev, df, basal, vasc, actinic keratosis, melanoma
        var = rd.randint(1,101)
        if(var <= 23):
            return Lesions.lst[4]
        if(23 < var <= 43):
            return Lesions.lst[1]
        if(43 < var <= 57):
            return Lesions.lst[0]
        if(57 < var <= 69):
            return Lesions.lst[2]
        if(69 < var <= 80):
            return Lesions.lst[3]
        if(80 < var <= 90):
            return Lesions.lst[5]
        if(90 < var <= 100):
            return Lesions.lst[6]






