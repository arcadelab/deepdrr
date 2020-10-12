import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNetGenerator(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, gpu_ids = [0]):
        super(SimpleNetGenerator, self).__init__()
        model = [nn.ReflectionPad2d(5),
            nn.Conv2d(1,4,11,padding=0),
            nn.ReLU(True),
            nn.ReflectionPad2d(5),
            nn.Conv2d(4, 8, 11, padding=0),
            nn.ReLU(True),
            nn.ReflectionPad2d(5),
            nn.Conv2d(8, 16, 11, padding=0),
            nn.ReLU(True),
            nn.ReflectionPad2d(5),
            nn.Conv2d(16, 16, 11, padding=0),
            nn.ReLU(True),
            nn.ReflectionPad2d(5),
            nn.Conv2d(16, 8, 11, padding=0),
            nn.ReLU(True),
            nn.ReflectionPad2d(5),
            nn.Conv2d(8, 1, 11, padding=0),
            nn.ReflectionPad2d(15),
            nn.Conv2d(1, 1, 31, padding=0),
            nn.ReLU(True),
            nn.ReflectionPad2d(15),
            nn.Conv2d(1, 1, 31, padding=0),
            nn.ReLU(True),
            nn.ReflectionPad2d(15),
            nn.Conv2d(1, 1, 31, padding=0),
            nn.ReLU(True),
            nn.ReflectionPad2d(15),
            nn.Conv2d(1, 1, 31, padding=0)]

        self.generator = nn.Sequential(*model)

    def forward(self, input):
        return self.generator(input)