import numpy as np
import torch
import torch.nn as nn

from torchvision.models.resnet import ResNet, BasicBlock
from torchvision import transforms as T


class ResNetTL(ResNet):
    def __init__(self, block=BasicBlock, layers=[3, 4, 6, 3], num_classes=[8], **kvargs):
        super(ResNetTL, self).__init__(block, layers, **kvargs)
        self.fc_out = nn.ModuleList([nn.Linear(512 * block.expansion, num) for num in num_classes])

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return [fc(x) for fc in self.fc_out]


class Predictor:
    def __init__(self, model: ResNetTL, output_keys, device):
        self.model = model
        self.device = device
        self.transform_val = T.Compose(
            [
                T.Resize((240, 120)),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        self.output_keys = ['pred_' + key for key in output_keys]
    
    def __call__(self, images):
        if len(images) == 0:
            return {key: np.array([]) for key in self.output_keys}
        
        if type(images) != torch.Tensor:
            images = [self.transform_val(im.copy()) for im in images]
            images = torch.stack(images, axis=0).to(self.device)
        
        outputs = self.model(images)
        
        output_dict = {}
        for i in range(len(outputs)):
            if outputs[i].shape[1] == 1:
                pred = (torch.sigmoid(outputs[i]) > 0.5).reshape(-1).cpu().numpy()
            else:
                pred = torch.max(outputs[i].data, 1)[1].cpu().numpy()
            output_dict[self.output_keys[i]] = pred
        
        return output_dict
