import numpy as np
import torch
import torch.nn as nn

from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision import transforms as T
import torch.nn.functional as F
import os 
import logging
logger = logging.basicConfig(level=logging.INFO)

class ResNetTL(ResNet):
    def __init__(self, block_str="BasicBlock", layers=[3, 4, 6, 3], num_classes=[8], **kvargs):
        if block_str == "BasicBlock":
            block = BasicBlock
        elif block_str == "Bottleneck":
            block = Bottleneck
        else:
            raise Exception("Use BasicBlock or Bottleneck as resnet block")
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
    def __init__(self, model: ResNetTL, args, device):
        self.model = model
        self.device = device
        self.args = args
        self.transform_val = T.Compose(
            [
                T.Resize((args.train_config.resize_h, args.train_config.resize_w)),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        self.output_keys = ['pred_' + key for key in args.classifier.keys_outputs]
    
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
    
    def create_onnx(self):
        dummy_input = torch.randn(1, 3, self.args.train_config.resize_h, self.args.train_config.resize_w).to(self.device) #TODO chech correct NCHW/NCWH
        save_path = os.path.join(self.args.train_config.weights_path, self.args.config_name, 'checkpoint.onnx')
        torch.onnx.export(self.model,
                  dummy_input,
                  save_path,
                  verbose=False,
                  export_params=True,
                  )

# TensorRT model using torch2trt
class ClassifierNew:
    def __init__(self, path_to_model: str, cfg, batch_size: int=1, name_add='', device: str="cuda", use_trt=True):
        self.device = torch.device(device)
        self.bs = batch_size
        self.cfg = cfg

        logger.info('Loading model...')
        model_data = torch.load(path_to_model)
        self.num_classes = model_data['classifier_config'].num_classes
        self.output_keys = ['pred_' + key for key in model_data['classifier_config'].keys_outputs]
        logger.info('Model loaded.')
        
        self.h, self.w = cfg.train_config.resize_h, cfg.train_config.resize_w 
        block = cfg.classifier.block_str
        layers = cfg.classifier.resnet_layers

        if 'cpu' in self.device.type:
            # Jit
            self.model = ResNetTL(block, layers, self.num_classes)  # BasicBlock - ResNet34
            self.model.load_state_dict(model_data['state_dict'])
            self.model.eval()
            self.model = torch.jit.trace(self.model, torch.ones(batch_size, 3, self.h, self.w).to(self.device), strict=False)
        elif 'cuda' in self.device.type and use_trt:
            # TensorRT
            from torch2trt import torch2trt, TRTModule
            # trt_model_name = "classifier{}_bs{}.trt".format(name_add, batch_size)
            trt_model_name = "{}_{}_bs{}.trt".format("".join(os.path.basename(path_to_model).split('.')[:-1]), name_add, batch_size)
            trt_model_path = os.path.join(os.path.dirname(path_to_model), trt_model_name)
            if os.path.isfile(trt_model_path):
                logger.info("Loading existing TRT weights...")
                self.model = TRTModule()
                self.model.load_state_dict(torch.load(trt_model_path))
                logger.info("The model is loaded.")
            else:
                logger.info("Converting to TensorRT...")
                self.model = ResNetTL(block, layers, self.num_classes)  # BasicBlock - ResNet34
                self.model.load_state_dict(model_data['state_dict'])
                self.model.eval()
                x = torch.ones(1, 3, self.h, self.w).cuda()
                self.model = torch2trt(self.model, [x], fp16_mode=True, max_batch_size=batch_size)
                torch.save(self.model.state_dict(), trt_model_path)
                logger.info("Converted to TensorRT.")
            self.model = self.model.cuda()
            
        elif 'cuda' in self.device.type:
            self.model = ResNetTL(block, layers, self.num_classes)  # BasicBlock - ResNet34
            self.model.load_state_dict(model_data['state_dict'])
            self.model.eval()
            self.model = self.model.cuda()
            
        self.normalize = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def _transform(self, im):
        return self.normalize(F.interpolate(im.permute(2, 0, 1).unsqueeze(0), size=(self.h, self.w)) / 255)

    def __call__(self, images):
        if len(images) == 0:
            return {key: np.array([]) for key in self.output_keys}
        
        outputs = []
        for i in range(int(np.ceil(len(images) / self.bs))):
            in_model = torch.cat([self._transform(im) for im in images[i*self.bs:i*self.bs + self.bs]])
            output = self.model(in_model)
            outputs.append(output)
        
        outputs = [torch.cat([out[i] for out in outputs]) for i in range(len(self.output_keys))]
        
        output_dict = {}
        for i in range(len(outputs)):
            if outputs[i].shape[1] == 1:
                pred = (torch.sigmoid(outputs[i]) > 0.5).reshape(-1).cpu().numpy()
            else:
                pred = torch.max(outputs[i].data, 1)[1].cpu().numpy()
            output_dict[self.output_keys[i]] = pred
        
        return output_dict
