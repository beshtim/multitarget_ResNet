import os
import json
import torch
import argparse

from types import SimpleNamespace
from scripts.Classifier import ResNetTL, Predictor
from scripts.TensorRT.build_engine import EngineBuilder
                
def main():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--config', default='configs/config.json')
    config_file = parser.parse_args().config
    
    with open(config_file, "r") as f:
        args = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    
    model_path = os.path.join(args.train_config.weights_path, args.config_name, 'checkpoint.pth')
    state_dict = torch.load(model_path)['state_dict']
    
    model = ResNetTL(num_classes=args.classifier.num_classes).cuda()
    model.load_state_dict(state_dict)
    model.eval()
    net = Predictor(model, args, device='cuda')
    net.create_onnx()
    
    onnx_path = os.path.join(args.train_config.weights_path, args.config_name, 'checkpoint.onnx')
    
    builder = EngineBuilder(True, 8)
    builder.create_network(onnx_path)
    
    save_path = os.path.join(args.train_config.weights_path, args.config_name, 'TRT_{}.engine'.format(args.trt_precision))
    if args.trt_precision in ['fp16', 'fp32']:
        builder.create_engine(save_path, args.trt_precision)
    elif args.trt_precision == "int8":
        print("Not implemented correctly due to calibration files. Free to uncomment and fix scripts/image_batcher.py")
        # builder.create_engine(save_path, "int8", "data/images", "weights/calibration.cache", 256, 10)
    else: 
        raise Exception('Not a valid trt_precision')

    


if __name__ == '__main__':
    main()
