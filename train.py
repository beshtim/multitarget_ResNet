import os
import json
import argparse

import torch
import torch.optim
import torch.backends.cudnn as cudnn

from scripts.utils import *
from scripts.Classifier import ResNetTL
from types import SimpleNamespace


def main_worker(args):
    best_f1 = 0

    # create model
    print("=> creating model")
    model = ResNetTL(layers=args.classifier.resnet_layers, num_classes=args.classifier.num_classes)
    model = model.cuda()

    # Data loading code
    transform = get_transform(args, train=True)
    train_loader = get_data_loader(args.data_type)(args, args.data.path_to_train, transform)
    transform = get_transform(args, train=False)
    val_loader = get_data_loader(args.data_type)(args, args.data.path_to_val, transform, shuffle=False)
    
    # define loss function (criterion) and optimizer
    criterion = get_criterion(args, train_loader) 
    
    optimizer = torch.optim.SGD(model.parameters(), args.train_config.learning_rate,
                                momentum=args.train_config.momentum,
                                weight_decay=args.train_config.weight_decay)
    
    # optionally resume from a checkpoint
    if args.train_config.use_pytorch_pretrained_model:
        state_dict = torch.load(args.data.path_to_pytorch_pretrained_model)
        model.load_state_dict(state_dict, strict=False)
    
    if args.train_config.resume:
        if os.path.isfile(args.train_config.resume):
            print("=> loading checkpoint '{}'".format(args.train_config.resume))
            checkpoint = torch.load(args.train_config.resume)
            args.train_config.start_epoch = checkpoint['epoch']
            best_f1 = checkpoint['best_f1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.train_config.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.train_config.resume))
    
    cudnn.benchmark = True
    
    for epoch in range(args.train_config.start_epoch, args.train_config.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)
        
        # evaluate on validation set
        f1_mean = validate(val_loader, model, criterion, args)
        
        # remember best acc@1 and save checkpoint
        if f1_mean > best_f1:
            best_f1 = f1_mean
            
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_f1': best_f1,
                'optimizer': optimizer.state_dict(),
                'classifier_config': args.classifier
            }
            dir_path = os.path.join(args.train_config.weights_path, args.config_name)
            save_checkpoint(state, dir=dir_path)


def main():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--config', default='configs/config.json')
    config_file = parser.parse_args().config
    
    with open(config_file, "r") as f:
        args = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    
    # Simply call main_worker function
    main_worker(args)


if __name__ == '__main__':
    main()
