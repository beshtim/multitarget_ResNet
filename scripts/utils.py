import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data

from kornia.augmentation import RandomMotionBlur, RandomPlanckianJitter, RandomAffine, Resize, Normalize
from torchvision import transforms as T
from sklearn.metrics import f1_score as f1_score_metric
from sklearn.utils.class_weight import compute_class_weight
from scripts.Dataloader import CocoAtribDataset, ImFolDataset, CSVDataset

class Handler:
    def __new__(cls, data_type):

        mp = {"IF": IFChooser, "CSV": CSVChooser, "COCO": COCOChooser}

        if data_type not in mp.keys():
            raise NotImplementedError
        
        instance = super().__new__(mp[data_type])

        print(f'DataType is {data_type}')
        return instance

class CSVChooser:
    def get_data_loader(self, args, path_to_csv, transform, shuffle=True): 
    
        categorical_type_to_int = {}

        for key_o in args.classifier.keys_outputs:
            try:
                if key_o in args.classifier.categorical.__dict__.keys():
                    from_type_to_int = {v: k for k, v in enumerate(args.classifier.categorical.__dict__[key_o])}
                    categorical_type_to_int[key_o] = from_type_to_int
            except AttributeError:
                continue
        
        dataset = CSVDataset(
            args.data.path_to_images,
            path_to_csv,
            args.classifier.keys_outputs,
            transform,
            categorical_type_to_int,
            #TODO add need_crop 2 args
        )
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.train_config.batch_size,
            shuffle=shuffle,
            num_workers=args.train_config.workers,
            pin_memory=True
        )
        
        return loader

    def get_criterion_weights(self, args, train_loader): 
        if not args.train_config.use_criterion_weights:
            return [None] * len(args.classifier.num_classes)
        
        data = []
        for ann in train_loader.dataset.data:
            attributes = [int(ann['attributes'][key]) for key in train_loader.dataset.keys_outputs]
            data.append(attributes)
        data = np.asanyarray(data).T
        
        weights = []
        for i in range(len(data)):
            class_weight = compute_class_weight('balanced', classes=np.unique(data[i]), y=data[i])
            if args.classifier.num_classes[i] == 1:
                class_weight = class_weight[1] / class_weight[0]
            weights.append(torch.tensor(class_weight))
        
        return weights
        pass

class CriterionMixin:
    def get_criterion(self, args, train_loader):
        criterion = []
        weights = self.get_criterion_weights(args, train_loader)
        
        for i in range(len(args.classifier.num_classes)):
            if args.classifier.num_classes[i] > 1:
                criterion.append(nn.CrossEntropyLoss(weights[i]))
            else:
                criterion.append(nn.BCEWithLogitsLoss(pos_weight=weights[i]))
        
        return criterion

class IFChooser(CriterionMixin):
    def get_data_loader(self, args, path_2_data, transform, shuffle=True):
    
        dataset = ImFolDataset(
            path_2_data,
            transform
        )
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.train_config.batch_size,
            shuffle=shuffle,
            num_workers=args.train_config.workers,
            pin_memory=True
        )
        
        return loader

    def get_criterion_weights(self, args, train_loader):
        if not args.train_config.use_criterion_weights:
            return [None] * len(args.classifier.num_classes)
        
        data = []
        for fp in train_loader.dataset.files:
            attributes = [os.path.basename(os.path.dirname(fp))]
            data.append(attributes)
        data = np.asanyarray(data).T
        
        weights = []
        for i in range(len(data)):
            class_weight = compute_class_weight('balanced', classes=np.unique(data[i]), y=data[i])
            if args.classifier.num_classes[i] == 1:
                class_weight = class_weight[1] / class_weight[0]
            weights.append(torch.tensor(class_weight))
        
        return weights

class COCOChooser(CriterionMixin):
    def get_data_loader(self, args, path_to_json, transform, shuffle=True):
        
        categorical_type_to_int = {}

        for key_o in args.classifier.keys_outputs:
            try:
                if key_o in args.classifier.categorical.__dict__.keys():
                    from_type_to_int = {v: k for k, v in enumerate(args.classifier.categorical.__dict__[key_o])}
                    categorical_type_to_int[key_o] = from_type_to_int
            except AttributeError:
                continue
        
        dataset = CocoAtribDataset(
            args.data.path_to_images,
            path_to_json,
            args.classifier.keys_outputs,
            transform,
            categorical_type_to_int
        )
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.train_config.batch_size,
            shuffle=shuffle,
            num_workers=args.train_config.workers,
            pin_memory=True
        )
        
        return loader
    
    def get_criterion_weights(self, args, train_loader):
        if not args.train_config.use_criterion_weights:
            return [None] * len(args.classifier.num_classes)
        
        data = []
        for ann in train_loader.dataset.annotations:
            attributes = [int(ann['attributes'][key]) for key in train_loader.dataset.keys_outputs]
            data.append(attributes)
        data = np.asanyarray(data).T
        
        weights = []
        for i in range(len(data)):
            class_weight = compute_class_weight('balanced', classes=np.unique(data[i]), y=data[i])
            if args.classifier.num_classes[i] == 1:
                class_weight = class_weight[1] / class_weight[0]
            weights.append(torch.tensor(class_weight))
        
        return weights

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = [AverageMeter(f'Acc[{t}]@1', ':6.2f') for t in args.classifier.keys_outputs]
    f1 = [AverageMeter(f'F1[{t}]', ':6.2f') for t in args.classifier.keys_outputs]
    progress = ProgressMeter(len(train_loader), losses, *top1, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        images = images.cuda(non_blocking=True)
        
        # compute output
        output = model(images)
        
        loss = 0
        for j in range(len(target)):
            if output[j].shape[1] == 1:
                loss += criterion[j](output[j].cpu().reshape(-1), target[j].float())
            else:
                loss += criterion[j](output[j].cpu().double(), target[j])
        
        # measure accuracy and record loss
        losses.update(loss.item(), images.size(0))
        for j in range(len(target)):
            acc, _ = accuracy(output[j], target[j])
            top1[j].update(acc, images.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.train_config.print_freq == 0:
            progress.print(i)

def validate(val_loader, model, criterion, args):
    print('val')
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = [AverageMeter(f'Acc[{t}]@1', ':6.2f') for t in args.classifier.keys_outputs]
    f1 = [AverageMeter(f'F1-score[{t}]', ':6.2f') for t in args.classifier.keys_outputs]
    progress = ProgressMeter(len(val_loader), batch_time, losses, *top1, prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)

            # compute output
            output = model(images)
            
            loss = 0
            for j in range(len(target)):
                if output[j].shape[1] == 1:
                    loss += criterion[j](output[j].cpu().reshape(-1), target[j].float())
                else:
                    loss += criterion[j](output[j].cpu().double(), target[j])

            # measure accuracy and record loss
            losses.update(loss.item(), images.size(0))
            for j in range(len(target)):
                acc, f1_score = accuracy(output[j], target[j])
                top1[j].update(acc, images.size(0))
                f1[j].update(f1_score, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.train_config.print_freq == 0:
                progress.print(i)

        acc = [f' * Acc[{args.classifier.keys_outputs[j]}]@1 {top1[j].avg:.3f}' for j in range(len(args.classifier.keys_outputs))]
        print('\n'.join(acc))
        acc = [f' * F1-score[{args.classifier.keys_outputs[j]}] {f1[j].avg:.3f}' for j in range(len(args.classifier.keys_outputs))]
        print('\n'.join(acc))
    
    return sum([t.avg for t in f1]) / len(f1)

def save_checkpoint(state, filename='checkpoint.pth', dir='weights'):
    os.makedirs(dir, exist_ok=True)
    path = os.path.join(dir, filename)
    torch.save(state, path)

def do_nothing(input_img, params=None, transform=None):
    return input_img

def enable_if(condition, obj):
    return obj if condition else do_nothing

def get_transform(args, train=False):
    transform = [
        enable_if(args.train_config.transform_train.use_motion_blur, RandomMotionBlur(3, 35., 0.5, p=0.5)),
        enable_if(args.train_config.transform_train.use_planckian_jitter, RandomPlanckianJitter(mode='CIED')),
        enable_if(args.train_config.transform_train.use_random_affine, RandomAffine((-15., 15.), p=0.3)), 
    ] if train else []
    
    resize = Resize((args.train_config.resize_h, args.train_config.resize_w))
    normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    output = [T.ToTensor()] + transform + [resize, normalize]
    
    return T.Compose(output)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.train_config.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    output = output.cpu()
    target = target.cpu()
    
    with torch.no_grad():
        if output.shape[1] == 1:
            output_sigmoid = torch.sigmoid(output)
            output_pred = (output_sigmoid > 0.5).to(int).reshape(-1)
            acc = (output_pred == target).sum() / target.shape[0] * 100
            f1_score = f1_score_metric(target, output_pred, average='binary')
        else:
            output_pred = torch.argmax(output, axis=1)
            acc = (output_pred == target).sum() / target.shape[0] * 100
            f1_score = f1_score_metric(target, output_pred, average='weighted')
        
        return acc, f1_score
