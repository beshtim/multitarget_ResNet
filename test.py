import os
import json
import torch
import argparse
import numpy as np
import pandas as pd

from tabulate import tabulate
from types import SimpleNamespace
from scripts.Classifier import ResNetTL, Predictor, ClassifierNew
from scripts.utils import get_data_loader, get_transform


def save_data(statistics, confusion_matrix, keys_outputs, path_to_output):
    writer = pd.ExcelWriter(path_to_output)
    workbook = writer.book
    
    header_format = workbook.add_format({
                        'bold': True,
                        'text_wrap': True,
                        'align': 'center',
                        'valign': 'vcenter',
                        'border': 1})
    
    other_format = workbook.add_format({
                        'font_size': 14,
                        'align': 'center',
                        'valign': 'vcenter'})
    
    other_format_rotated = workbook.add_format({
                                'font_size': 14,
                                'align': 'center',
                                'valign': 'vcenter',
                                'rotation': 90})
    
    def adjust_and_write(df, sheet_name, startrow=0, startcol=0, cm=False):
        df.to_excel(writer, 
                    sheet_name=sheet_name, 
                    startrow=startrow, 
                    startcol=startcol)
        
        sheet = writer.sheets[sheet_name]
        
        # Auto-adjust columns' width
        for column in df:
            max_width = 0
            col_name = str(column).split('\n')
            for string in col_name:
                cur_width = len(string)
                if cur_width > max_width:
                    max_width = cur_width
            
            column_width = max(df[column].astype(str).map(len).max(), max_width)
            col_idx = df.columns.get_loc(column)
            sheet.set_column(startcol+col_idx+1, startcol+col_idx+1, column_width+1)
            
        # Center cells
        for col_idx, col_val in enumerate(df.columns.values):
            for row_idx, row_val in enumerate(df.index):
                value = df[col_val][row_val]
                sheet.write(startrow+row_idx+1, startcol+col_idx+1, value, other_format)
            
        # Center headers
        for col_idx, value in enumerate(df.columns.values):
            sheet.write(startrow, startcol+col_idx+1, value, header_format)
            
        # Center indexes
        max_width = 0
        for i in df.index:
            cur_width = len(str(i))
            if cur_width > max_width:
                max_width = cur_width
        sheet.set_column(startcol, startcol, max_width+1)
        
        if cm:
            sheet.merge_range(first_row=0, first_col=2, 
                              last_row=0, last_col=1+len(df.columns.values), 
                              data='prediction', cell_format=other_format)
            sheet.merge_range(first_row=2, first_col=0, 
                              last_row=1+len(df.columns.values), last_col=0, 
                              data='gt', cell_format=other_format_rotated)
            
            sheet.set_column(startcol-1, startcol-1, 2)
    
    list_of_dfs = statistics + confusion_matrix
    
    CM = []
    list_of_corresponding_names = []
    for key in keys_outputs:
        list_of_corresponding_names.append(key)
        CM.append(False)
    for key in keys_outputs:
        list_of_corresponding_names.append('CM_' + key)
        CM.append(True)
    
    for i in range(len(list_of_dfs)):
        if CM[i]:
            startrow=1
            startcol=1
        else:
            startrow=0
            startcol=0
            
        adjust_and_write(df=list_of_dfs[i], 
                         sheet_name=list_of_corresponding_names[i],
                         startrow=startrow, 
                         startcol=startcol,
                         cm=CM[i])
        
    writer.save()

def test(model_predictor, args):
    ''' PARAMS '''
    # path_to_test_coco_json = args.data.path_to_test_json
    path_to_output = os.path.join(args.data.path_to_test_result_output_folder,
                                  args.config_name + '.xlsx')
    
    transform = get_transform(args, train=False)
    
    device = torch.device('cuda')
    
    num_classes = args.classifier.num_classes
    keys_outputs = args.classifier.keys_outputs

    categorical_names = {}

    for key_o in keys_outputs:
        try:
            if key_o in args.classifier.categorical.__dict__.keys():
                types_names = list(args.classifier.categorical.__dict__[key_o])
                categorical_names[key_o] = types_names
        except AttributeError:
            continue

    ''''''
    
    testloader = get_data_loader(args.data_type)(args, args.data.path_to_test, transform, shuffle=False)
    
    # init cols names for tables
    total_col = 'total'
    correct_col = 'correct'
    accuracy_col = 'accuracy \n(correct/all), %'
    precision_col = 'precision'
    recall_col = 'recall'
    f1_score_col = 'f1'
    
    cols = [correct_col, total_col, accuracy_col, precision_col, recall_col, f1_score_col]
    
    # init tables for saving statistics
    statistics = []
    confusion_matrix = []
    for i, num in enumerate(num_classes):
        if num != 1:
            if keys_outputs[i] in categorical_names:
                index = categorical_names[keys_outputs[i]]
            else:
                index = list(range(num))
            df = pd.DataFrame(0, index=index + ['Sum'], columns=cols)
            statistics.append(df)
            
            df = pd.DataFrame(0, index=index, columns=index)
            confusion_matrix.append(df)
        else:
            df = pd.DataFrame(0, index=[keys_outputs[i]], columns=cols)
            statistics.append(df)
            
            df = pd.DataFrame(0, index=[0, 1], columns=[0, 1])
            confusion_matrix.append(df)
    
    with torch.no_grad():
        for i, data in enumerate(testloader):
            inputs, labels = data
            inputs = inputs.to(device)
            
            predictions = model_predictor(inputs)
            
            for j, key in enumerate(keys_outputs):
                gt = labels[j].tolist()
                pred = predictions['pred_' + key].astype(int).tolist()
                
                for label, prediction in zip(gt, pred):
                    confusion_matrix[j].iloc[label][prediction] += 1
                    
                    if label == prediction:
                        if num_classes[j] == 1:
                            statistics[j][correct_col] += 1
                        else:
                            statistics[j][correct_col].iloc[label] += 1
                    
                    if num_classes[j] == 1:
                        statistics[j][total_col] += 1
                    else:
                        statistics[j][total_col].iloc[label] += 1
            
            print(f'\r[{i+1}/{len(testloader)}]', end='')

    # 'Sum'
    for i, num in enumerate(num_classes):
        if num != 1:
            statistics[i].iloc[-1] = statistics[i].sum()
    
    # 'Accuracy'
    for i, num in enumerate(num_classes):
        statistics[i][accuracy_col] = np.around((100 * statistics[i][correct_col] / 
                                                 statistics[i][total_col]), 2)
        
        statistics[i][accuracy_col] = statistics[i][accuracy_col].fillna(100)
    
    # 'precision', 'recall', 'f1'
    for i, num in enumerate(num_classes):
        numerator = np.diag(confusion_matrix[i])
        denominator = confusion_matrix[i].sum()
        precision = numerator / denominator
        
        denominator = confusion_matrix[i].sum(axis=1)
        recall = numerator / denominator
        
        f1 = 2 * (precision * recall) / (precision + recall)
        
        if num != 1:
            statistics[i][precision_col] = precision
            statistics[i][recall_col] = recall
            statistics[i][f1_score_col] = f1
        else:
            statistics[i][precision_col] = precision[1]
            statistics[i][recall_col] = recall[1]
            statistics[i][f1_score_col] = f1[1]
        statistics[i] = statistics[i].fillna(0).round(2)
    
    # Print
    print()
    for i, key in enumerate(keys_outputs):
        print(key)
        print(tabulate(statistics[i], headers='keys', tablefmt='psql'))
        print(tabulate(confusion_matrix[i], headers='keys', tablefmt='psql'))
    
    # Save
    save_data(statistics, confusion_matrix, keys_outputs, path_to_output)


def main():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--config', default='configs/config.json')
    parser.add_argument('-trt', '--torch2trt', action='store_true', help="Use torch2trt for testing")
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-na', '--name_add', default='')

    argprs = parser.parse_args()

    config_file = argprs.config
    trt = argprs.torch2trt
    d = argprs.device
    na = argprs.name_add

    with open(config_file, "r") as f:
        args = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    
    model_path = os.path.join(args.train_config.weights_path, args.config_name, 'checkpoint.pth')
    state_dict = torch.load(model_path)['state_dict']
    
    model = ResNetTL(num_classes=args.classifier.num_classes).cuda()
    model.load_state_dict(state_dict)
    model.eval()
    net = Predictor(model, args, device='cuda')

    # net = ClassifierNew(path_to_model=model_path, cfg=args, batch_size=1, name_add=na, device=d, use_trt=trt) #TODO need check
    
    test(model_predictor=net, args=args)


if __name__ == '__main__':
    main()
