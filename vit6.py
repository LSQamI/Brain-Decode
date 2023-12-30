import torchvision.models as models
from timm import create_model
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
from wait_gpu import *
import VisionTransformer3D
from mmcls.models.losses import SeesawLoss

def main(config):
    if config.cat =='supercat12':
        class_num = 12
    elif config.cat =='cat80':
        class_num = 80
    if not config.test:
        num = 37*750
    else:
        num = 1*750
    #labelset = np.load('/home/lisq/labels/subj01_labels.npy')
    wait(config.gpu)
    if config.cat == 'supercat12':
        labelset = pd.read_csv('/home/lisq/labels/subj01_labels_45.csv',index_col=0).values[:,2:]
    elif config.cat =='cat80':
        labelset = pd.read_csv('/home/lisq/labels/subj01_labels_80categories.csv',index_col=0).values[:,2:]
    train_loader, val_loader, test_loader, img_shape = data_preparation(labelset, num)
    if config.cat == 'supercat12':
        new_dir = '/SUPERCAT12/'+config.model_name+'_'+config.optimizer+'_btplus'+str(config.beta_plus)+'_bs'+str(
            config.batch_size)+'_lr'+str(config.learning_rate)+'_gm'+str(config.gamma)+'_ep'+str(config.epochs)+'_pt'+str(
            config.pretrained)
    elif config.cat =='cat80':
        new_dir = '/CAT80/'+config.model_name+'_'+config.optimizer+'_btplus'+str(config.beta_plus)+'_bs'+str(
            config.batch_size)+'_lr'+str(config.learning_rate)+'_gm'+str(config.gamma)+'_ep'+str(config.epochs)+'_pt'+str(
            config.pretrained)
        
    #new_dir = new_dir[:-7]

    print(new_dir)
    if (not config.test) and (not os.path.exists('/home/lisq/subj01_records'+new_dir)):
        os.mkdir('/home/lisq/subj01_records'+new_dir)
    parameters = {'optimizer':config.optimizer, 'beta_plus':config.beta_plus,
        'learning_rate':config.learning_rate, 'gamma':config.gamma, 'epochs':config.epochs}
    model = model_preparation(config.model_name, img_shape, class_num)

    if config.training:
        train(parameters, train_loader, model, new_dir, labelset, class_num)
    else:
        model.load_state_dict(torch.load('/home/lisq/subj01_records'+new_dir+'/model.pth'))
    val_accuracy, val_f1, confusion_matrix, FP_matrix, FN_matrix, prediction, probability  = test(val_loader, model)
    test_accuracy, test_f1 = test(test_loader, model)[0:2]
    result_exhibition(val_accuracy, val_f1, confusion_matrix, test_accuracy, test_f1, new_dir,
        FP_matrix, FN_matrix, val_loader.dataset.indices, prediction, probability)


def train(parameters, data_loader, model, new_dir, labelset, class_num):
    if parameters['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=parameters['learning_rate'])
    elif parameters['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=parameters['learning_rate'], betas=(0.9,0.999), amsgrad=False)
    else:
        raise ValueError('Optimizer not found!')
    
    if parameters['gamma'] >= 0:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, parameters['gamma'])

    #criterion = nn.BCELoss(reduction='sum')
    if config.loss == 'bBCE':
        criterion = BalancedBCELoss(labelset[data_loader.dataset.indices, :], parameters['beta_plus'])
    elif config.loss == 'Ss':
        criterion = SeesawLoss(p=0.8, q=2.0, num_classes=class_num, reduction='sum')

    model.train()
    model.cuda()
    train_losses = []
    train_numbers = []
    train_accuracy = []
    train_f1 = []
    counter = 0
    for epoch in range(parameters['epochs']):
        for batch_idx, (data, labels) in enumerate(data_loader):
            #data, labels = data.cuda(), labels.cuda()
            out_layer = nn.Sigmoid()
            output = out_layer(model(data))
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            counter += data.shape[0]
            if batch_idx % int(3000/config.batch_size) == 0:
                result = output.gt(0.5)
                TP = torch.logical_and(labels, result).float().mean(dim=0)
                FP = torch.logical_and(torch.logical_not(labels), result).float().mean(dim=0)
                FN = torch.logical_and(torch.logical_not(result), labels).float().mean(dim=0)
                f1_0 =torch.div(2*TP, 2*TP+FN+FP)
                f1 = torch.where(torch.isnan(f1_0), torch.full_like(f1_0,0.0), f1_0).mean()
                accuracy = torch.eq(labels,result).float().mean()
                print('Train Epoch: {}/{} [{}/{}({:.0f}%)] F1:{:.5f} Acc:{:.5f} Loss:{:.5f}'.format(
                    epoch+1, parameters['epochs'], batch_idx*len(data), len(data_loader.dataset),
                    100.*batch_idx/len(data_loader), f1.item(), accuracy.item(), loss.item()))
                train_losses.append(loss.item())
                train_accuracy.append(accuracy.item())
                train_f1.append(f1.item())
                train_numbers.append(counter)
        if parameters['gamma'] >= 0:
            scheduler.step()
    x_value = [train_numbers[i] / len(data_loader.dataset) for i in range(len(train_numbers))]
    if not config.test:
        torch.save(model.cpu().state_dict(), '/home/lisq/subj01_records'+new_dir+'/model.pth')
        plt.figure()
        plt.plot(x_value, train_f1, label='f1')
        plt.plot(x_value, train_accuracy, label='accuracy')
        plt.legend()
        plt.savefig('/home/lisq/subj01_records'+new_dir+'/train_result.jpg')
        plt.figure()
        plt.plot(x_value, train_losses, label='loss')
        plt.legend()
        plt.savefig('/home/lisq/subj01_records'+new_dir+'/train_loss.jpg')

def test(data_loader, model):
    model.eval()
    model.cuda()
    class_num = data_loader.dataset.dataset.labelset.shape[1]
    smaple_num = len(data_loader.dataset)
    TP = torch.zeros((1,class_num)).cuda()
    FP = torch.zeros((1,class_num)).cuda()
    FN = torch.zeros((1,class_num)).cuda()
    FP_matrix = torch.zeros((smaple_num,class_num)).cuda()
    FN_matrix = torch.zeros((smaple_num,class_num)).cuda()
    TP_matrix = torch.zeros((smaple_num,class_num)).cuda()
    result = torch.zeros((smaple_num,class_num)).cuda()
    probability = torch.zeros((smaple_num,class_num)).cuda()
    counter = 0

    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.cuda(), labels.cuda()
            bs = data.shape[0]
            out_layer = nn.Sigmoid()
            output = out_layer(model(data))
            result[counter:counter+bs,:] = output.gt(0.5)
            probability[counter:counter+bs,:] = output
            FP_matrix[counter:counter+bs,:] = torch.logical_and(torch.logical_not(
                labels),result[counter:counter+bs,:]).float()
            FN_matrix[counter:counter+bs,:] = torch.logical_and(torch.logical_not(
                result[counter:counter+bs,:]), labels).float()
            TP_matrix[counter:counter+bs,:] = torch.logical_and(labels,
                result[counter:counter+bs,:]).float()
            counter += bs
    
    TP = TP_matrix.mean(dim=0)
    FP = FP_matrix.mean(dim=0)
    FN = FN_matrix.mean(dim=0)
    f1_0 =torch.div(2*TP, 2*TP+FN+FP)
    f1 = torch.where(torch.isnan(f1_0), torch.full_like(f1_0,1.0), f1_0).mean()
    accuracy = (1-FP-FN).mean()
    confusion_matrix = np.concatenate((TP.unsqueeze(0).cpu().numpy(), FP.unsqueeze(0).cpu().numpy(), FN.unsqueeze(0).cpu().numpy(),
                                       (1-TP-FP-FN).unsqueeze(0).cpu().numpy()), axis=0)
    return (accuracy, f1, confusion_matrix, FP_matrix.cpu().numpy(), 
        FN_matrix.cpu().numpy(), result.float().cpu().numpy(), probability.cpu().numpy())

def data_preparation(labelset, num):
    dataset = betas('/home/lisq/data', labelset, num)
    train_size = int(0.8*num)
    val_size = int(0.1*num)
    test_size = num - train_size - val_size
    train_set,val_set,test_set = random_split(dataset,[train_size,val_size,test_size],
                                              torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size)
    test_loader = DataLoader(test_set, batch_size=config.batch_size)
    #val_loader = DataLoader(val_set, batch_size=val_size)
    #test_loader = DataLoader(test_set, batch_size=test_size)
    return train_loader, val_loader, test_loader, dataset.get_shape()

def model_preparation(model_name, img_shape, class_num):
    if 'resnet' in model_name:
        model = models.resnet50(pretrained=config.pretrained)
        oc,ks,s,p,b = [model.conv1.out_channels, model.conv1.kernel_size,
            model.conv1.stride, model.conv1.padding, model.conv1.bias]
        model.conv1 = nn.Conv2d(img_shape[0], oc, kernel_size=ks, stride=s, padding=p, bias=b)
        model.fc = nn.Linear(model.fc.in_features, class_num)
    elif 'vit' in model_name:
        if config.three_d:
            model = create_model(model_name, pretrained=config.pretrained, in_chans=1, num_classes=class_num, img_size=img_shape)
        else:
            model = create_model(model_name, pretrained=config.pretrained, in_chans=img_shape[0],num_classes=class_num, img_size=img_shape[1:])
    return model

def result_exhibition(val_accuracy, val_f1, confusion_matrix, test_accuracy, test_f1, new_dir, FP_matrix,
    FN_matrix, indices, prediction, probability):
    print('===========================')
    valtxt = "val F1 score:{:.5f} accuracy:{:.5f}".format(val_f1, val_accuracy)
    testtxt = "test F1 score:{:.5f} accuracy:{:.5f}".format(test_f1, test_accuracy)
    print(valtxt)
    print(testtxt)
    print('confusion matrix:\n'+str(np.round(confusion_matrix,3)))
    if not config.test:
        if config.cat == 'supercat12':
            classes = ['accessory','animal','appliance','electronic','food','furniture','indoor',
                'kitchen','outdoor','person','sports','vehicle']
        elif config.cat =='cat80':
            classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
                       'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
                       'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
                       'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
                       'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
                       'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                       'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
                       'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
                       'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
                       'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        pd.DataFrame(FP_matrix, index=indices, columns=classes).to_csv('/home/lisq/subj01_records'+new_dir+'/FP_matrix.csv')
        pd.DataFrame(FN_matrix, index=indices, columns=classes).to_csv('/home/lisq/subj01_records'+new_dir+'/FN_matrix.csv')
        pd.DataFrame(prediction, index=indices, columns=classes).to_csv('/home/lisq/subj01_records'+new_dir+'/prediction.csv')
        pd.DataFrame(probability, index=indices, columns=classes).to_csv('/home/lisq/subj01_records'+new_dir+'/probability.csv')
        pd.DataFrame(np.round(confusion_matrix,3), index=['TP','FP','FN','TN'], columns=classes).to_csv('/home/lisq/subj01_records'+new_dir+'/confusion_matrix.csv')
        with open('/home/lisq/subj01_records'+new_dir+'/val.txt','a') as f:
            f.truncate(0)
            f.write(valtxt+'\n')
            f.write(testtxt+'\n')

class betas(Dataset):
    def __init__(self, img_path, labelset, num):
        self.img_path = img_path
        self.num = num
        self.labelset = labelset
    def __getitem__(self, i):
        img = np.load(self.img_path+'/subj01/subj01_'+str(i)+'.npy')
        if config.three_d:
            data = torch.from_numpy(img.astype(np.float32)).cuda().unsqueeze(0)
        else:
            data = torch.from_numpy(img.astype(np.float32)).cuda()
        labels = torch.from_numpy(self.labelset[i,:]).cuda()
        return data, labels
    def __len__(self):
        return self.num
    def get_shape(self):
        return np.load(self.img_path+'/subj01/subj01_0.npy').shape

class BalancedBCELoss(nn.Module):
	def __init__(self, train_labels, beta_plus):
		super(BalancedBCELoss, self).__init__()
		self.beta = torch.from_numpy(1 - np.mean(train_labels, axis=0) + beta_plus).cuda()
		#self.beta = torch.from_numpy(1 - np.mean(train_labels, axis=0) + beta_plus)
	def forward(self, input, target):
		input = torch.where(torch.lt(input,1e-6), torch.full_like(input,1e-6), input)
		input = torch.where(torch.gt(input,1-1e-6), torch.full_like(input,1-1e-6), input)
		output = - (self.beta*target*torch.log(input) + (1-self.beta)*(1-target)*torch.log(1-input))
		return output.sum()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--learning_rate', type=float, default=0.0004)
    parser.add_argument('--gamma', type=float, default=0.75)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--beta_plus', type=float, default=0.0)
    parser.add_argument('--model_name', type=str, default='resnet50')
    parser.add_argument('--three_d', type=bool, default=False)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--loss', type=str, default='bBCE')
    parser.add_argument('--cat', type=str, default='supercat12')
    parser.add_argument('--training', type=bool, default=False)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--test', type=bool, default=False)
    config = parser.parse_args()
    main(config)