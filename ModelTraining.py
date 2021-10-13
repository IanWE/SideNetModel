from torch import nn
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torch.nn import init
from torch.autograd import Variable
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import warnings
import os
import pandas as pd
from matplotlib import cm
import pickle
import argparse
from scipy.sparse import vstack
import gc
import pickle
from sklearn.model_selection import StratifiedKFold
#from sparselearning.core import add_sparse_args, CosineDecay, Masking

warnings.filterwarnings('ignore')
def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean()#.asscalar()

def minmaxscaler(data,test):
    min = np.amin(data)
    max = np.amax(data)    
    return (data - min)/(max-min),(test-min)/(max-min)

def feature_normalize(data,test):
    mu = np.mean(data)
    std = np.std(data)
    return (data - mu)/std, (test-mu)/std

def to_number(x):
    if x=='BaseLine':
        return [0]
    else:
        lb = []
        for i in x:
            lb.append(int(i))
        return lb

def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv1d') != -1:
        init.xavier_normal_(m.weight.data)
        #init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.constant_(m.weight.data, 1)
        init.constant_(m.bias.data, 0.0)

import random
def data_iter(batch_size, features, labels):
    num_examples = features.shape[0]
    indices = list(range(num_examples))
    random.seed(epoch)
    random.shuffle(indices) 
    for i in range(0, num_examples/batch_size*batch_size, batch_size):
        j = indices[i: min(i + batch_size, num_examples)]
        yield (torch.FloatTensor(features[j]), torch.LongTensor(labels[j]))  

def replace_layers(model, i, indexes, layers):
    if i in indexes:
        return layers[indexes.index(i)]
    return model[i]

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = LR * (0.3 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.8

BASE = "/mnt/sda1/jianwen/sidescan/"
#data/  model/  net_50.pkl
#data/multiple_mx.pkl  model/trainedmodel.pkl

def conv(in_planes,out_planes,kernel_size=8,stride=1):                          
        "3x3 convolution with padding"
        return nn.Conv1d(
                in_planes,
                out_planes,
                kernel_size,
                stride=stride,
                padding=(kernel_size-1)/2,
                bias=False)

class BasicBlock(nn.Module):
        def __init__(self,in_planes,planes,kernel_size,stride=1,downsample=None):
                super(BasicBlock, self).__init__()
                self.conv1 = conv(in_planes,planes,kernel_size,1)
                self.bn1 = nn.BatchNorm1d(planes)
                self.relu = nn.ReLU()
                self.downsample = downsample
                self.stride = 1 
                #
                self.conv2 = conv(planes,planes,kernel_size,1)
                self.bn2 = nn.BatchNorm1d(planes)
                #
                self.conv3 = conv(planes,planes,kernel_size,1)
                self.bn3 = nn.BatchNorm1d(planes)
        def forward(self,x):
                residual = x
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                #
                out = self.conv2(out)
                out = self.bn2(out)
                out = self.relu(out)
                #                                                                                    
                out = self.conv3(out)
                out = self.bn3(out)
        
                if self.downsample is not None:
                    residual = self.downsample(x)  
                out += residual
                out = self.relu(out)
                return out 

def conv(in_planes,out_planes,kernel_size=8,stride=1):                          
        "3x3 convolution with padding"
        return nn.Conv1d(
                in_planes,
                out_planes,
                kernel_size,
                stride=stride,
                padding=(kernel_size-1)/2,
                bias=False)

class BasicBlock(nn.Module):
        def __init__(self,in_planes,planes,kernel_size,stride=1,downsample=None):
                super(BasicBlock, self).__init__()
                self.conv1 = conv(in_planes,planes,kernel_size,1)
                self.bn1 = nn.BatchNorm1d(planes)
                self.relu = nn.ReLU()
                self.downsample = downsample
                self.stride = 1 
                #
                self.conv2 = conv(planes,planes,kernel_size,1)
                self.bn2 = nn.BatchNorm1d(planes)
                #
                self.conv3 = conv(planes,planes,kernel_size,1)
                self.bn3 = nn.BatchNorm1d(planes)
        def forward(self,x):
                residual = x
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                #
                out = self.conv2(out)
                out = self.bn2(out)
                out = self.relu(out)
                #                                                                                    
                out = self.conv3(out)
                out = self.bn3(out)
        
                if self.downsample is not None:
                    residual = self.downsample(x)  
                out += residual
                out = self.relu(out)
                return out 
    
class ResEncoder(nn.Module):
        def __init__(self,block,kernel_size,num_classes=6,in_planes=10):#block means BasicBlock
            self.in_planes = in_planes
            super(ResEncoder,self).__init__()
            self.layer1 = self._make_layer(block,kernel_size[0],64)#128)
            self.layer2 = self._make_layer(block,kernel_size[1],128)#256)
            self.layer3 = self._make_layer(block,kernel_size[2],128)#512)
            
            self.pool = nn.MaxPool1d(2,stride=2)
            #self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(64,num_classes)
            self.softmax = torch.nn.Softmax(dim=1)
            #self.inm = nn.InstanceNorm1d(256)
 
        def _make_layer(self, block, kernel_size, planes, stride=1):
            downsample = None
            if stride != 1 or self.in_planes != planes:
                downsample = nn.Sequential(
                nn.Conv1d(self.in_planes, planes,
                kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
              )
            layers = []
            layers.append(block(self.in_planes,planes,kernel_size,stride,downsample))
            self.in_planes = planes
            #for i in range(1,blocks):
            #       layers.append(block(self.inplanes,planes))
            return nn.Sequential(*layers)
        
        def forward(self,x):
            x = self.layer1(x)
            x = self.pool(x)
            x = self.layer2(x)
            x = self.pool(x)
            x = self.layer3(x) #batch,512,60 
            x = x[:,x.size(1)/2:,:].mul(self.softmax(x[:,:x.size(1)/2,:])) #batch,256,60
            x = x.sum(2)
            x = x.view(x.size(0),-1)
            #x = self.inm(x)
            x = self.fc(x)
            return x

#Model Training
def train(x_train,x_test,y_train,y_test,path):
    epochs = 1500
    batch_size = x_train.shape/10 #you may need to adjust the number
    # Training
    print x_train.shape
    #define network
    net = ResEncoder(BasicBlock,[9,5,3],7,17)
    net = nn.DataParallel(net)
    net = net.cuda()
    net.apply(weights_init) 
    LR = 0.01
    loss_func = torch.nn.CrossEntropyLoss()#loss function
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999))
    net.train()#change to train mode(activate the dropout)
    #train          
    for epoch in range(epochs):  
        sum_loss = 0
        for step,(batch_x,batch_y) in enumerate(data_iter(batch_size,x_train,y)):
            batch_x = Variable(batch_x).cuda()
            batch_y = Variable(batch_y).cuda()
            outputs = net(batch_x)
            loss = loss_func(outputs,batch_y)    
            optimizer.zero_grad()
            loss.backward()                     # calculate the gradients
            sum_loss += loss.item()
            optimizer.step()

            gc.collect()
            if (step+1) % sp == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, (step+1)*batch_size, x_train.shape[0],
                        100*batch_size * (step+1) / x_train.shape[0], sum_loss/sp))
                sum_loss = 0
                net.eval()
                y_pred = F.softmax(net(x_t)).cpu() 
                macro = f1_score(y_test,y_pred.detach().numpy().argmax(axis=1),labels=[0,1,2,3,4,5,6],average='macro')
                acc = accuracy(y_pred.detach().numpy(),y_test)
                net.train()
                if macro>best_accuracy:
                    best_accuracy = macro
                    beat_epoch = epoch
                    torch.save(net,path)
                print 'Train Epoch: {}, Test Accuracy:{}, Macro F1:{}'.format(epoch,acc,macro) 
        print best_accuracy

#Cross Validation
print 'Start..'
dataset,label = pickle.load(open(BASE+'data/multiple_mx.pkl','rb'))
dataset = dataset.swapaxes(1,2).astype('float')#(X,17,60)
#filter
print dataset.shape
for i in range(7):#print the number for different classes
    print label[label==i].shape
skf = StratifiedKFold(n_splits=5, random_state=42)
for n, (train, test) in enumerate(skf.split(dataset,label)):#cross validation
    best_accuracy = 0
    print train, test
    print np.array(dataset)[train].shape, np.array(dataset)[test].shape
    # feature selection
    x_train = np.array(dataset)[train]
    x_t = np.array(dataset)[test]
    x_t = torch.FloatTensor(np.array(dataset)[test]).cuda()
    y = label[train]
    y_test = label[test]
    print 'train:',y.shape,'test:',y_test.shape
    path = BASE+'model/testmodel.pkl'
    #train the model and save it to path
    train(x_train,x_test,y_train,y_test,path)
    print 'Done!'
    #break

#Predict
net = torch.load(BASE+'model/testmodel.pkl')
net.eval()
#x_t.shape is (X,17,60)
result = F.softmax(net(x_t)).cpu().detach().numpy()
y_pred = result.argmax(axis=1)
print y_pred
