# -*- coding: utf-8 -*-
import os
import io
import struct
import random
import glob
import csv
import time

import numpy as np
import cv2
	
import sys

import torch
from torch import nn
from torchvision import transforms
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as ticker
from PIL import Image

#変数定義
#5
#TESTDATASET_DIR_IMG   = '../datasets/CompCars-origin/data/image/**/**/**/'
#TESTDATASET_DIR_LBL   = '../datasets/CompCars-origin/data/label/**/**/**/'
#8
TESTDATASET_DIR_IMG   = '../datasets/CompCars/data/image/**/**/**/'
TESTDATASET_DIR_LBL   = '../datasets/CompCars/data/label/**/**/**/'
IMG_TESTDATASET_PATH_IMG  = sorted(glob.glob('{}*.jpg'.format(TESTDATASET_DIR_IMG)))
IMG_TESTDATASET_PATH_LBL  = sorted(glob.glob('{}*.txt'.format(TESTDATASET_DIR_LBL)))

number_of_label = 8
number_of_img = 10000
#MODEL_PATH = '../voe/model/model-{}label.pt'.format(number_of_label)
MODEL_PATH = '../voe/model/model-test-8-1206.pt'
#MODEL_PATH = '../voe/model/model-3ch.pt'


img_width = 113
img_height = 113
INPUT_FEATURES = number_of_label+1
MIDDLE_LAYER = 6
MIDDLE_LAYER2 = 3
OUTPUT_FEATURES = 1

def available_cuda():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'

device = torch.device(available_cuda())

class I_module(nn.Module):
    def __init__(self, in_dim: int, h_dim: int, out_dim: int):
        super(I_module, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_dim, h_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(h_dim, out_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=1),
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        return torch.cat([branch1, branch2], 1)

class F_module(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(F_module, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3),
            nn.ReLU(inplace=True)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        return x

class VoNet(nn.Module):
    def __init__(self):
        super(VoNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            I_module(in_dim=64, h_dim=96, out_dim=128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            F_module(256, 256),
            F_module(256, 384),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            F_module(384, 384),
            F_module(384, 512),
        )
        self.classifier = nn.Sequential(
            nn.Dropout2d(p=0.25),
            nn.Conv2d(512, 1000, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

#-------------------------------------------------------------------------#

def plot(losses):
    # ここからグラフ描画-------------------------------------------------
    # フォントの種類とサイズを設定する。
    """ グラフを描画してpng, svgに書き出す """
    it = range(1, len(losses)+1)
    plt.plot(it, losses, label='Training loss')
    plt.xlabel('iter')
    plt.ylabel('Losses')
    plt.ylim(17.0, 50.0)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(100000)) # iter軸を5刻みに
    # plt.legend()
    plt.savefig('loss(8lbl).svg')
    plt.savefig('loss(8lbl).png')
    plt.show()

# 3層の全結合ニューラルネットワーク
class NeuralNet(nn.Module):
    def __init__(self, in_features, hidden_size1, hidden_size2, out_features):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_size1) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, out_features)  

    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


def sort_node(input):
    if number_of_label == int(8):
        tmp1 = input[0][4]
        input[0][4] = input[0][3]
        input[0][3] = input[0][1]
        input[0][1] = tmp1
        tmp2 = input[0][5]
        input[0][5] = input[0][6]
        input[0][6] = input[0][7]
        input[0][7] = tmp2
        #print("sorted_node[8label]")
    else:
        tmp = input[0][3]
        input[0][3] = input[0][1]
        input[0][1] = input[0][4]
        input[0][4] = tmp
        #print("sorted_node[5label]")
    return input

if __name__ == "__main__":
    model = VoNet()
    model.to(device)
    
    # 学習済みモデルを読み込む
    model.load_state_dict(torch.load(MODEL_PATH))
    #transformの定義(画層のリサイズとテンソル型への変換)
    transform = transforms.Compose([
            transforms.Resize((img_width, img_height)),
            transforms.ToTensor()
        ])
    #transformの定義(テンソル型への変換)
    transform_array = transforms.Compose([
            transforms.ToTensor()
        ])
    
    #print(len(IMG_TESTDATASET_PATH_IMG))
    #test_array = []
    second_input = []
    label_val = []

    #linear_regression用のデータセット作成
    #for n in range(len(IMG_TESTDATASET_PATH_IMG)):
    for n in range(number_of_img):
        IMG_PATH = IMG_TESTDATASET_PATH_IMG[n]
        img = mpimg.imread(IMG_PATH)
        #imgplot = plt.imshow(img)
        #plt.show()
        image = Image.fromarray(img)
        image = transform(image).unsqueeze(0).to(device)
        ##線形回帰の入力値であるvonetの出力値をoutputに得る
        output = model(image)
        _, pred = torch.max(output, 1)
        z = np.int64(pred[0].item())
        #print('[%4d]枚目の予測ラベル（最大値） %1d.' % (n, z))
        #print('[%4d]枚目の正解ラベル（最大値） %1d.' % (n, ))
        #numpy ndarrayに変換して不要な要素を削除
        d = output.device
        output_numpy = output.to('cpu').detach().numpy().copy()
        #output_numpy = np.delete(output_numpy,np.s_[number_of_label::],1)
        output_numpy = np.delete(output_numpy,np.s_[number_of_label+1::],1)
        output_numpy = np.delete(output_numpy,0,1)
        #要素の順番を入れ替える（意味があるか不明）
        output_numpy = sort_node(output_numpy)
        second_input.append(sort_node(output_numpy.tolist()))
        # 実角度データ(label)の読み込み
        with open(IMG_TESTDATASET_PATH_LBL[n]) as f:
            data = f.readlines()[1]
        label_val.append(data)
        #print("読み込んだ実角度：",label_val[n])
    #データの確認
    #conf = 6
    #print("{}枚目の画像のパス：{}".format(conf,IMG_TESTDATASET_PATH_IMG[conf]))
    #print("{}枚目のlabelのパス：{}".format(conf,IMG_TESTDATASET_PATH_LBL[conf]))
    #print("{}枚目のlabelの実角度：{}".format(conf,label_val[conf]))

    #sys.exit()
    #listをndarrayに変換
    second_input = np.array(second_input)
    label_val = np.array(label_val)
    #tensor型に変換
    second_input = torch.tensor(second_input.astype(np.float32)).clone()
    second_input = second_input.to(device)
    second_input1 = torch.reshape(second_input, (-1, number_of_label))

    label_val = torch.tensor(label_val.astype(np.float32)).clone()
    label_val = label_val.to(device)
    label_val = torch.reshape(label_val,(-1, 1))

    x = torch.reshape(torch.ones(number_of_img),(-1,1))
    x = x.to(device)
    
    #8labelのとき
    if(number_of_label == 8):
        second_input = torch.cat((x, second_input1), 1) 
    else:
    #5labelのとき
        #print("xsize:",x)
        #print("second:",second_input)
        second_input = torch.cat((x, second_input1), 1) 
    
    #全結合層に入力（学習を行う）
    #linear_regression_train(second_input,label_val)
    y = label_val

    model = NeuralNet(INPUT_FEATURES, MIDDLE_LAYER, MIDDLE_LAYER2, OUTPUT_FEATURES)
    model.to(device)
    model.train()
    criterion = nn.L1Loss()    
    opt = optim.SGD(model.parameters(), lr=0.005) 
    niter = 150000 * 2 # number of iteration
    losses = []
    for i in range(niter):
        # batch dataの取得                     
        opt.zero_grad()                    
        outputs = model(second_input)
        loss = criterion(outputs.reshape(y.shape), y)
        print("loss:",loss.item())
        # 誤差逆伝播で勾配を更新
        loss.backward()
        opt.step()
        losses.append(loss.item())    #損失値の蓄積
    print("重み：", list(model.parameters()))
    plot(losses)
    


#--------------------------------------------------
"""
def linear_regression_train(x, y):
    model_linear_detection = nn.Linear(INPUT_FEATURES, MIDDLE_LAYER, bias = True)
    model_linear_detection1 = nn.Linear(MIDDLE_LAYER, OUTPUT_FEATURES, bias = True)

    model_linear_detection.to(device)
    model_linear_detection1.to(device)
    opt = optim.SGD(model_linear_detection.parameters(), lr=0.03) 
    opt1 = optim.SGD(model_linear_detection1.parameters(), lr=0.03)                
    criterion = nn.L1Loss()      
    #criterion = nn.MSELoss()      

    niter = 150000 # number of iteration
    losses = []
    for i in range(niter):
        # batch dataの取得                     
        opt.zero_grad()                  
        opt1.zero_grad()                  
        outputs = model_linear_detection(x)
        outputs = nn.functional.relu(outputs)
        outputs = model_linear_detection1(outputs)
        #print("out:",outputs.data[0])
        #print("label:",y.data[0])
        #print("reshapedout:",outputs.reshape(y.shape))
        loss = criterion(outputs.reshape(y.shape), y)
        print("loss:",loss.item())
        # 誤差逆伝播で勾配を更新
        loss.backward()
        opt.step()
        opt1.step()
        losses.append(loss.item())    #損失値の蓄積
    #print("1層",list(model_linear_detection1.parameters()))
    #print("2層",list(model_linear_detection2.parameters()))
    print("1層",list(model_linear_detection.parameters()))
    print("2層",list(model_linear_detection1.parameters()))
    print("loss:",loss.item())
    #print("重み１：",model_linear_detection1.bias.to('cpu').detach().data.numpy().copy()[0])
    #print("重み２：",model_linear_detection2.bias.to('cpu').detach().data.numpy().copy()[0])
    #b = model_linear_detection.bias.to('cpu').detach().data.numpy().copy()[0]
    #b = model_linear_detection.weight.to('cpu').detach().data.numpy()[0, 0].copy()
    #w1 = model_linear_detection.weight.to('cpu').detach().data.numpy()[0, 1].copy()
    #w2 = model_linear_detection.weight.to('cpu').detach().data.numpy()[0, 2].copy()
    #w3 = model_linear_detection.weight.to('cpu').detach().data.numpy()[0, 3].copy()
    #w4 = model_linear_detection.weight.to('cpu').detach().data.numpy()[0, 4].copy()
    #w5 = model_linear_detection.weight.to('cpu').detach().data.numpy()[0, 5].copy()
    #w6 = model_linear_detection.weight.to('cpu').detach().data.numpy()[0, 6].copy()
    #w7 = model_linear_detection.weight.to('cpu').detach().data.numpy()[0, 7].copy()
    #w8 = model_linear_detection.weight.to('cpu').detach().data.numpy()[0, 8].copy()
    #x_new = np.linspace(np.min(x.T[1].data.to('cpu').detach().numpy().copy()), np.max(x.T[1].data.to('cpu').detach().numpy().copy()), len(x))
    #y_curve =  b + (w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8) * x_new
    #print(x.T[1])
    plot(losses)
    #plot2(x.T[1], y, x_new, y_curve, losses)
    """