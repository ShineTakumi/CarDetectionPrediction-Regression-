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

import torch
from torch import nn
from torchvision import transforms
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as ticker
from PIL import Image

#変数定義
#TESTDATASET_DIR_IMG   = '../datasets/OurShots/smartphone/'
TESTDATASET_DIR_IMG   = 'validation/all_detection/'
IMG_TESTDATASET_PATH_IMG  = sorted(glob.glob('{}*.jpg'.format(TESTDATASET_DIR_IMG)))
number_of_label = 8
number_of_img = len(IMG_TESTDATASET_PATH_IMG)

#MODEL_PATH = '../voe/model/model-{}label.pt'.format(number_of_label)
#MODEL_PATH = '../voe/model/model-test-osawa.pt'
MODEL_PATH = '../voe/model/model-test-8-1206.pt'

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
    plt.ylim(0, 100.0)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1)) # epoch軸を5刻みに
    # plt.legend()
    plt.savefig('loss.svg')
    plt.savefig('loss.png')
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
    lblout = []
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
        print('[%4d]枚目の予測ラベル（最大値） %1d.' % (n, z))
        lblout.append(z)
        #print('[%4d]枚目の正解ラベル（最大値） %1d.' % (n, ))
        #numpy ndarrayに変換して不要な要素を削除
        d = output.device
        output_numpy = output.to('cpu').detach().numpy().copy()
        output_numpy = np.delete(output_numpy,np.s_[number_of_label+1::],1)
        output_numpy = np.delete(output_numpy,0,1)
        #output_numpy = np.delete(output_numpy,np.s_[number_of_label::],1)
        print("out-node:",output_numpy)
        #要素の順番を入れ替える（意味があるか不明）
        output_numpy = sort_node(output_numpy)
        second_input.append(sort_node(output_numpy.tolist()))

    #listをndarrayに変換
    second_input = np.array(second_input)
    #print("se:",second_input)
    label_val = np.array(label_val)
    #tensor型に戻す
    second_input = torch.tensor(second_input.astype(np.float32)).clone()
    second_input = second_input.to(device)
    second_input1 = torch.reshape(second_input, (-1, number_of_label))
    label_val = torch.tensor(label_val.astype(np.float32)).clone()
    label_val = label_val.to(device)
    label_val = torch.reshape(label_val,(-1, 1))
    x = torch.reshape(torch.ones(number_of_img),(-1,1))
    x = x.to(device)
    second_input = torch.cat((x, second_input1), 1) 
    #print(second_input.shape)

    #全結合層に入力（学習を行う）
    #output_f = np.array(linear_regression_train(second_input1, second_input,label_val))
    #output_numpy = output_f.to('cpu').detach().numpy().copy()

    y = label_val
    model = NeuralNet(INPUT_FEATURES, MIDDLE_LAYER, MIDDLE_LAYER2, OUTPUT_FEATURES)
    """ 2層のときの重み
    weight_array1 = nn.Parameter(
    torch.tensor([[ 3.3203e+01, -5.2136e-01,  5.3672e-01,  3.0305e+00,  1.2978e+00, -2.3997e+00, -1.9494e+00, -1.5894e+00, -1.4437e+00],
        [ 9.0683e+00,  1.3380e+00,  1.4629e+00, -2.8756e+00, -3.8436e+00,  4.6153e-01, -6.2011e-01, -2.3238e-01,  3.0524e-01],
        [ 3.0536e+01,  1.9427e-02, -1.1055e+00, -1.4567e-01,  1.0621e+00, -2.2244e+00,  2.6687e-01, -2.4251e-01,  5.6940e-01]]))  # 重み
    bias_array1 = nn.Parameter(
    torch.tensor([33.4961,  9.1448, 30.6627]))  # バイアス

    weight_array2 = nn.Parameter(
    torch.tensor([[1.5590, 1.0777, 1.0846]]))  # 重み
    bias_array2 = nn.Parameter(
    torch.tensor([13.1126]))  # バイアス
    """
    #3層のときの重み
    weight_array1 = nn.Parameter(
    torch.tensor([[ 9.6967e+00, -4.4960e-01, -4.7249e-02,  2.4421e-01,  5.9173e-01,
         -2.3507e-01, -1.2295e-01,  3.1057e-01, -3.9005e-01],
        [-3.2963e-01, -2.6641e-01,  1.1607e-01, -1.9628e-01, -2.3467e-01,
          2.2306e-02, -1.5232e-01, -7.5379e-02, -1.7331e-01],
        [ 3.1334e+00, -2.4133e-01, -4.4008e-01, -3.1997e-01, -3.0693e-01,
         -2.3370e-01, -8.6461e-02,  3.4574e-01,  4.4700e-01],
        [ 3.1035e+00,  4.1649e-03, -5.3374e-01, -7.6516e-01, -5.8733e-01,
          3.1129e-01,  1.1899e+00,  2.1133e-01, -7.0595e-01],
        [ 2.7032e-01,  4.5249e-02, -3.0272e-01, -3.2329e-01,  3.9507e-02,
         -2.3818e-01, -3.7337e-02, -2.6677e-01, -3.0078e-01],
        [ 5.8349e+00, -1.7983e+00, -2.5135e+00, -2.4158e+00, -1.5535e+00,
          6.7805e+00, -8.2411e-01, -4.2985e-01, -3.1050e+00]]))  # 重み
    bias_array1 = nn.Parameter(
    torch.tensor([ 9.4580,  0.1100,  3.0993,  2.7163, -0.1760,  5.9090]))  # バイアス

    weight_array2 = nn.Parameter(
    torch.tensor([[ -0.1498,  -0.0459,   0.9988,  -0.0519,   0.3382,  -2.8654],
        [ -0.1188,   0.0930,  -0.9652,  -4.8274,  -0.2330,  -7.5520],
        [  1.2513,  -0.2205, -10.1099,  -7.3269,  -0.1216,  -8.6729]]))  # 重み
    bias_array2 = nn.Parameter(
    torch.tensor([24.9188, 40.1759, 16.3606]))  # バイアス

    weight_array3 = nn.Parameter(
    torch.tensor([[1.9916, 2.1372, 2.6651]]))  # 重み
    bias_array3 = nn.Parameter(
    torch.tensor([-0.0011]))  # バイアス

    # 重みとバイアスの初期値設定
    model.fc1.weight = weight_array1
    model.fc1.bias = bias_array1

    model.fc2.weight = weight_array2
    model.fc2.bias = bias_array2

    model.fc3.weight = weight_array3
    model.fc3.bias = bias_array3

    model.to(device)
    model.eval()
    losses = []
    out_array = []
    for i in range(number_of_img):
        outputs = model(second_input[i])
        out_array.append(outputs.to('cpu').detach().numpy().copy())

    output_f = np.array(out_array)
    print('予測角度：',output_f[:,0])
    print("mean:",np.average(output_f[:,0]))
    print("out:",out_array)

    """
    output_f = np.array(linear_regression_train(second_input1, second_input,label_val))
    print('予測角度：',output_f[:,0])
    print("mean:",np.average(output_f[:,0]))
    #0ラベルのとき
    t = output_f[:,0]
    t1 = output_f[:,0]
    print("lbl:",lblout)
    """

    """
    
def linear_regression_train(x1, x, y):
    model_linear_detection = nn.Linear(INPUT_FEATURES, MIDDLE_LAYER, bias = False)
    model_linear_detection1 = nn.Linear(MIDDLE_LAYER, OUTPUT_FEATURES, bias = False)
    model_linear_detection.eval()
    model_linear_detection1.eval()
    #8labelの重み(1layer)
    #my_weights = torch.tensor([[149.7510,   1.5426,   0.9063,   2.6603,   2.8272,  -4.2685,  -2.1984, -0.6917,  -0.9903]])
    
    #8labelの重み2layer
    my_weights = torch.tensor([[23.2732,  0.8688,  0.6386,  1.9302,  2.1198, -2.3855, -1.5636, -1.4896, -1.7308],
        [28.1596, -0.6389, -1.3214, -1.3883, -1.4186,  1.8440,  1.5645,  0.2057, -1.8042],
        [-0.1127, -0.2982, -0.2576,  0.1726, -0.0384, -0.1573,  0.1286, -0.2989, -0.1908]])
    #my_weights2 = torch.tensor([23.6241, 27.8800,  0.1696])
    my_weights2 = torch.tensor([[ 1.5895, -1.3077, -0.3403]])

    #5labelの重み
    #my_weights = torch.tensor([[ 69.0735,  10.4062, -11.0691,   2.4577,  10.0009, -12.2952]])

    model_linear_detection.weight.data = my_weights
    model_linear_detection1.weight.data = my_weights2
    model_linear_detection.to(device)
    model_linear_detection1.to(device)
    #print("weight:",model_linear_detection.weight)

    criterion = nn.L1Loss()        
    niter = number_of_img # number of iteration
    losses = []
    outputs = []
    for i in range(number_of_img):
        tmp = model_linear_detection(x[i])
        tmp = model_linear_detection(tmp)
        outputs.append(tmp.to('cpu').detach().numpy().copy())
        print("i:{}/output:{}".format(i,tmp))
        #print("out:",outputs)
    print("weight:",model_linear_detection.weight)
    print("weight:",model_linear_detection1.weight)
    return outputs

    """
    