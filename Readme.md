# 동영상 복원 네트워크

### 1. 코드구성
---
```bash
video_restoration
  ├──video_module
    ├──Memory_Module
    ├──Network_Module
    ├──training
    ├──jutils, Total_Datset, ..etc
```
Memory_Module   : 메모리 네트워크 구성하기 위한 모듈, Unet과 Resnet을 위한 메모리가 구분되어 있음. 

Network_Module  : Encoder과 Decoder 네트워크를 구성하기 위한 모듈, Unet과 Resnet을 위한 메모리가 구분되어 있음.

training        : 학습을 돌리기 위한 파일와 결과를 첨부했음. 결과는 Results 내부에 저장되어있음.




### 2. 데이터세트 설정
---

#### Charades_v1_480_loader
```bash
Charades_v1_480_img
  ├──0A8CF
    ├──0000000.jpg
    ├──0000001.jpg
    ...
  ├──0A8ZT
  ...
```
#### HumanEva
```bash
HumanEva
  ├──image
    ├──S1_Box_1_(C1)
      ├──000000.jpg
      ├──000001.jpg
    ├──S1_Box_1_(C2)
    ...
  ├──video
    ├──S1
      ├──Box_1_(C1).avi
      ├──Box_1_(C2).avi
      ...
    ├──S2
    ...
  ├──test_list.csv, train_list.csv, val_list.csv
```
폴더는 위 구성을 따름. 
Charades 데이터 세트는 ```Charades_v1_480_img``` 폴더 안에 각 동영상 클립 이름으로 폴더로 구성하며 각 프레임은 ```000000.jpg```로 시작함.
HumanEva 데이터 세트는 ```HumanEva``` 안에 ```image```,```video``` 폴더로 구성되며 각 jpg와 avi 파일로 구성됨.

폴더를 구성하고 Charades와 HumanEva 데이터세트의 경로를 반영하기 위해서는 ```video_restoration/video_module/Total_Train_*.py``` 파일에서 loader의 ```root_dir```을 수정하면 경로를 인식할 수 있음.



### 트레이닝 설정
---
```bash
training
  ├──resnet
    ├──pixsim_1by1
      ├──Results
      ├──main.py
    ├──pixsim_1by1_flat2d_stride1
    ├──pixsim_1by1_flat2d_stride2
  ├──unet
    ├──Skipconnection
    ├──Embedding
    ├──Spatial_weight
    ├──Flat
```
```unet``` 폴더 내부는 skip connection 연결에 따른 실험 ```Skip Connection```, FC/Conv 임베딩 구조 개선 실험 ```Embedding```, 유사도 검출 및 공간 가중합산 실험 ```Spatial_weight```, 축소 계층 실험 ```Flat``` 실험으로 구성되어있다.

```resnet```폴더 내부는 action recognition 정보를 활용하기 위한  인코더 디코더 구조 개선실험이 첨부되어있다.

각 실험에 대한 결과는 ```Results``` 에 저장되어있으며 함께 있는 python 파일은 실험을 돌리기 위한 main.py이다. 이때, main.py는 다음과 같이 구성되어 있다.

```Python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #GPU

# Define Test Param
input_dic={}
input_dic['test_type'] = 'nextframe' # 'nextframe', 'subtract', 'opticalflow'
input_dic['wsupervised'] = 0
input_dic['QSsep'] = False
input_dic['load'] = False

# Dataset
input_dic['dataset'] = 'Humaneva'
input_dic['frame_interval'] = 50
input_dic['random_interval'] = False
input_dic['batch_size'] = 1
input_dic['W'] = 64
input_dic['H'] = 64
input_dic['input_size'] = 10 # Senetences size

# Enc / Dec
from video_module.Network_Module.Unet.Unet_conv import Unet_Up
from video_module.Network_Module.Unet.Unet_conv import Unet_Down
channels = [3, 32, 64, 128]
noskip = [True,True,True]
input_dic['encoder'] = Unet_Down(channels=channels, skip_ind=noskip)
input_dic['decoder'] = Unet_Up(channels=channels, skip_ind=noskip)

# Memory
from video_module.Memory_Module.Static.unet.Static_Conv_multi_sep1by1_pixsim import mem
Mem_param={}
Mem_param['hops'] = 1
Mem_param['input_size'] = input_dic['input_size'] # Senetences size
Mem_param['embed_size'] = 128 # Unet/Resnet's last dimension
Mem_param['memory_size']  = 10 # Memory size
Mem_param['pad']  = 3
Mem_param['noskip'] = noskip
Mem_param['channels'] = channels
input_dic['mem_net'] = mem(Mem_param).cuda()

# Visdom
input_dic['vis_port'] = 7097
# Training
input_dic['max_iter'] = 500000 #iter
input_dic['val_iter'] = 100 #iter
input_dic['save_iter'] = 1000 #iter


# Define optimizer/lr/loss
import torch
optimList = [{"params": input_dic['encoder'].parameters(), "lr": 1e-3},
             {"params": input_dic['decoder'].parameters(), "lr": 1e-3},
             {"params": input_dic['mem_net'].parameters(), "lr": 1e-3}]
input_dic['optim'] = torch.optim.Adam(optimList)#,weight_decay=1e-5)
input_dic['criterion'] = torch.nn.L1Loss()

# Start Training
from video_module.Total_Prepare import Prepare
from video_module.Total_Train_unet import training
input_param = Prepare(input_dic)
training(input_dic)

```
실행파일은 ```input_dic``` dictionary 설정한 옵션에 따라 동작된다. 연구를 수행하면서 다양한 옵션에 따라 학습법을 달리해보았으나 최종보고서에 기재된 내용과 무관한 


#### 저장된 결과
---
### - Example 1
#### - Make a function that sums from 1 to N and output the result.
```C++
#include<stdio.h>
int sum(int n);

void main() {
	int n;
	printf("1부터 n까지의 합 계산 \n");
	printf("정수 n 입력 :");
	scanf_s("%d", &n);
	printf("1부터 %d까지의 합:%d \n", n, sum(n));
}
int sum(int n) {
	int a = 0;
	for (int i = 1; i <= n; i += 1) {
		a += i;
	}
	return a;
}
```
