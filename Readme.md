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



### 3. 트레이닝 설정
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
```unet``` 폴더 내부는 skip connection 연결에 따른 실험 ```Skip Connection```, FC/Conv 임베딩 구조 개선 실험 ```Embedding```, 유사도 검출 및 공간 가중합산 실험 ```Spatial_weight```, 축소 계층 실험 ```Flat``` 실험으로 구성되어있음. 이때 pixsim 모델은 Weight 단일 가중합, pixwise 모델은 공간 가중합 모델임. 
```resnet```폴더 내부는 action recognition 정보를 활용하기 위한  인코더 디코더 구조 개선실험이 첨부됨.
각 실험에 대한 결과는 ```Results``` 에 저장되어있으며 함께 있는 python 파일은 실험을 돌리기 위한 main.py이다. 이때, main.py는 다음과 같이 구성되어 있음.

```python
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

실행파일은 dictionary ```input_dic```에서 설정한 옵션에 따라 동작됨. 
다양한 옵션이 있는것을 확인할 수 있지만 실제 최종보고서에서 실험한 세팅과 관련없는 옵션들은 꺼두었음.
```input_dic```과 기타 조정할 수 있는 옵션으로는 다음이 있음.

```input_dic['dataset']``` : 데이터세트 종류 (Humaneva, Charades_v1_480)

```input_dic['frame_interval']``` : 프레임 간격 (defalut=50)

```input_dic['W']```,```input_dic['H']``` : 이미지 크기 (default=64)

```input_dic['vis_port']``` : Visdom 포트 번호 (defalut=7097)

```input_dic['max_iter']``` : 학습 Iteration

```input_dic['val_iter']``` : 평가 Iteration

```input_dic['val_iter']``` : 저장할 Iteration

```input_dic['optim']``` : Optimizer (default=Adam)

```input_dic['criterion']``` : 손실함수 (default=L1)

```Mem_param['embed_size']``` : 메모리네트워크에 저장될 채널 사이즈

#### 기타 옵션

```channels, noskip``` : 인코더/디코더의 각 채널과 스킵커넥션 연결 구성




### 4. 저장된 결과
---
```training``` 각 실험의 ```Results``` 폴더 또는 새롭게 학습시켜 얻은 ```Results```폴더는 다음 구성을 따름. 

(일부 폴더 구성이 다른 경우가 있을 수 있음.)
```bash
Results
  ├──nextframe_fi50_Humaneva # 테스트이름_프레임간격_데이터세트 종류
    ├──Static_Conv_multi_sep1by1_m10_e128_hop1 # 사용한 메모리 네트워크 이름
      ├──unet_conv[3,32,64,128][True, True, True] # 인코더 디코더 네트워크 이름
        ├──L1Loss_e1e-03_d1e-03_W64H64 # 손실함수_러닝레이트_이미지 
          ├──best_loss
            ├──out_image
              ├──0
                ├──*.jpg
              ├──1
              ...
            ├──weight_image
              ├──0
                ├──*.jpg
              ├──1
              ...
          ├──last
            ├──out_image
            ├──weight_image
              ...
          ├──weight
            ├──best_loss.pth
            ├──last.pth
```

```best_loss```는 학습 중간 가장 최적의 validation loss를 달성했을때의 validation 결과를 저장한 것이고, ```last```는 매 ```input_dic['save_iter']```마다 저장한 결과임.
이때 validation 결과로 출력된 결과는 ```out_image```에 저장되며, 각 계층의 weight을 시각화 한 그림은```weight_image```로 저장됨.
