import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu = '--gpu 0' 

train_cfg = ' --epoch 400 --lr 1e-4 --save-iter-step 100 --log-per-iter 10 '
data_sirstaug = ' --dataset sirstaug '
data_irstd1k = ' --dataset irstd1k '
data_nudt = ' --dataset nudt '
SirstDataset = ' --dataset SIRSTv1 '

for i in range(1):
    os.system('python train.py --net-name scdenet --batch-size 8 --base-dir train_logs' + train_cfg + data_irstd1k + gpu)
