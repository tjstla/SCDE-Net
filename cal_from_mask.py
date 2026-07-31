import torch.utils.data as Data

import os
import os.path as osp
import scipy.io as scio
import numpy as np

import cv2

from utils.data import resize_pair
from utils.evaluation.roc_cruve import ROCMetric
from utils.evaluation.TPFNFP import SegmentationMetricTPFNFP
from utils.evaluation.pd_fa import PD_FA


MAT_DATASETS = {
    'NUDT-SIRST': ('./result/NUDT-SIRST/mat1', './datasets/NUDT-SIRST/test/masks'),
    'IRSTD-1k': ('./result/IRSTD-1k/mat1', './datasets/IRSTD-1k/test/masks'),
    'SIRST-aug': ('./result/sirst_aug/mat1', './datasets/sirst_aug/test/masks'),
    'SIRSTv1': ('./result/SIRSTv1/mat1', './datasets/SIRSTv1/SIRST/BinaryMask'),
}


def log_line(f, message):
    print(message)
    f.write(message + '\n')


class Dataset_mat(Data.Dataset):
    def __init__(self, dataset, base_size=256, thre=0.):

        self.base_size = base_size
        self.dataset = dataset
        if dataset not in MAT_DATASETS:
            raise NotImplementedError
        self.mat_dir, self.mask_dir = MAT_DATASETS[dataset]

        file_mat_names = os.listdir(self.mat_dir)
        self.file_names = [s[:-4] for s in file_mat_names]

        self.thre = thre

    def __getitem__(self, i):
        name = self.file_names[i]
        if self.dataset == 'SIRSTv1':
            mask_path = osp.join(self.mask_dir, name) + "_pixels0.png"
        else:
            mask_path = osp.join(self.mask_dir, name) + ".png"
        mat_path = osp.join(self.mat_dir, name) + ".mat"

        rstImg = scio.loadmat(mat_path)['T']
        rstImg = np.asarray(rstImg)

        rst_seg = np.zeros(rstImg.shape)
        rst_seg[rstImg > self.thre] = 1

        mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), -1)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask / mask.max()

        return resize_pair(rstImg, mask, self.base_size)

    def __len__(self):
        return len(self.file_names)


def cal_fpr_tpr(dataname, nbins=200, fileName=None):
    f = open(fileName, mode='a+')
    log_line(f, 'Running data: {:s}'.format(dataname))

    thre = 0.5

    baseSize = 256
    dataset = Dataset_mat(dataname, base_size=baseSize, thre=thre)

    roc = ROCMetric(bins=200)
    eval_PD_FA = PD_FA()
    eval_mIoU_P_R_F = SegmentationMetricTPFNFP(nclass=1)

    for i in range(dataset.__len__()):
        rstImg, mask = dataset.__getitem__(i)
        size = rstImg.shape
        roc.update(pred=rstImg, label=mask)
        eval_PD_FA.update(rstImg, mask, size)
        eval_mIoU_P_R_F.update(labels=mask, preds=rstImg)

    fpr, tpr, auc = roc.get()
    pd, fa = eval_PD_FA.get()
    miou, prec, recall, fscore = eval_mIoU_P_R_F.get()

    log_line(f, 'AUC: %.6f' % (auc))
    log_line(f, 'Pd: %.6f, Fa: %.8f' % (pd, fa))
    log_line(f, 'mIoU: %.6f, Prec: %.6f, Recall: %.6f, fscore: %.6f' % (miou, prec, recall, fscore))
    f.write('\n')

    save_dict = {'tpr': tpr, 'fpr': fpr, 'Our Pd': pd, 'Our Fa': fa}
    matDir = './eval/IndicatorResult/matResult/'
    os.makedirs(matDir, exist_ok=True)
    matFile = osp.join(matDir, '{:s}.mat'.format(dataname))
    scio.savemat(matFile, save_dict)


if __name__ == '__main__':
    specific = True
    data_list = ['NUDT-SIRST', ]  # 'IRSTD-1k', 'SIRST-aug', 'NUDT-SIRST' SIRSTv1

    fileDir = './eval/IndicatorResult/txtResult/'

    for data in data_list:
        fileName = fileDir + f'{data}_mat_result.txt'
        os.makedirs(fileDir, exist_ok=True)

        with open(fileName, mode='w+') as f:
            pass

        cal_fpr_tpr(dataname=data, nbins=200, fileName=fileName)