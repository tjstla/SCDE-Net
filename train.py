import os
import sys
import os.path as osp
import time
import datetime
import shutil
from argparse import ArgumentParser

import numpy as np
import torch
import torch.utils.data as Data
from tensorboardX import SummaryWriter
# from tqdm import tqdm
# from utils.loss import HybridLoss
from utils.loss import SoftLoULoss, HybridLoss, SLSIoULoss
from models import get_model
from utils.data import *
from utils.lr_scheduler import *
from utils.evaluation.my_pd_fa import my_PD_FA
from utils.evaluation.TPFNFP import SegmentationMetricTPFNFP
from utils.logger import setup_logger


def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Implement of RPCANet')

    #
    # Dataset parameters
    #
    parser.add_argument('--base-size', type=int, default=256, help='base size of images')
    parser.add_argument('--crop-size', type=int, default=256, help='crop size of images')
    parser.add_argument('--dataset', type=str, default='SirstDataset', help='choose datasets')

    #
    # Training parameters
    #

    parser.add_argument('--batch-size', type=int, default=8, help='batch size for training')
    parser.add_argument('--epochs', '--epoch', dest='epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--warm-up-epochs', type=int, default=0, help='warm up epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='GPU number')
    parser.add_argument('--seed', type=int, default=1, help='seed') # 42 1
    parser.add_argument('--lr-scheduler', type=str, default='poly', help='learning rate scheduler')

    parser.add_argument('--seg-loss', type=str, default='softiou', choices=['softiou', 'hybrid', 'sls'],
                        help='segmentation loss: softiou | hybrid | sls')
    parser.add_argument('--lambda-iou', type=float, default=0.5, help='HybridLoss weight for SoftIoU')
    parser.add_argument('--lambda-bce', type=float, default=0.5, help='HybridLoss weight for BCE')
    parser.add_argument('--sls-with-shape', action='store_true', default=False,
                        help='enable shape term (LLoss) in SLSIoULoss')

    #
    # Net parameters
    #
    parser.add_argument('--net-name', type=str, default='rpcanet',
                        help='net name: fcn')
    # Rank parameters
    #
    # parser.add_argument('--rank', type=int, default=8,
    #                     help='rank number')

    #
    # Save parameters
    #
    parser.add_argument('--save-iter-step', type=int, default=1, help='save model per step iters')
    parser.add_argument('--log-per-iter', type=int, default=1, help='interval of logging')
    parser.add_argument('--base-dir', type=str, default='./train_logs/', help='saving dir')

    args = parser.parse_args()

    # Save folders
    #args.base_dir = r'D:\WFY\dun_irstd\result'
    args.time_name = time.strftime('%Y%m%dT%H-%M-%S', time.localtime(time.time()))
    args.folder_name = '{}_{}_{}'.format(args.time_name, args.net_name, args.dataset)
    args.save_folder = osp.join(args.base_dir, args.folder_name)

    # seed
    if args.seed != 0:
        set_seeds(args.seed)

    # logger
    args.logger = setup_logger("Robust PCA Network", args.save_folder, 0, filename='log.txt')
    return args


def set_seeds(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


def backup_model_files(save_folder):
    """
    备份当前训练使用的模型文件到训练输出文件夹
    """
    # 创建模型备份文件夹
    model_backup_dir = osp.join(save_folder, 'model_backup')
    os.makedirs(model_backup_dir, exist_ok=True)
    
    # 需要备份的文件列表
    files_to_backup = [
        'models/DRPCANet.py',
        'models/__init__.py', 
        'train.py',
        'utils/loss.py',
        'run_config.py'
    ]
    
    # 复制文件
    for file_path in files_to_backup:
        if osp.exists(file_path):
            filename = osp.basename(file_path)
            dest_path = osp.join(model_backup_dir, filename)
            shutil.copy2(file_path, dest_path)
            print(f"✓ 已备份: {file_path} -> {dest_path}")
        else:
            print(f"⚠ 文件不存在，跳过备份: {file_path}")
    
    # 创建备份信息文件
    backup_info_path = osp.join(model_backup_dir, 'backup_info.txt')
    
    def get_real_run_cmd():
        try:
            pid = os.getpid()
            cmd_list = []
            for _ in range(5):
                with open(f"/proc/{pid}/cmdline", "r") as f:
                    cmdline = f.read().replace('\0', ' ').strip()
                if not cmdline or any(s in cmdline for s in ['bash', 'zsh', 'tmux', 'fish']):
                    break
                cmd_list.append(cmdline)
                with open(f"/proc/{pid}/status", "r") as f:
                    for line in f:
                        if line.startswith("PPid:"):
                            pid = int(line.split()[1])
                            break
                if pid <= 1:
                    break
            valid_cmds = [c for c in cmd_list if 'sh -c' not in c and '/bin/sh' not in c]
            if valid_cmds:
                return valid_cmds[-1]
        except Exception:
            pass
        return f"{sys.executable} {' '.join(sys.argv)}"

    with open(backup_info_path, 'w', encoding='utf-8') as f:
        f.write(f"训练备份信息\n")
        f.write(f"运行命令: {get_real_run_cmd()}\n")
        f.write(f"备份时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"备份文件列表:\n")
        for file_path in files_to_backup:
            if osp.exists(file_path):
                f.write(f"  ✓ {file_path}\n")
            else:
                f.write(f"  ✗ {file_path} (文件不存在)\n")
        f.write(f"\n备份说明:\n")
        f.write(f"- DRPCANet.py: 主要模型架构文件\n")
        f.write(f"- __init__.py: 模型初始化文件\n") 
        f.write(f"- train.py: 训练脚本\n")
        f.write(f"- loss.py: 损失函数定义\n")
        f.write(f"- run_config.py: 运行配置\n")
    
    print(f"✓ 模型文件备份完成! 备份位置: {model_backup_dir}")


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.iter_num = 0

        # 在训练开始前备份模型文件
        backup_model_files(args.save_folder)

        ## dataset
        if args.dataset == 'sirstaug':
            trainset = SirstAugDataset(base_dir=r'./datasets/sirst_aug',
                                       mode='train', base_size=args.base_size)
            valset = SirstAugDataset(base_dir=r'./datasets/sirst_aug',
                                     mode='test', base_size=args.base_size)
        elif args.dataset == 'irstd1k':
            trainset = IRSTD1kDataset(base_dir=r'./datasets/IRSTD-1k', mode='train', base_size=args.base_size)
            valset = IRSTD1kDataset(base_dir=r'./datasets/IRSTD-1k', mode='test', base_size=args.base_size)
        elif args.dataset == 'nudt':
            trainset = NUDTDataset(base_dir=r'./datasets/NUDT-SIRST', mode='train', base_size=args.base_size)
            valset = NUDTDataset(base_dir=r'./datasets/NUDT-SIRST', mode='test', base_size=args.base_size)
        elif args.dataset == 'SIRSTv1':
            trainset = SirstDataset(base_dir=r'datasets/SIRSTv1', mode='train', base_size=args.base_size)
            valset = SirstDataset(base_dir=r'datasets/SIRSTv1', mode='test', base_size=args.base_size)
        else:
            raise NotImplementedError

        self.train_data_loader = Data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        self.val_data_loader = Data.DataLoader(valset, batch_size=args.batch_size, shuffle=True)
        self.iter_per_epoch = len(self.train_data_loader)
        self.max_iter = args.epochs * self.iter_per_epoch

        ## GPU
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        self.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

        ## model
        self.net = get_model(args.net_name)

        # self.net.apply(self.weight_init)
        self.net = self.net.to(self.device)

        ## criterion
        if args.seg_loss == 'hybrid':
            self.criterion = HybridLoss(lambda_iou=args.lambda_iou, lambda_bce=args.lambda_bce)
        elif args.seg_loss == 'softiou':
            self.criterion = SoftLoULoss()
        elif args.seg_loss == 'sls':
            self.criterion = SLSIoULoss()
        else:
            raise ValueError(f"Unsupported seg_loss: {args.seg_loss}")
        self.mse = torch.nn.MSELoss()

        ## lr scheduler
        self.scheduler = LR_Scheduler_Head(args.lr_scheduler, args.lr,
                                           args.epochs, len(self.train_data_loader), lr_step=10)

        ## optimizer
        # self.optimizer = torch.optim.Adagrad(self.net.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.learning_rate,
        #                                  momentum=0.9, weight_decay=1e-4)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)

        ## evaluation metrics
        self.metric = SegmentationMetricTPFNFP(nclass=1)
        self.best_miou = 0
        self.best_fmeasure = 0
        self.eval_loss = 0  # tmp values
        self.miou = 0
        self.fmeasure = 0
        self.eval_my_PD_FA = my_PD_FA()

        ## SummaryWriter
        self.writer = SummaryWriter(log_dir=args.save_folder)
        self.writer.add_text(args.folder_name, 'Args:%s, ' % args)

        ## log info
        self.logger = args.logger
        self.logger.info(args)
        self.logger.info("Using device: {}".format(self.device))

    def training(self):
        # training step
        start_time = time.time()
        base_log = "Epoch-Iter: [{:d}/{:d}]-[{:d}/{:d}] || Lr: {:.6f} || Loss: {:.4f}={:.4f}+{:.4f} || " \
                   "Cost Time: {} || Estimated Time: {}"
        for epoch in range(args.epochs):
            for i, (data, labels) in enumerate(self.train_data_loader):
                self.net.train()

                self.scheduler(self.optimizer, i, epoch, self.best_miou)

                data = data.to(self.device)

                labels = labels.to(self.device)
                out_D, out_T = self.net(data)

                if self.args.seg_loss == 'sls':
                    loss_seg = self.criterion(out_T, labels, self.args.warm_up_epochs, epoch, with_shape=self.args.sls_with_shape)
                else:
                    loss_seg = self.criterion(out_T, labels)
                
                loss_mse = self.mse(out_D, data)
                gamma = torch.Tensor([0.1]).to(self.device)
                
                loss_all = loss_seg + torch.mul(gamma, loss_mse)

                self.optimizer.zero_grad()
                loss_all.backward() 
                self.optimizer.step()

                self.iter_num += 1

                cost_string = str(datetime.timedelta(seconds=int(time.time() - start_time)))
                eta_seconds = ((time.time() - start_time) / self.iter_num) * (self.max_iter - self.iter_num)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                self.writer.add_scalar('Train Loss/Loss All', np.mean(loss_all.item()), self.iter_num)
                self.writer.add_scalar('Train Loss/Loss Seg', np.mean(loss_seg.item()), self.iter_num)
                self.writer.add_scalar('Train Loss/Loss MSE', np.mean(loss_mse.item()), self.iter_num)
                self.writer.add_scalar('Learning rate/', self.optimizer.param_groups[0]['lr'], self.iter_num)

                if self.iter_num % self.args.log_per_iter == 0:
                    self.logger.info(base_log.format(epoch + 1, args.epochs, self.iter_num % self.iter_per_epoch,
                                                     self.iter_per_epoch, self.optimizer.param_groups[0]['lr'],
                                                     loss_all.item(), loss_seg.item(), loss_mse.item(),
                                                     cost_string, eta_string))

                # if (self.iter_num % args.save_iter_step) == 0 or self.iter_num % self.iter_per_epoch == 0:
                #     self.validation()
                if (self.iter_num % args.save_iter_step) == 0 or self.iter_num % self.iter_per_epoch == 0:
                    self.validation(epoch) # 传入 epoch 参数

    def validation(self, epoch):  # 添加epoch参数
        self.metric.reset()
        # self.eval_my_PD_FA.reset()
        self.net.eval()
        base_log = "Data: {:s}, mIoU: {:.4f}/{:.4f}, F1: {:.4f}/{:.4f} "
        # base_log = "Data: {:s}, mIoU: {:.4f}/{:.4f}, F1: {:.4f}/{:.4f}, Pd:{:.4f}, Fa:{:.8f} "
        for i, (data, labels) in enumerate(self.val_data_loader):
            with torch.no_grad():
                out_D, out_T = self.net(data.to(self.device))
            out_D, out_T = out_D.cpu(), out_T.cpu()
            pred = out_T


            if self.args.seg_loss == 'sls':
                loss_seg = self.criterion(out_T, labels, self.args.warm_up_epochs, epoch, with_shape=self.args.sls_with_shape)
            else:
                loss_seg = self.criterion(out_T, labels)
            loss_mse = self.mse(out_D, data)
            gamma = torch.Tensor([0.1])  # 保持在CPU上
            loss_all = loss_seg + torch.mul(gamma, loss_mse)

            self.metric.update(labels, out_T)


        miou, prec, recall, fmeasure = self.metric.get()
        torch.save(self.net.state_dict(), osp.join(self.args.save_folder, 'latest.pkl'))
        if miou > self.best_miou:
            self.best_miou = miou
            torch.save(self.net.state_dict(), osp.join(self.args.save_folder, 'best.pkl'))
        if fmeasure > self.best_fmeasure:
            self.best_fmeasure = fmeasure


        self.writer.add_scalar('Test/mIoU', miou, self.iter_num)
        self.writer.add_scalar('Test/F1', fmeasure, self.iter_num)
        self.writer.add_scalar('Best/mIoU', self.best_miou, self.iter_num)
        self.writer.add_scalar('Best/Fmeasure', self.best_fmeasure, self.iter_num)

        self.logger.info(base_log.format(self.args.dataset, miou, self.best_miou, fmeasure, self.best_fmeasure))


if __name__ == '__main__':
    args = parse_args()


    trainer = Trainer(args)
    trainer.training()

    print('Best mIoU: %.5f, Best Fmeasure: %.5f\n\n' % (trainer.best_miou, trainer.best_fmeasure))
