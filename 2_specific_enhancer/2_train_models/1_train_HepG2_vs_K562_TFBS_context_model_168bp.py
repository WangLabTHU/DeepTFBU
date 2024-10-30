import torch
import torch.nn.functional as torch_F
from torch.utils.data import DataLoader
from pre_data_iter import ProbDataset
from SeqRegressionModel import DenseLSTM_models
from matplotlib import pyplot as plt
import numpy as np
import collections
import pandas as pd
from pytorchtools import EarlyStopping
from tqdm import tqdm
from loss import *
import torch.multiprocessing as mp
import h5py
from sklearn.metrics import roc_curve, auc
import os


class Seq2ScalarTraining:
    def __init__(self,TF_name):
        self.batch_size = 128
        self.lr_expr = 0.001
        seq_len = 168
        self.gpu = True
        self.patience = 30
        self.mode = 'denselstm_classi'
        self.name = 'denselstm_mc_0.001_mask_' + str(seq_len)
        self.TF_name = TF_name

        f = h5py.File('../1_prepare_data_for_HepG2_vs_K562/data_dir/step0_HepG2_vs_K562_motif_center_seq_mask_'+str(seq_len)+'_'+TF_name+'_data.h5', 'r')

        Neg_seqs = f['neg_' + self.TF_name][:]
        Pos_seqs = f['pos_' + self.TF_name][:]
        

        np.random.seed(42)
        np.random.shuffle(Neg_seqs)
        np.random.seed(42)
        np.random.shuffle(Pos_seqs)

        
        dataset_size = min(len(Neg_seqs),len(Pos_seqs))
        self.data_size = dataset_size
        Neg_seq_train = Neg_seqs[0:int(dataset_size*0.8)]
        Pos_seq_train = Pos_seqs[0:int(dataset_size*0.8)]
        Neg_seq_val = Neg_seqs[int(dataset_size*0.8):int(dataset_size*0.9)]
        Pos_seq_val = Pos_seqs[int(dataset_size*0.8):int(dataset_size*0.9)]
        Neg_seq_test = Neg_seqs[int(dataset_size*0.9):dataset_size]
        Pos_seq_test = Pos_seqs[int(dataset_size*0.9):dataset_size]

        train_seq = np.concatenate((Neg_seq_train,Pos_seq_train))
        train_labels = np.concatenate((np.zeros(len(Neg_seq_train)),np.ones(len(Pos_seq_train))))
        val_seq = np.concatenate((Neg_seq_val,Pos_seq_val))
        val_labels = np.concatenate((np.zeros(len(Neg_seq_val)),np.ones(len(Pos_seq_val))))
        test_seq = np.concatenate((Neg_seq_test,Pos_seq_test))
        test_labels = np.concatenate((np.zeros(len(Neg_seq_test)),np.ones(len(Pos_seq_test))))


        self.dataset_train = DataLoader(dataset=ProbDataset(seq=train_seq,label=train_labels), batch_size=self.batch_size, shuffle=True)
        self.dataset_test = DataLoader(dataset=ProbDataset(seq=val_seq,label=val_labels), batch_size=self.batch_size, shuffle=False)
        self.dataset_test_final = DataLoader(dataset=ProbDataset(seq=test_seq,label=test_labels), batch_size=self.batch_size,shuffle=False)
        self.epoch = 1000
        self.model_ratio = DenseLSTM_models(input_nc=4, seqL=seq_len, out_length=1, mode=self.mode)
        self.save_path = './step1_HepG2_vs_K562_model_path/'
        if self.gpu:
            self.model_ratio=self.model_ratio.cuda()
        self.loss_y = torch.nn.CrossEntropyLoss()
        if self.mode == 'deepinfomax':
            self.optimizer_ratio = torch.optim.Adam(self.model_ratio.fc.parameters(), lr=self.lr_expr)
        else:
            self.optimizer_ratio = torch.optim.Adam(self.model_ratio.parameters(), lr=self.lr_expr)


    def training(self):
        trainingLog = collections.OrderedDict()
        trainingLog['test_auc'] = []
        trainingLog['train_loss'] = []
        trainingLog['test_loss'] = []
        early_stopping = EarlyStopping(patience=self.patience, verbose=True, path=self.save_path + 'test_' + self.name + '_' + self.TF_name +'.pth', stop_order='max')
        for ei in range(self.epoch):
            train_loss_y = 0
            train_num_y = 0
            test_loss_y = 0
            test_num = 0
            self.model_ratio.train()
            print('Training iters')
            for x,y in tqdm(self.dataset_train):
                train_data = x
                train_y = y
                train_data = train_data.float()
                train_y = train_y.long()
                train_data = train_data.cuda(non_blocking=True)
                train_y = train_y.cuda(non_blocking=True)
                predict = self.model_ratio(train_data)
                loss_y = self.loss_y(predict, train_y)
                self.optimizer_ratio.zero_grad()
                loss_y.backward()
                self.optimizer_ratio.step()
                train_loss_y += loss_y
                train_num_y = train_num_y + 1
            test_predict_expr = []
            test_real_expr = []
            self.model_ratio.eval()
            print('Test iters')
            for x,y in tqdm(self.dataset_test):
                test_data = x
                test_y = y
                test_data = test_data.float()
                test_y = test_y.long()
                test_data = test_data.cuda(non_blocking=True)
                test_y = test_y.cuda(non_blocking=True)
                predict_y = self.model_ratio(test_data)
                predict_y = predict_y.detach()
                predict_y2 = predict_y
                predict_y = predict_y.cpu().float()
                predict_result = torch_F.softmax(predict_y, dim=1)
                predict_result = predict_result.numpy()[:,1]
                real_y = test_y.cpu().float().numpy()
                test_loss_y += self.loss_y(predict_y2, test_y)
                test_num = test_num + 1

                for i in range(np.size(real_y)):
                    test_real_expr.append(real_y[i])
                    test_predict_expr.append(predict_result[i])

            test_real_expr = np.asarray(test_real_expr)
            test_predict_expr = np.asarray(test_predict_expr)
            fpr, tpr, threshold = roc_curve(test_real_expr, test_predict_expr)
            roc_auc = auc(fpr, tpr)

            trainingLog['test_auc'].append(roc_auc)
            trainingLog['train_loss'].append(float(train_loss_y)/train_num_y)
            trainingLog['test_loss'].append(float(test_loss_y)/test_num)
            print('epoch:{}train_loss y:{} test_loss y:{} test_auc:{}'.format(ei, train_loss_y/train_num_y, test_loss_y/test_num, roc_auc))
            early_stopping(val_loss=roc_auc, model=self.model_ratio)
            if early_stopping.early_stop:
                print('Early Stopping......')
                break
        predict_ratio = []
        real_ratio = []
        self.model_ratio = torch.load(self.save_path + 'test_' + self.name + '_' + self.TF_name +'.pth')
        self.model_ratio.eval()
        for x,y in self.dataset_test_final:
            test_data = x
            test_y = y
            test_data = test_data.float()
            test_y = test_y.long()
            test_data = test_data.cuda(non_blocking=True)
            test_y = test_y.cuda(non_blocking=True)
            predict_y = self.model_ratio(test_data)

            predict_y = predict_y.detach()
            predict_y = predict_y.cpu().float()
            predict_result = torch_F.softmax(predict_y, dim=1)
            predict_result = predict_result.numpy()[:,1]
            real_y = test_y.cpu().float().numpy()

            for i in range(np.size(real_y)):
                real_ratio.append(real_y[i])
                predict_ratio.append(predict_result[i])        

        real_expr = np.asarray(real_ratio)
        predict_expr = np.asarray(predict_ratio)
        np.savetxt('./step1_HepG2_vs_K562_log/'+self.name+'_' + self.TF_name +'_real_expr_final.txt',real_expr)
        np.savetxt('./step1_HepG2_vs_K562_log/'+self.name+'_' + self.TF_name +'_predict_expr_final.txt',predict_expr)
        
        fpr, tpr, threshold = roc_curve(real_expr, predict_expr)
        roc_auc = auc(fpr, tpr)
        auc_log_file = open('./step1_HepG2_vs_K562_log/log_masked.txt','a')
        auc_log_file.write(self.TF_name + '\t' +str(roc_auc) + '\n')
        auc_log_file.close()
        plt.figure()
        lw = 2
        plt.figure(figsize=(10, 10))
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig('./step1_HepG2_vs_K562_log/'+self.name+'_' + self.TF_name +'_predict_expr_final.png')
        plt.close()
        log_file = open('./step1_HepG2_vs_K562_log/result_log.txt','a')
        log_file.write(self.TF_name + '\t' + self.name + '\t' + str(roc_auc) + '\n')
        log_file.close()
        trainingLog = pd.DataFrame(trainingLog)
        trainingLog.to_csv('step1_HepG2_vs_K562_log/training_log' + 'test_' + self.name +'_' + self.TF_name + '.csv', index=False)


def main():
    os.makedirs('./step1_HepG2_vs_K562_model_path', exist_ok=True)
    os.makedirs('./step1_HepG2_vs_K562_log', exist_ok=True)
    TF_list = ['GATA2']
    for TF_name in TF_list:
        print(TF_name)
        analysis = Seq2ScalarTraining(TF_name)
        if analysis.data_size<10:
            print('error: the number of sample counts is too small')
            continue
        analysis.training()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
