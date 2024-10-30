import torch
import torch.nn.functional as torch_F
from torch.utils.data import DataLoader
from pre_data_iter import ProbDataset
from SeqRegressionModel import DenseLSTM_models
from matplotlib import pyplot as plt
import numpy as np
import random
import math
from loss import *
import torch.multiprocessing as mp
import h5py
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
        self.seq_len = seq_len


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
        self.opti_start_seq = np.copy(test_seq)


        self.dataset_train = DataLoader(dataset=ProbDataset(seq=train_seq,label=train_labels), batch_size=self.batch_size, shuffle=True)
        self.dataset_test = DataLoader(dataset=ProbDataset(seq=val_seq,label=val_labels), batch_size=self.batch_size, shuffle=False)
        self.dataset_test_final = DataLoader(dataset=ProbDataset(seq=test_seq,label=test_labels), batch_size=self.batch_size,shuffle=False)
        self.epoch = 1000


        self.total_train_num = 1
        # it's set to be 10 for ten replicated trained model in the paper
        self.models_HepG2_vs_K562 = []
        for i in range(self.total_train_num):
            model_temp = DenseLSTM_models(input_nc=4, seqL=seq_len, out_length=1, mode=self.mode)
            if self.gpu:
                model_temp=model_temp.cuda()
            model_temp = torch.load('../2_train_models/step1_HepG2_vs_K562_model_path/test_denselstm_mc_0.001_mask_168_'+self.TF_name+'.pth')
            # the path above should contain the variable 'i' as the replicated number in it
            model_temp.eval()
            self.models_HepG2_vs_K562.append(model_temp)

        self.models_HepG2 = []
        for i in range(self.total_train_num):
            model_temp = DenseLSTM_models(input_nc=4, seqL=seq_len, out_length=1, mode=self.mode)
            if self.gpu:
                model_temp=model_temp.cuda()
            model_temp = torch.load('../../1_HepG2_training_and_gen_TFBU/step0_HepG2_model_path/test_' + self.name + '_' + self.TF_name +'.pth')
            model_temp.eval()
            self.models_HepG2.append(model_temp)

        self.models_K562 = []
        for i in range(self.total_train_num):
            model_temp = DenseLSTM_models(input_nc=4, seqL=seq_len, out_length=1, mode=self.mode)
            if self.gpu:
                model_temp=model_temp.cuda()
            model_temp = torch.load('../2_train_models/step0_K562_model_path/test_denselstm_mc_0.001_mask_168_'+self.TF_name+'.pth')
            model_temp.eval()
            self.models_K562.append(model_temp)

        self.weight_HepG2_vs_K562 = -1
        self.weight_HepG2 = -1
        self.weight_K562 = 1


    def optimize(self):
        pnew = 0.3
        pelite = 0.3        
        Maxpoolsize = 2000
        opti_iter_num=300
        if len(self.opti_start_seq)<Maxpoolsize:
            Maxpoolsize = len(self.opti_start_seq)
        result_log = np.zeros((Maxpoolsize,opti_iter_num))

        start_seq_onehot = self.opti_start_seq[0:Maxpoolsize]
        start_seq_onehot = torch.tensor(start_seq_onehot).float().cuda(non_blocking=True)
        start_seq = []
        for seq_onehot in self.opti_start_seq[0:Maxpoolsize]:
            start_seq.append(self.one_hot_to_seq(seq_onehot))
        start_seq = np.array(start_seq)
        

        result_start_all_HepG2_vs_K562 = np.zeros((len(start_seq_onehot),self.total_train_num))
        for tr_num in range(self.total_train_num):
            result_temp = self.models_HepG2_vs_K562[tr_num](start_seq_onehot)
            result_temp = result_temp.detach().cpu().float()
            result_temp = torch_F.softmax(result_temp, dim=1)
            result_temp = result_temp.numpy()[:,1]
            result_start_all_HepG2_vs_K562[:,tr_num] = result_temp
        result_start_HepG2_vs_K562 = np.mean(result_start_all_HepG2_vs_K562,axis=1)

        result_start_all_HepG2 = np.zeros((len(start_seq_onehot),self.total_train_num))
        for tr_num in range(self.total_train_num):
            result_temp = self.models_HepG2[tr_num](start_seq_onehot)
            result_temp = result_temp.detach().cpu().float()
            result_temp = torch_F.softmax(result_temp, dim=1)
            result_temp = result_temp.numpy()[:,1]
            result_start_all_HepG2[:,tr_num] = result_temp
        result_start_HepG2 = np.mean(result_start_all_HepG2,axis=1)

        result_start_all_K562 = np.zeros((len(start_seq_onehot),self.total_train_num))
        for tr_num in range(self.total_train_num):
            result_temp = self.models_K562[tr_num](start_seq_onehot)
            result_temp = result_temp.detach().cpu().float()
            result_temp = torch_F.softmax(result_temp, dim=1)
            result_temp = result_temp.numpy()[:,1]
            result_start_all_K562[:,tr_num] = result_temp
        result_start_K562 = np.mean(result_start_all_K562,axis=1)


        result_start = result_start_HepG2_vs_K562*self.weight_HepG2_vs_K562 + result_start_HepG2*self.weight_HepG2 + result_start_K562*self.weight_K562

        result_start_all = np.zeros((len(result_start_HepG2_vs_K562),3))
        result_start_all[:,0] = result_start_HepG2_vs_K562
        result_start_all[:,1] = result_start_HepG2
        result_start_all[:,2] = result_start_K562

        for i in range(opti_iter_num): 
            print(i)
            I = np.argsort(result_start)
            I = I[::-1]
            result_start = result_start[I]

            Poolsize = result_start.shape[0]
            Nnew = math.ceil(Poolsize*float(pnew))
            Nelite = math.ceil(Poolsize*float(pelite))
            IParent = self.select_parent(Nnew, Nelite, Poolsize)
            Parent = start_seq[IParent].copy()
            seq_new = self.act(Parent)

            onehot_new = self.seq_to_onehot(seq_new,self.seq_len)
            onehot_new = torch.tensor(onehot_new).float().cuda(non_blocking=True)

            Score_new_all_HepG2_vs_K562 = np.zeros((len(onehot_new),self.total_train_num))
            for tr_num in range(self.total_train_num):
                result_temp = self.models_HepG2_vs_K562[tr_num](onehot_new)
                result_temp = result_temp.detach().cpu().float()
                result_temp = torch_F.softmax(result_temp, dim=1)
                result_temp = result_temp.numpy()[:,1]
                Score_new_all_HepG2_vs_K562[:,tr_num] = result_temp
            Score_new_HepG2_vs_K562 = np.mean(Score_new_all_HepG2_vs_K562,axis=1)

            Score_new_all_HepG2 = np.zeros((len(onehot_new),self.total_train_num))
            for tr_num in range(self.total_train_num):
                result_temp = self.models_HepG2[tr_num](onehot_new)
                result_temp = result_temp.detach().cpu().float()
                result_temp = torch_F.softmax(result_temp, dim=1)
                result_temp = result_temp.numpy()[:,1]
                Score_new_all_HepG2[:,tr_num] = result_temp
            Score_new_HepG2 = np.mean(Score_new_all_HepG2,axis=1)

            Score_new_all_K562 = np.zeros((len(onehot_new),self.total_train_num))
            for tr_num in range(self.total_train_num):
                result_temp = self.models_K562[tr_num](onehot_new)
                result_temp = result_temp.detach().cpu().float()
                result_temp = torch_F.softmax(result_temp, dim=1)
                result_temp = result_temp.numpy()[:,1]
                Score_new_all_K562[:,tr_num] = result_temp
            Score_new_K562 = np.mean(Score_new_all_K562,axis=1)


            Score_new = Score_new_HepG2_vs_K562*self.weight_HepG2_vs_K562 + Score_new_HepG2*self.weight_HepG2 + Score_new_K562*self.weight_K562
            Score_new_all = np.zeros((len(Score_new_HepG2_vs_K562),3))
            Score_new_all[:,0] = Score_new_HepG2_vs_K562
            Score_new_all[:,1] = Score_new_HepG2
            Score_new_all[:,2] = Score_new_K562

            all_seqs = np.concatenate([start_seq, seq_new])
            all_Score = np.append(result_start,Score_new)
            all_Score_all = np.concatenate([result_start_all, Score_new_all])
            unique_temp = set(all_seqs.tolist())
            unique_I = []
            for seq_temp in unique_temp:
                unique_I.append(all_seqs.tolist().index(seq_temp))
            all_seqs = all_seqs[unique_I]
            all_Score = all_Score[unique_I]
            all_Score_all = all_Score_all[unique_I]

            I = np.argsort(all_Score)
            I = I[::-1]
            all_seqs = all_seqs[I]
            all_Score = all_Score[I]
            all_Score_all = all_Score_all[I]
            all_seqs = all_seqs[:Maxpoolsize]
            all_Score = all_Score[:Maxpoolsize]
            all_Score_all = all_Score_all[:Maxpoolsize]
            np.savetxt('./step0_K562_specific_GA_log/score_mask_' + self.TF_name +'_'+str(i)+'.txt', all_Score)
            np.save('./step0_K562_specific_GA_log/seqs_mask_' + self.TF_name +'_'+str(i)+'.npy', all_seqs)
            np.savetxt('./step0_K562_specific_GA_log/score_all_mask_' + self.TF_name +'_'+str(i)+'.txt', all_Score_all)
            result_log[:,i] = all_Score
            start_seq = all_seqs
            result_start = all_Score
            result_start_all = all_Score_all

        plt.boxplot(result_log)
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.savefig('./step0_K562_specific_GA_log/scores_mask_' + self.TF_name +'_'+str(pnew)+'_'+str(pelite)+'.png')
        plt.close()


        return 0

    def PMutate_withmask(self, z): # single nucleotide mutaion
        z = list(z)
        p_result = np.random.randint(0,4)
        p = np.random.randint(0,len(z))
        while(z[p]=='N'):
            p = np.random.randint(0,len(z))
        list_nuc = ['A','C','G','T']   
        z[p] = list_nuc[p_result]
        z = ''.join(z)
        return z

    def Reorganize(self, z, parent): 
        z = list(z)
        index = np.random.randint(0, 2,size=(len(z)))
        j = np.random.randint(0, parent.shape[0])
        z1 = list(parent[j])
        for i in range(len(z)):
            if index[i] == 1:
                z[i] = z1[i]
        z = ''.join(z)
        return z

    def select_parent(self, Nnew, Nelite, Poolsize):
        ParentFromElite = min(Nelite,Nnew//2)
        ParentFromNormal = min(Poolsize-Nelite, Nnew-ParentFromElite)
        I_e = random.sample([ i for i in range(Nelite)], ParentFromElite)
        I_n = random.sample([ i+Nelite for i in range(Poolsize - Nelite)], ParentFromNormal)
        I = I_e + I_n
        return I

    def act(self, parent):
        for i in range(parent.shape[0]):
            action = np.random.randint(0,2)
            if action == 0:
                parent[i] = self.PMutate_withmask(parent[i])
            elif action == 1:
                parent[i] = self.Reorganize(parent[i], parent)
        return parent

    def one_hot_to_seq(self, one_hot):
        mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}
        seq_lst = []
        for vec in one_hot:
            if sum(vec)==0:
                seq_lst.append('N')
            else:
                seq_lst.append(mapping[np.argmax(vec)])
        seq = ''.join(seq_lst)    
        return seq

    def seq_to_onehot(self, seq_list,length):
        data = np.zeros((len(seq_list),length,4))
        num_dic = {'A':0,'C':1,'G':2,'T':3}
        for i in range(len(seq_list)):
            for j in range(length):
                if seq_list[i][j]=='N':
                    continue
                data[i][j][num_dic[seq_list[i][j]]]=1
        return data



def main():
    TF_list = ['GATA2']
    os.system('mkdir ./step0_K562_specific_GA_log')        
    for TF_name in TF_list:
        print(TF_name)
        analysis = Seq2ScalarTraining(TF_name)
        if analysis.data_size<10:
            continue
        analysis.optimize()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()

