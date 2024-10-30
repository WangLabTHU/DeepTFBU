import torch
import torch.nn.functional as torch_F
from SeqRegressionModel import DenseLSTM_models
from matplotlib import pyplot as plt
import numpy as np
import random
import math
from loss import *
import torch.multiprocessing as mp
import os
from numba import jit
import numba


def replace_n_with_random_dna_char(input_str, random_seed):
    random.seed(random_seed)
    dna_chars = ['A', 'C', 'G', 'T']
    result = ''
    for char in input_str:
        if char == 'N':
            result += random.choice(dna_chars)
        else:
            result += char
    return result

@jit(nopython=True)
def seq_to_onehot(seq_list,length):
    data = np.zeros((len(seq_list),length,4))
    num_dic = {'A':0,'C':1,'G':2,'T':3}
    for i in range(len(seq_list)):
        for j in range(length):
            if seq_list[i][j]=='N':
                continue
            data[i][j][num_dic[seq_list[i][j]]]=1
    return data

def seq_to_onehot_single(seq):
    data = np.zeros((len(seq),4))
    num_dic = {'A':0,'C':1,'G':2,'T':3}
    for j in range(len(seq)):
        if seq[j]=='N':
            continue
        data[j][num_dic[seq[j]]]=1
    return data

def cal_infoconten(mat):
    info_conten = np.zeros(len(mat))
    for i in range(len(mat)):
        ratio = mat[i][0:4]/(sum(mat[i][0:4]))
        ratio_no0 = ratio[ratio!=0]
        info_conten[i] = 2+np.sum(ratio_no0 * np.log2(ratio_no0))
    return info_conten

def mat_to_seq(mat):
    dna_chars = ['A', 'C', 'G', 'T', 'N']
    seq = ''
    for lines in mat:
        max_pos = np.argmax(lines)
        seq+=dna_chars[max_pos]
    return seq

class Seq2ScalarTraining:
    def __init__(self,TF_name,back_seq,seq_mask,core_pos,motif_seq_temp):
        self.batch_size = 128
        self.lr_expr = 0.001
        seq_len = 168
        self.gpu = True
        self.patience = 30
        self.mode = 'denselstm_classi'
        self.name = 'denselstm_mc_0.001_mask_' + str(seq_len)
        self.TF_name = TF_name
        self.seq_len = seq_len

        self.core_pos = core_pos
        self.seq_mask = seq_mask
        self.motif_seq_temp = motif_seq_temp
        self.start_seq_list = []
        for i in range(2000):
            random_seed = i
            seq_temp = replace_n_with_random_dna_char(back_seq, random_seed)
            self.start_seq_list.append(seq_temp)
        self.all_start_onehot = seq_to_onehot(numba.typed.List(self.start_seq_list), len(self.start_seq_list[0]))

        self.total_train_num = 1
        # it's set to be 10 for ten replicated trained model in the paper
        self.model_list = []
        for i in range(self.total_train_num):
            model_temp = DenseLSTM_models(input_nc=4, seqL=seq_len, out_length=1, mode=self.mode)
            if self.gpu:
                model_temp=model_temp.cuda()
            model_temp = torch.load('../1_HepG2_training_and_gen_TFBU/step0_HepG2_model_path/test_denselstm_mc_0.001_mask_168_'+self.TF_name+'.pth')
            # the path above should contain the variable 'i' as the replicated number in it
            model_temp.eval()
            self.model_list.append(model_temp)


    def optimize(self):
        pnew = 0.3
        pelite = 0.3
        
        Maxpoolsize = 2000
        opti_iter_num=300
        result_log = np.zeros((Maxpoolsize,opti_iter_num))

        start_seq = self.start_seq_list
        onehot_start = self.all_start_onehot

        onehot_motif_ref = seq_to_onehot_single(self.motif_seq_temp)

        result_start_all = np.zeros((len(onehot_start),len(self.core_pos)))
        for pos_num,core_temp in enumerate(self.core_pos):
            onehot_temp = onehot_start[:,core_temp-84:core_temp-84+168,:].copy()

            for single_one_hot in onehot_temp:
                if np.array_equal(single_one_hot[int(168/2)-int(len(self.motif_seq_temp)/2):int(168/2)-int(len(self.motif_seq_temp)/2)+int(len(self.motif_seq_temp)),:],onehot_motif_ref):
                    single_one_hot[int(168/2)-int(len(self.motif_seq_temp)/2):int(168/2)-int(len(self.motif_seq_temp)/2)+int(len(self.motif_seq_temp)),:] = 0
                else:
                    raise ValueError("Core motif mismatch")

            onehot_temp_tensor = torch.tensor(onehot_temp).float().cuda(non_blocking=True)

            result_start_temp_all = np.zeros((len(onehot_temp_tensor),self.total_train_num))
            for tr_num in range(self.total_train_num):
                result_temp = self.model_list[tr_num](onehot_temp_tensor)
                result_temp = result_temp.detach().cpu().float()
                result_temp = torch_F.softmax(result_temp, dim=1)
                result_temp = result_temp.numpy()[:,1]
                result_start_temp_all[:,tr_num] = result_temp
            result_start_temp = np.mean(result_start_temp_all,axis=1)

            result_start_all[:,pos_num] = result_start_temp

        result_start = np.mean(result_start_all,axis=1)

        with open('./step1_halfmotif_GA_log/seqs_txt_' + self.TF_name +'/seqs_txt_start.txt','w') as f:
            for seq in start_seq:
                f.write(seq+'\n')

        np.savetxt('./step1_halfmotif_GA_log/scores_' + self.TF_name +'/score_mask_start.txt', result_start)
        np.savetxt('./step1_halfmotif_GA_log/scores_all_' + self.TF_name +'/score_all_mask_start.txt', result_start_all)

        for i in range(opti_iter_num): 

            start_seq = np.array(start_seq)
            print(i)
            I = np.argsort(result_start)
            I = I[::-1]
            result_start = result_start[I]
            Poolsize = result_start.shape[0]
            Nnew = math.ceil(Poolsize*float(pnew))
            Nelite = math.ceil(Poolsize*float(pelite))
            IParent = self.select_parent(Nnew, Nelite, Poolsize)
            Parent = start_seq[IParent].copy()
            seq_new = self.act(Parent,self.seq_mask)


            onehot_new = seq_to_onehot(numba.typed.List(seq_new), len(seq_new[0]))


            result_new_all = np.zeros((len(onehot_new),len(self.core_pos)))
            for pos_num,core_temp in enumerate(self.core_pos):
                onehot_temp = onehot_new[:,core_temp-84:core_temp-84+168,:].copy()

                for single_one_hot in onehot_temp:
                    if np.array_equal(single_one_hot[int(168/2)-int(len(self.motif_seq_temp)/2):int(168/2)-int(len(self.motif_seq_temp)/2)+int(len(self.motif_seq_temp)),:],onehot_motif_ref):
                        single_one_hot[int(168/2)-int(len(self.motif_seq_temp)/2):int(168/2)-int(len(self.motif_seq_temp)/2)+int(len(self.motif_seq_temp)),:] = 0
                    else:
                        raise ValueError("Core motif mismatch")

                onehot_temp_tensor = torch.tensor(onehot_temp).float().cuda(non_blocking=True)

                result_new_temp_all = np.zeros((len(onehot_temp_tensor),self.total_train_num))
                for tr_num in range(self.total_train_num):
                    result_temp = self.model_list[tr_num](onehot_temp_tensor)
                    result_temp = result_temp.detach().cpu().float()
                    result_temp = torch_F.softmax(result_temp, dim=1)
                    result_temp = result_temp.numpy()[:,1]
                    result_new_temp_all[:,tr_num] = result_temp
                result_new_temp = np.mean(result_new_temp_all,axis=1)
                result_new_all[:,pos_num] = result_new_temp

            result_new = np.mean(result_new_all,axis=1)


            all_seqs = np.concatenate([start_seq, seq_new])
            all_Score = np.append(result_start,result_new)
            all_Score_all = np.concatenate([result_start_all, result_new_all])
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
            with open('./step1_halfmotif_GA_log/seqs_txt_' + self.TF_name +'/seqs_txt_'+str(i)+'.txt','w') as f:
                for seq in all_seqs:
                    f.write(seq+'\n')

            np.savetxt('./step1_halfmotif_GA_log/scores_' + self.TF_name +'/score_mask_'+str(i)+'.txt', all_Score)
            np.save('./step1_halfmotif_GA_log/seqs_' + self.TF_name +'/seqs_mask_'+str(i)+'.npy', all_seqs)
            np.savetxt('./step1_halfmotif_GA_log/scores_all_' + self.TF_name +'/score_all_mask_'+str(i)+'.txt', all_Score_all)

            result_log[:,i] = all_Score
            start_seq = all_seqs
            result_start = all_Score
            result_start_all = all_Score_all

        plt.boxplot(result_log)
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.savefig('./step1_halfmotif_GA_log/scores_mask_' + self.TF_name +'_'+str(pnew)+'_'+str(pelite)+ '_'+self.TF_name+'.png')
        plt.close()


        return 0

    def PMutate_withmask(self, z, mask_seq): # single nucleotide mutaion
        z = list(z)
        p_result = np.random.randint(0,4)
        p = np.random.randint(0,len(z))
        while(mask_seq[p]==1):
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

    def act(self, parent, mask_seq):
        for i in range(parent.shape[0]):
            action = np.random.randint(0,2)
            if action == 0:
                parent[i] = self.PMutate_withmask(parent[i],mask_seq)
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



def count_leading_zeros(s):
    count = 0
    for char in s:
        if char == '0':
            count += 1
        else:
            break
    return count



def main():
    TF_list = ['GATA2']
    TF_mats = {} # the PFM or PPM matrix of each TF
    TF_mats['GATA2'] = np.array([[3303, 2978, 3201, 4731],
                                [1986, 4099, 2966, 5162],
                                [940, 10607, 1223, 1443],
                                [444, 259, 150, 13360],
                                [417, 34, 21, 13741],
                                [14098, 14, 43, 58],
                                [145, 77, 48, 13943],
                                [96, 13786, 126, 205],
                                [2373, 492, 367, 10981],
                                [2912, 3714, 3656, 3931],
                                [3278, 3874, 2216, 4845]])
    TF_ppm = {}
    for TF_temp in TF_list:
        mat = TF_mats[TF_temp]
        PPM_temp = np.zeros((len(mat), 4))
        for i in range(len(mat)):
            PPM_temp[i] = mat[i] / (sum(mat[i]))
        PPM_temp = np.concatenate((PPM_temp, np.ones((len(PPM_temp), 1)) * 0.25), axis=1)
        TF_ppm[TF_temp] = PPM_temp

    motif_seq_long = {} # full length consensus motif sequence
    for TF_name in TF_list:
        motif_seq_long[TF_name] = mat_to_seq(TF_ppm[TF_name])

    motif_seq = {} # consensus motif sequence removed posotion where infoconten<0.3
    for TF_name in TF_list:
        PPM_temp = TF_ppm[TF_name]
        info_conten_temp = cal_infoconten(PPM_temp)
        to_del = []
        for i in range(len(info_conten_temp)):
            if info_conten_temp[i]<0.3:
                to_del.append(i)
            else:
                break
        for i in range(len(info_conten_temp)):
            if info_conten_temp[len(info_conten_temp)-1-i]<0.3:
                to_del.append(len(info_conten_temp)-1-i)
            else:
                break
        PPM_new = np.delete(PPM_temp,to_del,0)
        motif_seq[TF_name] = mat_to_seq(PPM_new)


    for TF_name in TF_list:
        # sequence on the plasmid around designed enhancer:
        pre_backbone_seq = 'CCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCCTCGGCGGCCAAGCTTAGACAC'
        after_backbone_seq = 'CACTGCGGCTCCTGCGATAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCAC'

        motif_seq_temp = motif_seq_long[TF_name]
        middle_seq = ''
        for i in range(7):
            middle_seq += motif_seq_temp + 'NNN'
        middle_seq+=motif_seq_temp
        middle_seq+='N' 

        back_seq = pre_backbone_seq + middle_seq + after_backbone_seq

        motif_seq_short_temp = motif_seq[TF_name]

        matching_result = ['0'] * len(motif_seq_temp)
        for i in range(len(motif_seq_temp) - len(motif_seq_short_temp) + 1):
            if motif_seq_temp[i:i + len(motif_seq_short_temp)] == motif_seq_short_temp:
                for j in range(i, i + len(motif_seq_short_temp)):
                    matching_result[j] = '1'
                break
        matching_string = ''.join(matching_result)

        seq_mask_str = ''
        seq_mask_str += '1'*len(pre_backbone_seq)
        for i in range(7):
            if i%2==0:
                seq_mask_str += matching_string + '000'
            else:
                seq_mask_str += '0'*len(matching_string) + '000'
        seq_mask_str+='0'*len(matching_string)
        seq_mask_str+='0'
        seq_mask_str += '1'*len(after_backbone_seq)

        seq_mask = np.zeros(len(seq_mask_str))
        for i in range(len(seq_mask_str)):
            if seq_mask_str[i]=='1':
                seq_mask[i]=1

        pre_0 = count_leading_zeros(matching_result)
        core_pos = []
        for i in range(8):
            if i%2==0:
                core_pos.append(len(pre_backbone_seq)+3*(i)+len(motif_seq_temp)*i+int(len(motif_seq_short_temp)/2)+pre_0)


        analysis = Seq2ScalarTraining(TF_name,back_seq,seq_mask,core_pos,motif_seq_short_temp)

        os.system('mkdir ./step1_halfmotif_GA_log/')
        os.system('mkdir ./step1_halfmotif_GA_log/scores_' + TF_name)
        os.system('mkdir ./step1_halfmotif_GA_log/scores_all_' + TF_name)
        os.system('mkdir ./step1_halfmotif_GA_log/seqs_' + TF_name)
        os.system('mkdir ./step1_halfmotif_GA_log/seqs_txt_' + TF_name)
        analysis.optimize()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()

