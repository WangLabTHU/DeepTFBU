import numpy as np
from numba import jit
import numba
import pickle
import torch
from torch import nn
import torch.nn.functional as torch_F
from torch.utils.data import DataLoader
from pre_data_iter import ProbDataset
from SeqRegressionModel import DenseLSTM_models
import numpy as np
import random
from loss import *
import torch.multiprocessing as mp
import pickle
from numba import jit
import numba
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
from sklearn.preprocessing import StandardScaler
import os


class Seq2ScalarTraining:
    def __init__(self,TF_name):
        self.batch_size = 2048
        self.lr_expr = 0.001
        seq_len = 168
        self.gpu = True
        self.patience = 30
        self.mode = 'denselstm_classi'
        self.name = 'denselstm_mc_0.001_mask_' + str(seq_len)
        self.TF_name = TF_name
        self.seq_len = seq_len
        self.models = []

        self.total_train_num = 1
        # it's set to be 10 for ten replicated trained model in the paper
        for i in range(self.total_train_num):
            model_temp = DenseLSTM_models(input_nc=4, seqL=seq_len, out_length=1, mode=self.mode)
            if self.gpu:
                model_temp=model_temp.cuda()
            model_temp = torch.load('./all_model_dir/train_'+str(i)+'/test_' + self.name + '_' + self.TF_name +'.pth')
            # the path above should contain the variable 'i' as the replicated number in it
            model_temp.eval()
            self.models.append(model_temp)

    def predict(self,seq_to_pred):
        seq_to_pred_onehot = seq_to_onehot(numba.typed.List(seq_to_pred), 168)
        val_labels = np.ones(len(seq_to_pred_onehot))
        dataset_test = DataLoader(dataset=ProbDataset(seq=seq_to_pred_onehot,label=val_labels), batch_size=128, shuffle=False)
        neg_result_all = []
        for tr_num in range(self.total_train_num):
            model_temp = self.models[tr_num]
            neg_seqs = []
            neg_results = []
            for x,y in dataset_test:
                test_data = x
                test_data = test_data.float()
                test_data = test_data.cuda(non_blocking=True)
                predict_y = model_temp(test_data)
                predict_y = predict_y.detach()
                predict_y = predict_y.cpu().float()
                predict_result = torch_F.softmax(predict_y, dim=1)
                predict_result = predict_result.numpy()[:,1]
                for item in predict_result:
                    neg_results.append(item)
            neg_result_all.append(neg_results)

        neg_result_all = np.array(neg_result_all)
        self.pred_result = np.mean(neg_result_all,axis=0)
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


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    def forward(self, x):
        x = self.linear(x)
        return x


def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r

def cal_infoconten(mat):
    info_conten = np.zeros(len(mat))
    for i in range(len(mat)):
        ratio = mat[i][0:4]/(sum(mat[i][0:4]))
        ratio_no0 = ratio[ratio!=0]
        info_conten[i] = 2+np.sum(ratio_no0 * np.log2(ratio_no0))
    return info_conten

@jit(nopython=True)
def cal_max_pos(seq,PPM_temp):
    res_dic = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    all_values = np.zeros(len(seq)+1-len(PPM_temp))
    for i in range(len(seq)+ 1 - len(PPM_temp)):
        value = 1
        for k in range(len(PPM_temp)):
            value *= PPM_temp[k][res_dic[seq[i + k]]]
        all_values[i] = value
    all_max_pos = np.argwhere(all_values==np.max(all_values))
    all_max_pos+=int(len(PPM_temp)/2)
    return all_max_pos, np.max(all_values)

def reverse_seq(read):
    read = list(read)
    read.reverse()
    lettercomplemt = {'A': 'T','T': 'A','G': 'C','C': 'G','N':'N'}
    read = [lettercomplemt[letter] for letter in read]
    read = "".join(read)
    return read

def get_motif_feature(seq, TF_PPM, all_TF_list, pre_backbone_seq, after_backbone_seq):
    check_seq = pre_backbone_seq[-24:] + seq + after_backbone_seq[0:24]
    back_bone_seq = pre_backbone_seq + seq + after_backbone_seq
    back_bone_seq_rev = reverse_seq(back_bone_seq)
    max_pos_each_TF = []
    max_value_each_TF = []
    mask_seq_each_TF = []
    for TF_name in all_TF_list:
        PPM_temp = TF_PPM[TF_name]
        all_max_pos, max_value = cal_max_pos(check_seq,PPM_temp)
        check_seq_rev = reverse_seq(check_seq)
        all_max_pos_rev,max_value_rev = cal_max_pos(check_seq_rev, PPM_temp)
        seq_to_save = []
        pos_to_save = [] 
        if max_value_rev==max_value:
            if max_value_rev==0:
                record_max_value = 0
                seq_to_save = []
                pos_to_save = []
            else:
                record_max_value = max_value
                for pos_temp in all_max_pos:
                    selected_seq_temp = back_bone_seq[86+pos_temp[0]-int(168/2):86+pos_temp[0]-int(168/2)+168]
                    seq_list = list(selected_seq_temp)
                    seq_list[int(168/2)-int(len(PPM_temp)/2):int(168/2)-int(len(PPM_temp)/2)+int(len(PPM_temp))]='N'*(int(len(PPM_temp)))
                    seq_mask = ''.join(seq_list)                 
                    seq_to_save.append(seq_mask)
                    pos_to_save.append(pos_temp[0]-24)
                for pos_temp in all_max_pos_rev:
                    selected_seq_temp = back_bone_seq_rev[86+pos_temp[0]-int(168/2):86+pos_temp[0]-int(168/2)+168]
                    seq_list = list(selected_seq_temp)
                    seq_list[int(168/2)-int(len(PPM_temp)/2):int(168/2)-int(len(PPM_temp)/2)+int(len(PPM_temp))]='N'*(int(len(PPM_temp)))
                    seq_mask = ''.join(seq_list)                 
                    seq_to_save.append(seq_mask)
                    pos_to_save.append(168-(pos_temp[0]-24))  
        elif max_value>max_value_rev:
            record_max_value = max_value
            for pos_temp in all_max_pos:
                selected_seq_temp = back_bone_seq[86+pos_temp[0]-int(168/2):86+pos_temp[0]-int(168/2)+168]
                seq_list = list(selected_seq_temp)
                seq_list[int(168/2)-int(len(PPM_temp)/2):int(168/2)-int(len(PPM_temp)/2)+int(len(PPM_temp))]='N'*(int(len(PPM_temp)))
                seq_mask = ''.join(seq_list)                 
                seq_to_save.append(seq_mask)
                pos_to_save.append(pos_temp[0]-24)
        elif max_value<max_value_rev:
            record_max_value = max_value_rev
            for pos_temp in all_max_pos_rev:
                selected_seq_temp = back_bone_seq_rev[86+pos_temp[0]-int(168/2):86+pos_temp[0]-int(168/2)+168]
                seq_list = list(selected_seq_temp)
                seq_list[int(168/2)-int(len(PPM_temp)/2):int(168/2)-int(len(PPM_temp)/2)+int(len(PPM_temp))]='N'*(int(len(PPM_temp)))
                seq_mask = ''.join(seq_list)                 
                seq_to_save.append(seq_mask)
                pos_to_save.append(168-(pos_temp[0]-24))  
        mask_seq_each_TF.append(seq_to_save)
        max_value_each_TF.append(record_max_value)
        max_pos_each_TF.append(pos_to_save)
    return [max_pos_each_TF,max_value_each_TF,mask_seq_each_TF]


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

@jit(nopython=True)
def process_pred(i,mot_feature_len,all_index_to_pred,pred_result_all,all_value_to_pred,all_pos_to_pred,max_env_values,max_core_values,max_value_pos):
    for j in range(mot_feature_len):
        index_temp = all_index_to_pred==j
        if sum(index_temp)==0:
            continue
        pred_result = pred_result_all[index_temp]
        max_value_temp = all_value_to_pred[index_temp][0]
        all_pos_temp = all_pos_to_pred[index_temp]
        selected_index = np.argwhere(pred_result==np.max(pred_result))
        max_env_values[j,i] = max(pred_result)
        max_core_values[j,i] = max_value_temp
        max_value_pos[j,i] = all_pos_temp[selected_index[0][0]]
    return max_env_values,max_core_values,max_value_pos

def gen_seq_to_predict(start_seq):
    nucleotides = ['A', 'T', 'C', 'G']
    mutations = []
    mutations.append(start_seq)
    for i in range(len(start_seq)):
        for nuc in nucleotides:
            if nuc != start_seq[i]:
                mutation = start_seq[:i] + nuc + start_seq[i+1:]
                mutations.append(mutation)
    return mutations


if __name__ == '__main__':
    mp.set_start_method('spawn')
    os.makedirs('./step3_optimize_CMV_log', exist_ok=True)
    TF_ppm = load_variavle('./data_dir/all_TF_ppm_mat.pkl')
    all_TF_list = load_variavle('./data_dir/all_TF_list.pkl')

    for TF_name in all_TF_list:
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
        TF_ppm[TF_name] = PPM_new

    # sequence around the CMV enhancer:
    pre_backbone_seq = 'CCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCCTCGGCGGCCAAGCTTAGACAC'
    after_backbone_seq = 'AGGCGTGTACGGTGGGAGGTCTATATAAGCAGAGCTCGTTTAGTGAACCGTCAGATCACTAGAAGCTTTATTGCGGTAGTTTATCACAGTTAAATTGCTAACGCAGTCAG'


    with open('./data_dir/MPRA_exp_mean.tsv') as f_in:
        all_exp = []
        for lines in f_in:
            line = lines.strip().split('\t')
            seq = line[0]
            exp = float(line[1])
            all_exp.append(exp)

    all_exp = np.array(all_exp)
    all_exp = np.array(np.log10(all_exp))

    np.random.seed(42)
    np.random.shuffle(all_exp)
    exp_tensor = torch.tensor(all_exp, dtype=torch.float32)
    dataset_size = len(exp_tensor)
    train_exp = exp_tensor[0:int(dataset_size*0.8)]
    val_exp = exp_tensor[int(dataset_size*0.8):int(dataset_size*0.9)]
    test_exp = exp_tensor[int(dataset_size*0.9):dataset_size]
    scaler = StandardScaler()
    train_exp = scaler.fit_transform(train_exp.cpu().reshape(-1, 1))
    train_exp = torch.tensor(train_exp, dtype=torch.float32)

    hidden_size1 = 256
    hidden_size2 = 128
    output_size = 1
    input_size = 198*3
    best_model = LinearRegressionModel(input_size, hidden_size1, hidden_size2, output_size).cuda()

    # the CMV enhancer seq to be optimized
    start_seq = 'TCAATATTGGCCATTAGCCATATTATTCATTGGTTATATAGCATAAATCAATATTGGCTATTGGCCATTGCATACGTTGTATCTATATCATAATATGTACATTTATATTGGCTCATGTCCAATATGACCGCCATGTTGGCATTGATTATTGACTAGTTATTAATAGTAATCAATTACGGGGTCATTAGTTCATAGCCCATATATGGAGTTCCGCGTTACATAACTTACGGTAAATGGCCCGCCTGGCTGACCGCCCAACGACCCCCGCCCATTGACGTCAATAATGACGTATGTTCCCATAGTAACGCCAATAGGGACTTTCCATTGACGTCAATGGGTGGAGTATTTACGGTAAACTGCCCACTTGGCAGTACATCAAGTGTATCATATGCCAAGTCCGCCCCCTATTGACGTCAATGACGGTAAATGGCCCGCCTGGCATTATGCCCAGTACATGACCTTACGGGACTTTCCTACTTGGCAGTACATCTACGTATTAGTCATCGCTATTACCATGGTGATGCGGTTTTGGCAGTACACCAATGGGCGTGGATAGCGGTTTGACTCACGGGGATTTCCAAGTCTCCACCCCATTGACGTCAATGGGAGTTTGTTTTGGCACCAAAATCAACGGGACTTTCCAAAATGTCGTAATAACCCCGCCCCGTTGACGCAAATGGGCGGT'

    # optimize the CMV enhancer by intruducing mutation greedily:
    for optimize_iter_num in range(15):
        seq_to_predict = gen_seq_to_predict(start_seq)
        seq_motif_feature = []     
        for seq in seq_to_predict:
            seq_motif_feature.append(get_motif_feature(seq, TF_ppm, all_TF_list, pre_backbone_seq, after_backbone_seq))

        max_env_values = np.zeros((len(seq_motif_feature),198))
        max_value_pos = np.zeros((len(seq_motif_feature),198))
        max_core_values = np.zeros((len(seq_motif_feature),198))

        for i,TF_name in enumerate(all_TF_list):
            analysis = Seq2ScalarTraining(TF_name)
            all_seq_to_pred = []
            all_pos_to_pred = []
            all_value_to_pred = []
            all_index_to_pred = []
            for j,item in enumerate(seq_motif_feature):
                all_pos_temp = item[0][i]
                all_seq_temp = item[2][i]
                max_value_temp = item[1][i]
                if max_value_temp==0:
                    continue
                for seq_temp, pos_temp in zip(all_seq_temp,all_pos_temp):
                    all_seq_to_pred.append(seq_temp)
                    all_pos_to_pred.append(pos_temp)
                    all_value_to_pred.append(max_value_temp)
                    all_index_to_pred.append(j)
            if len(all_seq_to_pred)==0:
                continue
            analysis.predict(all_seq_to_pred)
            pred_result_all = analysis.pred_result

            all_index_to_pred = np.array(all_index_to_pred)
            all_value_to_pred = np.array(all_value_to_pred)
            all_pos_to_pred = np.array(all_pos_to_pred)

            max_env_values,max_core_values,max_value_pos = process_pred(i,len(seq_motif_feature),all_index_to_pred,pred_result_all,all_value_to_pred,all_pos_to_pred,max_env_values,max_core_values,max_value_pos)


        column_max = np.zeros(198)
        for i,TF_name in enumerate(all_TF_list):
            TF_PPM_temp = TF_ppm[TF_name]
            max_ppm = 1
            for item in TF_PPM_temp:
                max_ppm*=max(item)
            column_max[i] = max_ppm

        max_core_normed = max_core_values.copy()
        max_core_normed /= column_max


        lr = 0.001
        mode = 'env_conc_core_conc_envcore'
        motif_score = np.concatenate((np.array(max_env_values),np.array(max_core_normed)),axis=1)
        motif_score = np.concatenate((motif_score,np.array(max_core_normed)*np.array(max_env_values)),axis=1)
        input_size = 198*3
        motif_score_tensor = torch.tensor(motif_score, dtype=torch.float32)
        test_motif = motif_score_tensor
        test_exp = torch.zeros(len(test_motif))    
        batch_size = 128
        test_dataset = TensorDataset(test_motif, test_exp)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        all_result = np.zeros((10,len(test_motif)))

        for tr_num in range(10):
            best_model_state = torch.load('./step2_model_with_TFBU_linear/model_'+mode + '_' +str(lr)+ '_'+ str(tr_num) +'.pth')
            best_model.load_state_dict(best_model_state)
            best_model.eval()
            label_list = []
            pred_list = []
            with torch.no_grad():
                for batch_motif, batch_exp in test_loader:
                    batch_motif = batch_motif.cuda()
                    outputs = best_model(batch_motif)
                    pred_result = outputs.cpu().numpy()
                    pred_list.extend(list(pred_result))
                    label_list.extend(list(batch_exp.numpy()))
            predicted_exp = np.array(pred_list)
            predicted_exp = scaler.inverse_transform(predicted_exp.reshape(-1, 1)).flatten()
            all_result[tr_num,:] = predicted_exp

        final_evaluate_result = np.mean(all_result,axis=0)
        with open('./step3_optimize_CMV_log/result_iter_' + str(optimize_iter_num)+'.txt','w') as f_out:
            for i in range(len(seq_to_predict)):
                f_out.write(str(final_evaluate_result[i])+'\t'+seq_to_predict[i] + '\n')

        if optimize_iter_num==0:
            with open('./step3_optimize_CMV_log/result_log_CMV.txt','a') as f_out:
                f_out.write(str(final_evaluate_result[0])+'\t'+seq_to_predict[0] + '\n')

        max_index = np.argmax(final_evaluate_result)
        if max_index==0:
            break
        else:
            start_seq = seq_to_predict[max_index]
            with open('./step3_optimize_CMV_log/result_log_CMV.txt','a') as f_out:
                f_out.write(str(final_evaluate_result[max_index])+'\t'+seq_to_predict[max_index] + '\n')


