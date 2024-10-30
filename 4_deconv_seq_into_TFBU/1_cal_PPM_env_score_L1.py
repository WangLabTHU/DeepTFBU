import torch
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

def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r

def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


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




def main():
    seq_motif_feature = load_variavle('./data_dir/step0_seq_motif_features.pkl')
    TF_ppm = load_variavle('./data_dir/all_TF_ppm_mat.pkl')
    all_TF_list = load_variavle('./data_dir/all_TF_list.pkl')

    max_env_values = np.zeros((len(seq_motif_feature),198)) # 198 TFs for feature
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
        analysis.predict(all_seq_to_pred)
        pred_result_all = analysis.pred_result

        all_index_to_pred = np.array(all_index_to_pred)
        all_value_to_pred = np.array(all_value_to_pred)
        all_pos_to_pred = np.array(all_pos_to_pred)
        max_env_values,max_core_values,max_value_pos = process_pred(i,len(seq_motif_feature),all_index_to_pred,pred_result_all,all_value_to_pred,all_pos_to_pred,max_env_values,max_core_values,max_value_pos)


    save_variable(max_env_values,'./data_dir/step1_max_env_values_final.pkl')
    save_variable(max_value_pos,'./data_dir/step1_max_value_pos_final.pkl')
    save_variable(max_core_values,'./data_dir/step1_max_core_values_final.pkl')

    column_max = np.zeros(198)
    for i,TF_name in enumerate(all_TF_list):
        TF_ppm_temp = TF_ppm[TF_name]
        max_ppm = 1
        for item in TF_ppm_temp:
            max_ppm*=max(item)
        column_max[i] = max_ppm

    max_core_normed = max_core_values.copy()
    max_core_normed /= column_max

    save_variable(max_core_normed,'./data_dir/step1_max_core_normed_final.pkl')

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()

