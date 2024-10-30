import numpy as np
import random
import h5py
import numba
from numba import jit
import os


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

if __name__ == '__main__':
    os.makedirs('./data_dir', exist_ok=True)
    TF_list = ['GATA2']

    for TF_name in TF_list:
        pos_file_path = '../../0_prepare_training_sample/data_dir/step3_HepG2_pos_'+TF_name+'_data.bed'
        pos_bed_data = []
        with open(pos_file_path, 'r') as bed_file:
            for line in bed_file:
                pos_bed_data.append(line.strip().split('\t'))
        # pos denotes HepG2 samples neg denotes K562 samples
        neg_file_path = '../0_prepare_data_for_K562/data_dir/step3_K562_pos_'+TF_name+'_data.bed'

        neg_bed_data = []
        with open(neg_file_path, 'r') as bed_file:
            for line in bed_file:
                neg_bed_data.append(line.strip().split('\t'))


        all_pos_seqs = []
        all_neg_seqs = []
        pos_orig_num = 0
        neg_orig_num = 0

        pos_orig_num+=len(pos_bed_data)
        neg_orig_num+=len(neg_bed_data)

        pos_ratio_pre = [float(sublist[4]) for sublist in pos_bed_data]
        pos_seq_pre = [sublist[5] for sublist in pos_bed_data]
        pos_histome_pre = [sublist[7]+sublist[8] for sublist in pos_bed_data]


        neg_ratio_pre = [float(sublist[4]) for sublist in neg_bed_data]
        neg_seq_pre = [sublist[5] for sublist in neg_bed_data]
        neg_histome_pre = [sublist[7]+sublist[8] for sublist in neg_bed_data]


        intervals = [i/50 for i in range(51)]
        pos_indices = {}
        neg_indices = {}

        for i in range(len(intervals) - 1):
            pos_indices[i] = []
            neg_indices[i] = []

        for i in range(len(pos_ratio_pre)):
            for j in range(len(intervals) - 1):
                if intervals[j] <= pos_ratio_pre[i] <= intervals[j+1]:
                    pos_indices[j].append(i)
                    break

        for i in range(len(neg_ratio_pre)):
            for j in range(len(intervals) - 1):
                if intervals[j] <= neg_ratio_pre[i] <= intervals[j+1]:
                    neg_indices[j].append(i)
                    break

        selected_pos_indices = []
        selected_neg_indices = []
        for i in range(len(intervals) - 1):
            pos_indices_i = pos_indices[i]
            neg_indices_i = neg_indices[i]
            random.shuffle(pos_indices_i)
            random.shuffle(neg_indices_i)
            pos_histome_counts = {"00": 0, "01": 0, "10": 0, "11": 0}
            neg_histome_counts = {"00": 0, "01": 0, "10": 0, "11": 0}
            pos_indices_temp = {"00": [], "01": [], "10": [], "11": []}
            neg_indices_temp = {"00": [], "01": [], "10": [], "11": []}


            for pos_index in pos_indices_i:
                pos_histome_counts[pos_histome_pre[pos_index]] += 1
                pos_indices_temp[pos_histome_pre[pos_index]].append(pos_index)

            for neg_index in neg_indices_i:
                neg_histome_counts[neg_histome_pre[neg_index]] += 1
                neg_indices_temp[neg_histome_pre[neg_index]].append(neg_index)

            for histome_label in pos_histome_counts:
                count_to_selec = min(pos_histome_counts[histome_label],neg_histome_counts[histome_label])
                selected_pos_indices_temp = pos_indices_temp[histome_label][0:count_to_selec]
                selected_neg_indices_temp = neg_indices_temp[histome_label][0:count_to_selec]

                selected_pos_indices.extend(selected_pos_indices_temp)
                selected_neg_indices.extend(selected_neg_indices_temp)


        selected_pos_ratio_pre = [pos_ratio_pre[i] for i in selected_pos_indices]
        selected_pos_seq_pre = [pos_seq_pre[i] for i in selected_pos_indices]
        selected_pos_histome_pre = [pos_histome_pre[i] for i in selected_pos_indices]
        selected_pos_bed_data = [pos_bed_data[i] for i in selected_pos_indices]

        selected_neg_ratio_pre = [neg_ratio_pre[i] for i in selected_neg_indices]
        selected_neg_seq_pre = [neg_seq_pre[i] for i in selected_neg_indices]
        selected_neg_histome_pre = [neg_histome_pre[i] for i in selected_neg_indices]
        selected_neg_bed_data = [neg_bed_data[i] for i in selected_neg_indices]

        selec_record = open('./data_dir/step0_HepG2_vs_K562_selec_record.txt','a')
        selec_record.write(str(TF_name) + '\t' + str(pos_orig_num) + '\t' + str(len(selected_pos_ratio_pre)) + '\t' + str(len(selected_pos_ratio_pre)/pos_orig_num) + '\t' + str(neg_orig_num) + '\t' + str(len(selected_neg_ratio_pre)) + '\t' + str(len(selected_neg_ratio_pre)/neg_orig_num) + '\n')
        selec_record.close()

        f = h5py.File('./data_dir/step0_HepG2_vs_K562_motif_center_seq_mask_168_'+TF_name+'_data.h5', 'w')
        if len(selected_pos_seq_pre) != 0:
            seq_onehot = seq_to_onehot(numba.typed.List(selected_pos_seq_pre), 168)
        else:
            seq_onehot = []
        f.create_dataset('pos_' + TF_name, data=seq_onehot)
        if len(selected_neg_seq_pre) != 0:
            seq_onehot = seq_to_onehot(numba.typed.List(selected_neg_seq_pre), 168)
        else:
            seq_onehot = []
        f.create_dataset('neg_' + TF_name, data=seq_onehot)
        f.close()


        pos_region = open('./data_dir/step0_HepG2_vs_K562_pos_'+TF_name+'_data.bed','w')
        pos_ratio = []
        pos_seq = []
        for bed_data_temp in selected_pos_bed_data:
            for item in bed_data_temp:
                pos_region.write(item + '\t')
            pos_region.write('\n')
        pos_region.close()

        neg_region = open('./data_dir/step0_HepG2_vs_K562_neg_'+TF_name+'_data.bed','w')
        neg_ratio = []
        neg_seq = []
        for bed_data_temp in selected_neg_bed_data:
            for item in bed_data_temp:
                neg_region.write(item + '\t')
            neg_region.write('\n')
        neg_region.close()








