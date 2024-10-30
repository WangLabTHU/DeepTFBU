import numpy as np
import random
import h5py
import numba
from numba import jit


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
    TF_list = ['GATA2']

    for TF_name in TF_list:
        pos_file_path = './data_dir/step3_K562_pos_'+TF_name+'_data.bed'
        pos_bed_data = []
        with open(pos_file_path, 'r') as bed_file:
            for line in bed_file:
                pos_bed_data.append(line.strip().split('\t'))


        neg_file_path = './data_dir/step3_K562_neg_'+TF_name+'_data.bed'
        neg_bed_data = []
        with open(neg_file_path, 'r') as bed_file:
            for line in bed_file:
                neg_bed_data.append(line.strip().split('\t'))

        all_pos_seqs = []
        all_neg_seqs = []
        pos_orig_num = 0

        pos_orig_num+=len(pos_bed_data)
        pos_ratio_pre = [float(sublist[4]) for sublist in pos_bed_data]
        pos_seq_pre = [sublist[5] for sublist in pos_bed_data]
        pos_histome_pre = [sublist[7]+sublist[8] for sublist in pos_bed_data]

        neg_ratio_pre = [float(sublist[4]) for sublist in neg_bed_data]
        neg_seq_pre = [sublist[5] for sublist in neg_bed_data]
        neg_histome_pre = [sublist[7]+sublist[8] for sublist in neg_bed_data]

        interval_width = 1 / 50  # the bin length for GC ratio count
        intervals_pos = [0] * 50
        for gc_temp in pos_ratio_pre:
            interval_idx = int(gc_temp // interval_width)
            intervals_pos[interval_idx] += 1
        
        pos_hist_dist = {'00':0,'01':0,'10':0,'11':0}
        neg_hist_dist = {'00':0,'01':0,'10':0,'11':0}
        for histome_temp in pos_histome_pre:
            pos_hist_dist[histome_temp]+=1


        intervals_neg = [0] * 50 # set upper limit of negative sample counts acccording to positive samples
        neg_region = open('./data_dir/step4_K562_neg_'+TF_name+'_data.bed','a')
        neg_ratio = []
        neg_seq = []
        for gc_temp,seq_temp,histome_temp,bed_data_temp in zip(neg_ratio_pre,neg_seq_pre,neg_histome_pre,neg_bed_data):
            interval_idx = int(gc_temp // interval_width)
            if (intervals_neg[interval_idx]<intervals_pos[interval_idx])&(neg_hist_dist[histome_temp]<pos_hist_dist[histome_temp]):
                neg_seq.append(seq_temp)
                neg_ratio.append(gc_temp)
                for item in bed_data_temp:
                    neg_region.write(item + '\t')
                neg_region.write('\n')
                intervals_neg[interval_idx]+=1
                neg_hist_dist[histome_temp]+=1
        neg_region.close()


        # select positive samples according to negative samples
        intervals_pos_new = [0] * 50
        for key in pos_hist_dist.keys():
            pos_hist_dist[key]=0
        random.seed(42)
        random_indices = random.sample(range(len(pos_ratio_pre)), len(pos_ratio_pre))
        pos_ratio = []
        pos_seq = []
        
        pos_region = open('./data_dir/step4_K562_pos_'+TF_name+'_data.bed','a')
        for i in random_indices:
            gc_temp = pos_ratio_pre[i]
            seq_temp = pos_seq_pre[i]
            histome_temp = pos_histome_pre[i]
            interval_idx = int(gc_temp // interval_width)
            if (intervals_pos_new[interval_idx]<intervals_neg[interval_idx])&(pos_hist_dist[histome_temp]<neg_hist_dist[histome_temp]):
                pos_seq.append(seq_temp)
                pos_ratio.append(gc_temp)
                for item in pos_bed_data[i]:
                    pos_region.write(item + '\t')
                pos_region.write('\n')
                intervals_pos_new[interval_idx]+=1
                pos_hist_dist[histome_temp]+=1
        pos_region.close()

        all_pos_seqs.extend(pos_seq)
        all_neg_seqs.extend(neg_seq)

        selec_record = open('./data_dir/step4_K562_selec_record.txt','a')
        selec_record.write(str(TF_name) + '\t' + str(pos_orig_num) + '\t' + str(len(all_pos_seqs)) + '\t' + str(len(all_pos_seqs)/pos_orig_num) + '\n')
        selec_record.close()

        f = h5py.File('./data_dir/step4_K562_motif_center_seq_mask_168_'+TF_name+'_data.h5', 'w')

        if len(all_pos_seqs) != 0:
            seq_onehot = seq_to_onehot(numba.typed.List(all_pos_seqs), 168)
        else:
            seq_onehot = []
        f.create_dataset('pos_' + TF_name, data=seq_onehot)
        if len(all_neg_seqs) != 0:
            seq_onehot = seq_to_onehot(numba.typed.List(all_neg_seqs), 168)
        else:
            seq_onehot = []
        f.create_dataset('neg_' + TF_name, data=seq_onehot)
        f.close()

