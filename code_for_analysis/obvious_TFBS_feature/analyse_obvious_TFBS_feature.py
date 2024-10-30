import numpy as np
from numba import jit
import pickle
import matplotlib.pyplot as plt
import umap
import seaborn as sns
import csv


def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r

TF_ppm = load_variavle('all_TF_full_ppm.pkl')
all_TF_list = load_variavle('TF_list.pkl')

#sequence before and after enhancer on plasmid
pre_backbone_seq = 'ATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG'
after_backbone_seq = 'CACTGCGGCTCCTGCGATAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCAC'



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


def get_motif_feature(seq, TF_ppm, all_TF_list, pre_backbone_seq, after_backbone_seq):
    check_seq = 'TCTGGCCTAACTGGCCGCTTGACG' + seq + 'CACTGCGGCTCCTGCGATAGAGGG'
    back_bone_seq = pre_backbone_seq + seq + after_backbone_seq
    back_bone_seq_rev = reverse_seq(back_bone_seq)
    max_pos_each_TF = []
    max_value_each_TF = []
    mask_seq_each_TF = []
    for TF_name in all_TF_list:
        PPM_temp = TF_ppm[TF_name]
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


column_max = np.zeros(607)
for i,TF_name in enumerate(all_TF_list):
    TF_ppm_temp = TF_ppm[TF_name]
    max_ppm = 1
    for item in TF_ppm_temp:
        max_ppm*=max(item)
    column_max[i] = max_ppm


def top_sort_tsv_by_second_column(tsv_file):
    data = []
    with open(tsv_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            data.append(row)
    sorted_data = sorted(data, key=lambda x: float(x[1]), reverse=True)
    return sorted_data

def bottom_sort_tsv_by_second_column(tsv_file):
    data = []
    with open(tsv_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            data.append(row)
    sorted_data = sorted(data, key=lambda x: float(x[1]), reverse=False)
    return sorted_data


seq_mot_dict = {'ELF1':'CAGGAAGTG','HNF1A':'GTTAATGATTAAC','HNF4A':'CAAAGTCCA'}

for TF_name in ['ELF1','HNF1A','HNF4A']:
    seq_mot = seq_mot_dict[TF_name]
    top_sort = top_sort_tsv_by_second_column('../identify_obvious_TFBS/'+TF_name+'_with_motif.tsv')
    bottom_sort = bottom_sort_tsv_by_second_column('../identify_obvious_TFBS/'+TF_name+'_with_motif.tsv')
    top_100 = [row[0] for row in top_sort[:100]]
    bottom_100 = [row[0] for row in bottom_sort[:100]]
    top_motif_features = []
    for seq in top_100:
        if len(seq)==168:
            if seq[int(168/2)-int(len(seq_mot)/2):int(168/2)-int(len(seq_mot)/2)+int(len(seq_mot))]==seq_mot:
                seq_list = list(seq)
                seq_list[int(168/2)-int(len(seq_mot)/2):int(168/2)-int(len(seq_mot)/2)+int(len(seq_mot))]='N'*(int(len(seq_mot)))
                seq_mask = ''.join(seq_list)                       
                result = get_motif_feature(seq_mask, TF_ppm, all_TF_list, pre_backbone_seq, after_backbone_seq)
                result[1] = result[1]/column_max
                top_motif_features.append(result)                

    bottom_motif_features = []
    for seq in bottom_100:
        if len(seq)==168:
            if seq[int(168/2)-int(len(seq_mot)/2):int(168/2)-int(len(seq_mot)/2)+int(len(seq_mot))]==seq_mot:
                seq_list = list(seq)
                seq_list[int(168/2)-int(len(seq_mot)/2):int(168/2)-int(len(seq_mot)/2)+int(len(seq_mot))]='N'*(int(len(seq_mot)))
                seq_mask = ''.join(seq_list)                       
                result = get_motif_feature(seq_mask, TF_ppm, all_TF_list, pre_backbone_seq, after_backbone_seq)
                result[1] = result[1]/column_max
                bottom_motif_features.append(result)    


    feature_bottom = []
    for item in bottom_motif_features:
        feature_bottom.append(item[1])
    feature_top = []
    for item in top_motif_features:
        feature_top.append(item[1])


    group_bottom = np.array(feature_bottom)
    group_top = np.array(feature_top)
    all_samples = np.vstack((group_bottom, group_top))

    group_bottom_tr = group_bottom.transpose()
    group_top_tr = group_top.transpose()
    bottom_feature = np.sum(group_bottom_tr,axis=1)
    top_feature = np.sum(group_top_tr,axis=1)
    diff = top_feature-bottom_feature
    sorted_indices = np.argsort(np.abs(diff))
    descending_indices = sorted_indices[::-1]
    with open(f'{TF_name}_diff_TFBS_score_top30.tsv','w') as f_out:
        for i in range(30):
            f_out.write(str(diff[descending_indices[i]]) + '\t' + all_TF_list[descending_indices[i]]+'\n')


    labels = np.array([0] * len(group_bottom) + [1] * len(group_top))
    umap_model = umap.UMAP(n_components=2,random_state=42)
    umap_result = umap_model.fit_transform(all_samples)
    low_emb = umap_result[0:len(group_bottom)]
    high_emb = umap_result[len(group_bottom):]
    fig, ax = plt.subplots(figsize=(5, 5))


    sns.kdeplot(high_emb[:, 0], high_emb[:, 1], shade=False,
                colors='#ee7b6c',
                thresh=0.4,
                cmap=None, linewidths=1, alpha=0.7)

    sns.kdeplot(low_emb[:, 0], low_emb[:, 1], shade=False,
                colors='#8eb8d9',
                thresh=0.4,
                cmap=None, linewidths=1, alpha=0.7)

    plt.scatter(high_emb[:, 0], high_emb[:, 1], c='#ee7b6c', alpha=0.2, s=6,
                    linewidth=0, label='High Activity')
    plt.scatter(low_emb[:, 0], low_emb[:, 1], c='#8eb8d9', alpha=0.5, s=6,
                    linewidth=0, label='Low Activity')

    plt.legend(loc='upper right')

    plt.tick_params(
        axis='x',
        which='both',
        bottom=False, 
        top=False,
        labelbottom=False)
    plt.tick_params(
        axis='y',
        which='both',
        left=False,
        right=False,
        labelleft=False)

    plt.xlabel('Umap 1')
    plt.ylabel('Umap 2')
    plt.title("TFBS feature in "+TF_name + "'"+"s TFBS-context")
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    plt.savefig(TF_name + '_motif_feature.png')








