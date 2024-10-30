import numpy as np
from numba import jit
import pickle


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
    check_seq = pre_backbone_seq[-24:] + seq + after_backbone_seq[0:24]
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



if __name__ == '__main__':
    TF_ppm = load_variavle('./data_dir/all_TF_ppm_mat.pkl')
    all_TF_list = load_variavle('./data_dir/all_TF_list.pkl')
    # sequence on the plasmid around designed enhancer in MPRA:
    pre_backbone_seq = 'ATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG'
    after_backbone_seq = 'CACTGCGGCTCCTGCGATAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCAC'
    with open('./data_dir/MPRA_exp_mean.tsv') as f_in:
        seq_motif_feature = []
        for lines in f_in:
            line = lines.strip().split('\t')
            seq = line[0]
            exp = float(line[1])
            seq_motif_feature.append(get_motif_feature(seq, TF_ppm, all_TF_list, pre_backbone_seq, after_backbone_seq))
    save_variable(seq_motif_feature,'./data_dir/step0_seq_motif_features.pkl')







