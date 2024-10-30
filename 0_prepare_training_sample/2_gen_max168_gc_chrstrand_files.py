from pyfaidx import Fasta
import numpy as np
from numba import jit



def cal_infoconten(mat):
    info_conten = np.zeros(len(mat))
    for i in range(len(mat)):
        ratio = mat[i][0:4]/(sum(mat[i][0:4]))
        ratio_no0 = ratio[ratio!=0]
        info_conten[i] = 2+np.sum(ratio_no0 * np.log2(ratio_no0))
    return info_conten


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
def cal_max_pos(seq,PPM_temp,pos_orig):
    res_dic = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    all_values = np.zeros(len(seq)+1-len(PPM_temp))
    for i in range(len(seq)+ 1 - len(PPM_temp)):
        value = 1
        for k in range(len(PPM_temp)):
            value *= PPM_temp[k][res_dic[seq[i + k]]]
        all_values[i] = value
    all_max_pos = np.argwhere(all_values==np.max(all_values))
    all_max_pos+=int(len(PPM_temp)/2)
    if len(all_max_pos)>1:
        max_pos = all_max_pos[np.argmin(np.abs(all_max_pos - pos_orig))]
    else:
        max_pos = all_max_pos[0]
    return max_pos, np.max(all_values)

def reverse_seq(read):
    read = list(read)
    read.reverse()
    lettercomplemt = {'A': 'T','T': 'A','G': 'C','C': 'G','N':'N'}
    read = [lettercomplemt[letter] for letter in read]
    read = "".join(read)
    return read



if __name__ == '__main__':
    genes = Fasta('/home/jqli/hg38/hg38.fa') # reference genome dir
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
        TF_ppm[TF_name] = PPM_new


    cutoff_length = 168
    for TF_name in TF_list:
        print(TF_name)
        ## for positive data:
        PPM_temp = TF_ppm[TF_name]
        seq_part = []
        pos_regions = open('./data_dir/step2_HepG2_pos_'+TF_name+'_data.bed', 'w')
        data_file = open('./data_dir/HepG2_ChIP_' + TF_name + '.bed')
        gc_pos = []
        for lines in data_file:
            line = lines.split()
            seq_orig_info = genes[line[0]][int(line[1]):int(line[2])]
            seq_orig = seq_orig_info.seq.upper()
            pos_orig = int(line[3])
            max_pos, max_value = cal_max_pos(seq_orig, PPM_temp, pos_orig)
            max_pos = int(max_pos)

            seq_orig_rev = reverse_seq(seq_orig)
            pos_orig_rev = len(seq_orig_rev)-pos_orig
            max_pos_rev,max_value_rev = cal_max_pos(seq_orig_rev, PPM_temp, pos_orig_rev)
            max_pos_rev = int(max_pos_rev)

            if max_value>max_value_rev:
                strand_temp = '+'
                start_temp = int(line[1]) + max_pos - int(cutoff_length / 2)
                end_temp = int(line[1]) + max_pos + int(cutoff_length / 2)
                seq_final_info = genes[line[0]][int(line[1]) + max_pos - int(cutoff_length / 2):int(line[1]) + max_pos + int(
                    cutoff_length / 2)]
                seq_temp = seq_final_info.seq.upper()
            else:
                strand_temp = '-'
                start_temp = int(line[2])-max_pos_rev-int(cutoff_length / 2)
                end_temp = int(line[2])-max_pos_rev+int(cutoff_length / 2)
                seq_final_info_0 = genes[line[0]][int(line[2])-max_pos_rev-int(cutoff_length / 2):int(line[2])-max_pos_rev+int(cutoff_length / 2)]
                seq_temp_0 = seq_final_info_0.seq.upper()
                seq_temp = reverse_seq(seq_temp_0)

            if (len(seq_temp)==cutoff_length):
                masked = seq_temp[int(cutoff_length/2)-int(len(PPM_temp)/2):int(cutoff_length/2)-int(len(PPM_temp)/2)+int(len(PPM_temp))]
                seq_list = list(seq_temp)
                seq_list[int(cutoff_length/2)-int(len(PPM_temp)/2):int(cutoff_length/2)-int(len(PPM_temp)/2)+int(len(PPM_temp))]='N'*(int(len(PPM_temp)))
                seq_final = ''.join(seq_list)            
                seq_part.append(seq_final)
                at_count = 0
                gc_count = 0
                for item in seq_final:
                    if item in ['A', 'T', 'a', 't']:
                        at_count += 1
                    elif item in ['G', 'C', 'g', 'c']:
                        gc_count += 1
                gc_ratio = gc_count / (at_count + gc_count)
                gc_pos.append(gc_ratio)
                if max_value>max_value_rev:
                    pos_regions.write(line[0] + '\t' + str(int(line[1]) + max_pos - int(cutoff_length / 2)) + '\t' + str(int(line[1]) + max_pos + int(cutoff_length / 2)) + '\t+\t' + str(gc_ratio) +'\t'+ seq_final + '\t' +masked + '\n')
                else:
                    pos_regions.write(line[0] + '\t' + str(int(line[2])-max_pos_rev-int(cutoff_length / 2)) + '\t' + str(int(line[2])-max_pos_rev+int(cutoff_length / 2)) + '\t-\t' + str(gc_ratio) +'\t'+ seq_final + '\t' +masked + '\n')
        pos_regions.close()


        ## for negative data:
        PPM_temp = TF_ppm[TF_name]
        seq_part = []
        neg_regions = open('./data_dir/step2_HepG2_neg_'+TF_name+'_data.bed', 'w')
        data_file = open('./data_dir/step1_HepG2_negative_' + TF_name + '.bed')
        for lines in data_file:
            line = lines.split()
            seq_orig_info = genes[line[0]][int(line[1]):int(line[2])]
            seq_orig = seq_orig_info.seq.upper()
            pos_orig = int((int(line[2])-int(line[1]))/2) #set the middle position as peak position for negative samples
            max_pos, max_value = cal_max_pos(seq_orig, PPM_temp, pos_orig)
            max_pos = int(max_pos)

            seq_orig_rev = reverse_seq(seq_orig)
            pos_orig_rev = len(seq_orig_rev)-pos_orig
            max_pos_rev,max_value_rev = cal_max_pos(seq_orig_rev, PPM_temp, pos_orig_rev)
            max_pos_rev = int(max_pos_rev)
            
            if max_value>max_value_rev:
                strand_temp = '+'
                start_temp = int(line[1]) + max_pos - int(cutoff_length / 2)
                end_temp = int(line[1]) + max_pos + int(cutoff_length / 2)
                seq_final_info = genes[line[0]][int(line[1]) + max_pos - int(cutoff_length / 2):int(line[1]) + max_pos + int(
                    cutoff_length / 2)]
                seq_temp = seq_final_info.seq.upper()
            else:
                strand_temp = '-'
                start_temp = int(line[2])-max_pos_rev-int(cutoff_length / 2)
                end_temp = int(line[2])-max_pos_rev+int(cutoff_length / 2)
                seq_final_info_0 = genes[line[0]][int(line[2])-max_pos_rev-int(cutoff_length / 2):int(line[2])-max_pos_rev+int(cutoff_length / 2)]
                seq_temp_0 = seq_final_info_0.seq.upper()
                seq_temp = reverse_seq(seq_temp_0)

            if (len(seq_temp)==cutoff_length):
                masked = seq_temp[int(cutoff_length/2)-int(len(PPM_temp)/2):int(cutoff_length/2)-int(len(PPM_temp)/2)+int(len(PPM_temp))]
                seq_list = list(seq_temp)
                seq_list[int(cutoff_length/2)-int(len(PPM_temp)/2):int(cutoff_length/2)-int(len(PPM_temp)/2)+int(len(PPM_temp))]='N'*(int(len(PPM_temp)))
                seq_final = ''.join(seq_list)
                at_count = 0
                gc_count = 0
                for item in seq_final:
                    if item in ['A', 'T', 'a', 't']:
                        at_count += 1
                    elif item in ['G', 'C', 'g', 'c']:
                        gc_count += 1
                gc_ratio = gc_count / (at_count + gc_count)

                if max_value>max_value_rev:
                    neg_regions.write(line[0] + '\t' + str(int(line[1]) + max_pos - int(cutoff_length / 2)) + '\t' + str(int(line[1]) + max_pos + int(cutoff_length / 2)) + '\t+\t' + str(gc_ratio) +'\t'+ seq_final + '\t' +masked + '\n')
                else:
                    neg_regions.write(line[0] + '\t' + str(int(line[2])-max_pos_rev-int(cutoff_length / 2)) + '\t' + str(int(line[2])-max_pos_rev+int(cutoff_length / 2)) + '\t-\t' + str(gc_ratio) +'\t'+ seq_final + '\t' +masked + '\n')
        neg_regions.close()








