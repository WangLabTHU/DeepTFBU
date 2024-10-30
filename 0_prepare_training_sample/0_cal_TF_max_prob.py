import numpy as np
from numba import jit
import pickle
from pyfaidx import Fasta


@jit(nopython=True)
def cal_max_value(seq,PPM_temp):
    res_dic = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    all_values = np.zeros(len(seq)+1-len(PPM_temp))
    for i in range(len(seq)+ 1 - len(PPM_temp)):
        value = 1
        for k in range(len(PPM_temp)):
            value *= PPM_temp[k][res_dic[seq[i + k]]]
        all_values[i] = value
    return np.max(all_values)

def reverse_seq(read):
    read = list(read)
    read.reverse()
    lettercomplemt = {'A': 'T','T': 'A','G': 'C','C': 'G','N':'N'}
    read = [lettercomplemt[letter] for letter in read]
    read = "".join(read)
    return read


def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


if __name__ == '__main__':
    TF_list = ['GATA2'] # GATA2 as example
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


    all_chr = []
    all_start = []
    all_end = []
    all_max_value = {}
    for TF_name in TF_list:
        all_max_value[TF_name] = []

    bed_file = open('./data_dir/HepG2_ATAC.bed') # ATAC-seq bed narrow peak file
    genes = Fasta('/home/jqli/hg38/hg38.fa') # reference genome dir

    for lines in bed_file:
        line = lines.split()
        sequence = genes[line[0]][int(line[1]):int(line[2])]
        all_chr.append(line[0])
        all_start.append(int(line[1]))
        all_end.append(int(line[2]))
        seq_temp = sequence.seq.upper()
        seq_temp_rev = reverse_seq(seq_temp)
        for TF_name in TF_list:
            max_temp1 = cal_max_value(seq_temp, TF_ppm[TF_name])
            max_temp2 = cal_max_value(seq_temp_rev, TF_ppm[TF_name])
            max_temp = max(max_temp1,max_temp2)
            all_max_value[TF_name].append(max_temp)


    save_variable(all_chr,'./data_dir/step0_HepG2_chrs.pkl')
    save_variable(all_start,'./data_dir/step0_HepG2_start.pkl')
    save_variable(all_end,'./data_dir/step0_HepG2_end.pkl')
    for TF_name in TF_list:
        save_variable(all_max_value[TF_name], './data_dir/step0_HepG2_max_for_'+TF_name+'.pkl')





