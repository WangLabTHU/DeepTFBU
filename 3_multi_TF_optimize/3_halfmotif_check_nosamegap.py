import numpy as np
import os


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

if __name__ == '__main__':
    os.system('mkdir ./step3_halfmotif_result_norep/')
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
        seq_mask_gap1 = seq_mask_str



        out_file = open('./step3_halfmotif_result_norep/norep_final_result_gap1_'+TF_name+'.txt','w')
        selected_seq_list = []
        opti_iter_num = 300
        for f_num in range(opti_iter_num):
            file_num = opti_iter_num-1-f_num
            scores = np.loadtxt('./step1_halfmotif_GA_log/scores_' + TF_name +'/score_mask_'+str(file_num)+'.txt')
            seqs = np.load('./step1_halfmotif_GA_log/seqs_'+TF_name+'/seqs_mask_'+str(file_num)+'.npy')
            for score,seq in zip(scores,seqs):
                result = []
                current = ""
                for s, m in zip(seq, seq_mask_gap1):
                    if m == '0':
                        current += s
                    else:
                        if current:
                            result.append(current)
                            current = ""
                if current:
                    result.append(current)

                if len(result[0])==len(result[1]):
                    middle_elements = result[0:-1]
                else:
                    middle_elements = result[1:-1]
                if all(element == middle_elements[0] for element in middle_elements):
                    continue
                if seq in selected_seq_list:
                    continue
                selected_seq_list.append(seq)
                out_file.write(str(score) + '\t')
                out_file.write(seq + '\n')
        out_file.close()



