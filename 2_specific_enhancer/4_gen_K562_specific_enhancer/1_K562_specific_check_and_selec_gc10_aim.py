import numpy as np
import h5py
import os


def cal_infoconten(mat):
    info_conten = np.zeros(len(mat))
    for i in range(len(mat)):
        ratio = mat[i][0:4]/(sum(mat[i][0:4]))
        ratio_no0 = ratio[ratio!=0]
        info_conten[i] = 2+np.sum(ratio_no0 * np.log2(ratio_no0))
    return info_conten


if __name__ == '__main__':
    os.makedirs('./step1_K562_specific_gc10_check', exist_ok=True)
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


    TF_motif = {}
    for TF_name in TF_list:
        letter_list = ['A','C','G','T']
        motif_seq = ''
        PPM_temp = TF_ppm[TF_name]
        for temp_vec in PPM_temp:
            pos_temp = np.argmax(temp_vec)
            motif_seq = motif_seq + letter_list[pos_temp]
        TF_motif[TF_name] = motif_seq
    sorted_TF_motif = {k: v for k, v in sorted(TF_motif.items())}


    for TF_name in TF_list:
        gc_in_seq_pos = []
        gc_in_seq_neg = []
        f = h5py.File('../1_prepare_data_for_HepG2_vs_K562/data_dir/step0_HepG2_vs_K562_motif_center_seq_mask_'+str(168)+'_'+TF_name+'_data.h5', 'r')
        neg_seqs = f['neg_' + TF_name][:]
        pos_seqs = f['pos_' + TF_name][:]
        for item in pos_seqs:
            temp = np.sum(item,axis=0)
            at_count = temp[0] + temp[3]
            gc_count = temp[1] + temp[2]
            gc_ratio = gc_count / (at_count + gc_count)
            gc_in_seq_pos.append(gc_ratio)
        for item in neg_seqs:
            temp = np.sum(item,axis=0)
            at_count = temp[0] + temp[3]
            gc_count = temp[1] + temp[2]
            gc_ratio = gc_count / (at_count + gc_count)
            gc_in_seq_neg.append(gc_ratio)

        pos_gc = np.mean(gc_in_seq_pos)
        neg_gc = np.mean(gc_in_seq_neg)

        if abs(pos_gc-neg_gc)<0.06:
            target_gc = (pos_gc+neg_gc)/2
            print(target_gc)
        else:
            raise Exception("pos neg gc not equal")

        
        out_file = open('./step1_K562_specific_gc10_check/' + TF_name + '.txt','w')
        selected_seq_list = []
        opti_iter_num = 300
        for f_num in range(opti_iter_num):
            file_num = opti_iter_num-1-f_num
            loss = open('./step0_K562_specific_GA_log/score_mask_' + TF_name +'_'+str(file_num)+'.txt')
            all_scores = open('./step0_K562_specific_GA_log/score_all_mask_' + TF_name +'_'+str(file_num)+'.txt')
            seqs = np.load('./step0_K562_specific_GA_log/seqs_mask_' + TF_name +'_'+str(file_num)+'.npy')
            for loss_temps,all_score_temps,seq in zip(loss,all_scores,seqs):
                loss_temp = loss_temps.split()
                all_score_temp = all_score_temps.split()
                if seq in selected_seq_list:
                    continue
                if TF_motif[TF_name] in seq:  # remove NC-TFBU with consensus motif in it
                    continue    
                gc_temp = sum([1 for i in range(len(seq)) if ((seq[i] == 'C') | (seq[i] == 'G'))])
                at_temp = sum([1 for i in range(len(seq)) if ((seq[i] == 'A') | (seq[i] == 'T'))])
                gc_cont_temp = (gc_temp/(gc_temp+at_temp))
                if abs(gc_cont_temp-target_gc)>=0.1:
                    continue
                selected_seq_list.append(seq)
                out_file.write(loss_temp[0] + '\t')
                out_file.write(seq + '\t')
                for item in all_score_temp:
                    out_file.write(item + '\t')
                out_file.write(TF_motif[TF_name]+'\t')
                out_file.write(str(gc_cont_temp) + '\n')
        out_file.close()

