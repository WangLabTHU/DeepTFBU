import os
from Levenshtein import distance



if __name__ == '__main__':
    os.system('mkdir ./step4_selceted_result/')
    TF_list = ['GATA2'] 
    
    edit_thresh = 5
    for TF_name in TF_list:
        selected_seq = []
        out_fasta = open('./step4_selceted_result/multimotif_edit_'+str(edit_thresh)+'_'+TF_name+'.txt','w')
        file_in = open('./step2_multimotif_result_norep/norep_final_result_' + TF_name + '.txt','r')
        count_temp = 0
        for lines in file_in:
            if count_temp>=10:
                # get 10 optimizd sequences
                break
            line = lines.split()
            seq_temp = line[1]

            keep = True
            for seq_to_query in selected_seq:
                dist = distance(seq_temp, seq_to_query)
                if dist < edit_thresh:
                    keep = False
                    break
            if keep:
                selected_seq.append(seq_temp)
                count_temp+=1
                out_fasta.write(lines)
        out_fasta.close()

    edit_thresh = 10
    for TF_name in TF_list:
        selected_seq = []
        out_fasta = open('./step4_selceted_result/halfmotif_edit_'+str(edit_thresh)+'_'+TF_name+'.txt','w')
        file_in = open('./step3_halfmotif_result_norep/norep_final_result_gap1_' + TF_name + '.txt','r')
        count_temp = 0
        for lines in file_in:
            if count_temp>=10:
                # get 10 optimizd sequences
                break
            line = lines.split()
            seq_temp = line[1]

            keep = True
            for seq_to_query in selected_seq:
                dist = distance(seq_temp, seq_to_query)
                if dist < edit_thresh:
                    keep = False
                    break
            if keep:
                selected_seq.append(seq_temp)
                count_temp+=1
                out_fasta.write(lines)
        out_fasta.close()

