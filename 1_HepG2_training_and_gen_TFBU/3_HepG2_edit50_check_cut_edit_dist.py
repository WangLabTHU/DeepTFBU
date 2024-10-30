import os
from Levenshtein import distance


if __name__ == '__main__':
    os.makedirs('./step3_HepG2_fasta_edit50', exist_ok=True)
    TF_list = ['GATA2']
    pre = 'ACTGGCCGCTTGACG'
    after = 'CACTGCGGCTCCTGC'
    # adapter sequence in the MPRA experiment
    cut1 = 'GCGATCGC'
    cut2 = 'CGCTAGCG'
    # check the restriction enzyme cutting site in the sequences
    edit_thresh = 50
    total_train_num = 1 # it's set to be 10 for ten replicated trained model in the paper

    for TF_name in TF_list:
        selected_seq = []
        out_fasta = open('./step3_HepG2_fasta_edit50/'+TF_name+'.fasta','w')

        for aim in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
            file_in = open('./step2_HepG2_gc10_check/aim_' + str(aim) + '_' + TF_name + '.txt','r')
            count_temp = 0
            for lines in file_in:
                if count_temp>=10:
                    # generate 10 sequences for each aim
                    break        
                line = lines.split()
                seq_temp = line[1]
                seq_mot = line[total_train_num+2]
                if seq_temp[int(168/2)-int(len(seq_mot)/2):int(168/2)-int(len(seq_mot)/2)+int(len(seq_mot))]=='N'*(int(len(seq_mot))):
                    seq_list = list(seq_temp)
                    seq_list[int(168/2)-int(len(seq_mot)/2):int(168/2)-int(len(seq_mot)/2)+int(len(seq_mot))]=seq_mot
                    seq_temp = ''.join(seq_list)            
                    keep = True
                    for seq_to_query in selected_seq:
                        dist = distance(seq_temp, seq_to_query)
                        if dist < edit_thresh:
                            keep = False
                            break
                    if keep:
                        seq_final = pre + seq_temp + after
                        if (cut1 not in seq_final)&(cut2 not in seq_final)&('N' not in seq_final):
                            count_temp+=1
                            selected_seq.append(seq_temp)
                            value_temp = 0
                            for i in range(2,total_train_num+2):
                                value_temp+=float(line[i])
                            out_fasta.write('>'+str(value_temp/10) + '_aim_' + str(aim))
                            for i in range(total_train_num+4):
                                out_fasta.write('_'+line[i])
                            out_fasta.write('\n')
                            out_fasta.write(seq_final + '\n')  

        out_fasta.close()


