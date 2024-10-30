import os
from Levenshtein import distance


if __name__ == '__main__':
    os.makedirs('./step2_HepG2_specific_fasta_edit50', exist_ok=True)
    TF_list = ['GATA2']
    pre = 'ACTGGCCGCTTGACG'
    after = 'CACTGCGGCTCCTGC'
    # adapter sequence in the MPRA experiment
    cut1 = 'GCGATCGC'
    cut2 = 'CGCTAGCG'
    # check the restriction enzyme cutting site in the sequences
    edit_thresh = 50

    for TF_name in TF_list:
        selected_seq = []
        out_fasta = open('./step2_HepG2_specific_fasta_edit50/'+TF_name+'.fasta','w')
        file_in = open('./step1_HepG2_specific_gc10_check/' + TF_name + '.txt','r')
        count_temp = 0
        for lines in file_in:
            if count_temp>=100:
                # generate 100 sequences for each aim
                break        
            line = lines.split()
            seq_temp = line[1]
            seq_mot = line[5]
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
                    # seq_final = pre + seq_temp + after # these sequences are not in MPRA library, no need for such adapter
                    seq_final = seq_temp
                    if (cut1 not in seq_final)&(cut2 not in seq_final)&('N' not in seq_final):
                        count_temp+=1
                        selected_seq.append(seq_temp)
                        out_fasta.write('>'+line[0])
                        out_fasta.write('\n')
                        out_fasta.write(seq_final + '\n')  

        out_fasta.close()


