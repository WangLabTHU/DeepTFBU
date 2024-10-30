# Code for identifying obvious TFBS
import os

seq_mot_dic = {'ELF1':'CAGGAAGTG', 'HNF1A':'GTTAATGATTAAC', 'HNF4A':'CAAAGTCCA'} #consensus sequence of used motif
TF_list = ['ELF1','HNF1A','HNF4A']

for TF_temp in TF_list:
    seq_mot = seq_mot_dic[TF_temp]
    os.makedirs(f'fimo_{TF_temp}', exist_ok=True)
    with open('../Supplementary_Table1_MPRA_result_3TF.tsv','r') as f,open(f'{TF_temp}_with_motif.tsv','w') as f_out_with, open(f'{TF_temp}_without_motif.tsv','w') as f_out_without:
        for lines in f:
            line = lines.split()
            if ('aim' in line[2])or('pos' in line[2])or('neg' in line[2]):
                seq = line[0]
                if len(seq)==168:
                    if seq[int(168/2)-int(len(seq_mot)/2):int(168/2)-int(len(seq_mot)/2)+int(len(seq_mot))]==seq_mot:
                        seq_list = list(seq)
                        seq_list[int(168/2)-int(len(seq_mot)/2):int(168/2)-int(len(seq_mot)/2)+int(len(seq_mot))]='N'*(int(len(seq_mot)))
                        seq_mask = ''.join(seq_list)              
                        with open('temp.fasta','w') as f_temp: #Establish temporary file for calculating corrected p-value.
                            f_temp.write('>'+line[2]+'\n')
                            f_temp.write(seq_mask)
                        os.system(f'fimo --max-stored-scores 100000000 --verbosity 1 --qv-thresh --thresh 1e-4 --oc ./fimo_{TF_temp}/{line[2]} PFM_JASPAR2022_meme.txt temp.fasta')
                        with open(f'./fimo_{TF_temp}/'+line[2]+'/fimo.tsv') as f_fimo:
                            temp_lines = f_fimo.readline()
                            temp_line = temp_lines.split()
                            if len(temp_line)>3: #found matched obvious TFBS
                                f_out_with.write(lines)
                            else:
                                f_out_without.write(lines)


with open('Lib118_without_motif.tsv','w') as f_out:
    with open('ELF1_without_motif.tsv') as f_in:
        for lines in f_in:
            line = lines.split()
            if len(line)>1:
                f_out.write(lines)
    with open('HNF1A_without_motif.tsv') as f_in:
        for lines in f_in:
            line = lines.split()
            if len(line)>1:
                f_out.write(lines)
    with open('HNF4A_without_motif.tsv') as f_in:
        for lines in f_in:
            line = lines.split()
            if len(line)>1:
                f_out.write(lines)