import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import mannwhitneyu

top_ratio = 0.1
seq_mot_dic = {'ELF1':'CAGGAAGTG', 'HNF1A':'GTTAATGATTAAC', 'HNF4A':'CAAAGTCCA'} #consensus sequence of used motif
TF_list = ['ELF1','HNF1A','HNF4A']

for TF_temp in TF_list:
    seq_mot = seq_mot_dic[TF_temp]
    temp_seqs = []
    temp_values = []
    temp_masked = []
    with open('../Supplementary_Table1_MPRA_result_3TF.tsv','r') as f:
        for lines in f:
            line = lines.split()
            if ('aim' in line[2])or('pos' in line[2])or('neg' in line[2]):
                seq = line[0]
                value = float(line[1])
                if len(seq)==168:
                    if seq[int(168/2)-int(len(seq_mot)/2):int(168/2)-int(len(seq_mot)/2)+int(len(seq_mot))]==seq_mot:
                        temp_seqs.append(seq)
                        temp_values.append(value)
                        seq_list = list(seq)
                        seq_list[int(168/2)-int(len(seq_mot)/2):int(168/2)-int(len(seq_mot)/2)+int(len(seq_mot))]='N'*(int(len(seq_mot)))
                        seq_mask = ''.join(seq_list)                       
                        temp_masked.append(seq_mask)
    sorted_lst = sorted(temp_values)
    seq_num = len(sorted_lst)

    with open(f'./{TF_temp}_masked_top_seq.fasta','w') as f_out:
        for i,(value, seq_mask) in enumerate(zip(temp_values,temp_masked)):
            if value>sorted_lst[-1-int(seq_num*top_ratio)]:
                f_out.write('>' +str(i) + '_' + str(np.log10(value))+'\n' + seq_mask + '\n')

    with open(f'./{TF_temp}_masked_bottom_seq.fasta','w') as f_out:
        for i,(value, seq_mask) in enumerate(zip(temp_values,temp_masked)):
            if value<sorted_lst[int(seq_num*top_ratio)]:
                f_out.write('>' +str(i) + '_'  + str(np.log10(value))+'\n' + seq_mask + '\n')

TF_name = 'ELF1'
os.system(f'fimo --max-stored-scores 100000000 --text --thresh 1e-2  {TF_name}_meme.txt {TF_name}_masked_bottom_seq.fasta >{TF_name}_bottom.tsv')
os.system(f'fimo --max-stored-scores 100000000 --text --thresh 1e-2  {TF_name}_meme.txt {TF_name}_masked_top_seq.fasta >{TF_name}_top.tsv')


TF_name = 'HNF1A'
os.system(f'fimo --max-stored-scores 100000000 --text --thresh 1e-2  {TF_name}_meme.txt {TF_name}_masked_bottom_seq.fasta >{TF_name}_bottom.tsv')
os.system(f'fimo --max-stored-scores 100000000 --text --thresh 1e-2  {TF_name}_meme.txt {TF_name}_masked_top_seq.fasta >{TF_name}_top.tsv')


TF_name = 'HNF4A'
os.system(f'fimo --max-stored-scores 100000000 --text --thresh 1e-2  {TF_name}_meme.txt {TF_name}_masked_bottom_seq.fasta >{TF_name}_bottom.tsv')
os.system(f'fimo --max-stored-scores 100000000 --text --thresh 1e-2  {TF_name}_meme.txt {TF_name}_masked_top_seq.fasta >{TF_name}_top.tsv')


all_reads_count = {}
def count_sequences_in_fasta(fasta_file_path):
    sequence_count = 0
    with open(fasta_file_path, 'r') as file:
        for line in file:
            if line.startswith('>'):
                sequence_count += 1
    return sequence_count

for TF_name in ['ELF1','HNF1A','HNF4A']:
    all_reads_count[f'{TF_name}_top']=count_sequences_in_fasta(f'{TF_name}_masked_top_seq.fasta')
    all_reads_count[f'{TF_name}_bottom']=count_sequences_in_fasta(f'{TF_name}_masked_bottom_seq.fasta')



# fig = plt.figure(figsize=(8, 5),constrained_layout=True)
fig = plt.figure(figsize=(9, 5))
for ax_num,TF_name in enumerate(['ELF1','HNF1A','HNF4A']):

    df_bottom_orig = pd.read_csv(f'{TF_name}_bottom.tsv', sep='\t')
    df_top_orig = pd.read_csv(f'{TF_name}_top.tsv', sep='\t')
    filtered_df_bottom = df_bottom_orig[(df_bottom_orig['p-value'] < 1e-3) & (df_bottom_orig['p-value'] > 1e-4)]
    filtered_df_top = df_top_orig[(df_top_orig['p-value'] < 1e-3) & (df_top_orig['p-value'] > 1e-4)]
    bottom_feature = filtered_df_bottom['sequence_name'].value_counts().tolist()
    top_feature = filtered_df_top['sequence_name'].value_counts().tolist()
    bottom_feature.extend([0]*(all_reads_count[f'{TF_name}_bottom']-len(bottom_feature)))
    top_feature.extend([0]*(all_reads_count[f'{TF_name}_top']-len(top_feature)))

    mean1 = np.mean(bottom_feature)
    mean2 = np.mean(top_feature)
    std1 = np.std(bottom_feature, ddof=1)
    std2 = np.std(top_feature, ddof=1)
    n1 = len(bottom_feature)
    n2 = len(top_feature)

    statistic_wil, p_value_wil = mannwhitneyu(bottom_feature, top_feature)
    z = (statistic_wil - (n1 * n2) / 2) / np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
    r = z / np.sqrt(n1 + n2)
    print(TF_name+'\t'+str(p_value_wil)+'\t'+str(statistic_wil)+'\t'+str(r))
    print(str(len(bottom_feature))+'\t'+str(len(top_feature)))

    df_bottom = pd.DataFrame({'Value': bottom_feature, 'Group': 'Bottom 10%'})
    df_top = pd.DataFrame({'Value': top_feature, 'Group': 'Top 10%'})

    df = pd.concat([df_bottom, df_top])
    plt.subplot(1, 3, ax_num+1)
    ax = sns.violinplot(x='Group', y='Value', data=df)
    plt.text(0.5, 1.08, f'p-value = {p_value_wil:.4e}', ha='center', va='center', transform=plt.gca().transAxes)
    bottom_mean = np.mean(bottom_feature)
    top_mean = np.mean(top_feature)
    plt.text(0.5, 1.13, TF_name, ha='center', va='center', transform=ax.transAxes)
    plt.text(0.22, 1.03, f'mean:{bottom_mean:.4f}', ha='center', va='center', transform=ax.transAxes, color='k')
    plt.text(0.78, 1.03, f'mean:{top_mean:.4f}', ha='center', va='center', transform=ax.transAxes, color='k')
    plt.text(0.25, -0.08, 'n='+str(len(bottom_feature)), ha='center', va='center', transform=ax.transAxes, color='k')
    plt.text(0.75, -0.08, 'n='+str(len(top_feature)), ha='center', va='center', transform=ax.transAxes, color='k')
    ax.set_xlabel('')
    ax.set_ylabel('Weak binding sites count')

plt.subplots_adjust(left=0.1, right=0.9, hspace=100)
plt.savefig('final_together_count_wilcox.png')
plt.close()




