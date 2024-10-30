import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
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



TF_ppm = {}
PPM_temp = np.array([[0.216024,0.153449,0.299007,0.331520],
                    [0.210477,0.310403,0.199854,0.279266],
                    [0.031567,0.891171,0.026600,0.050662],
                    [0.931554,0.027030,0.022815,0.018601],
                    [0.864807,0.029460,0.077735,0.027998],
                    [0.930329,0.011698,0.040792,0.017181],
                    [0.015397,0.019697,0.936006,0.028901],
                    [0.019934,0.014945,0.046964,0.918158],
                    [0.059565,0.812919,0.039265,0.088250],
                    [0.029309,0.910481,0.013827,0.046383],
                    [0.788190,0.059823,0.063134,0.088853],
                    [0.291889,0.212089,0.279847,0.216175],
                    [0.328552,0.195037,0.251892,0.224518]])
PPM_temp = np.concatenate((PPM_temp, np.ones((len(PPM_temp), 1)) * 0.25), axis=1)
TF_ppm['HNF4A'] = PPM_temp.copy()

PPM_temp = np.array([[0.368153,0.172521,0.246661,0.212665],
                    [0.459276,0.027303,0.471579,0.041842],
                    [0.022804,0.064289,0.054767,0.858139],
                    [0.147343,0.109723,0.027507,0.715427],
                    [0.981751,0.000208,0.017069,0.000971],
                    [0.933681,0.041771,0.000528,0.024020],
                    [0.080363,0.026683,0.000442,0.892512],
                    [0.236076,0.265055,0.320893,0.177976],
                    [0.934051,0.000594,0.022379,0.042976],
                    [0.045235,0.000827,0.053760,0.900178],
                    [0.002912,0.015808,0.000277,0.981003],
                    [0.814988,0.013651,0.077012,0.094349],
                    [0.878111,0.043133,0.067958,0.010799],
                    [0.086481,0.822329,0.038068,0.053121],
                    [0.280727,0.242208,0.160789,0.316277]])
PPM_temp = np.concatenate((PPM_temp, np.ones((len(PPM_temp), 1)) * 0.25), axis=1)
TF_ppm['HNF1A'] = PPM_temp.copy()

PPM_temp = np.array([[0.390999,0.180595,0.241574,0.186833],
                    [0.390371,0.163994,0.236655,0.208981],
                    [0.365083,0.265836,0.218547,0.150534],
                    [0.064769,0.793866,0.100879,0.040486],
                    [0.882646,0.084530,0.019510,0.013314],
                    [0.006699,0.008813,0.976387,0.008101],
                    [0.013586,0.010571,0.962319,0.013523],
                    [0.961482,0.016663,0.014151,0.007704],
                    [0.947101,0.014758,0.016307,0.021834],
                    [0.085053,0.032866,0.873728,0.008353],
                    [0.062989,0.041721,0.042014,0.853276],
                    [0.119071,0.088675,0.727591,0.064664],
                    [0.262947,0.247917,0.354197,0.134938],
                    [0.251622,0.265459,0.282018,0.200900]])
PPM_temp = np.concatenate((PPM_temp, np.ones((len(PPM_temp), 1)) * 0.25), axis=1)
TF_ppm['ELF1'] = PPM_temp.copy()


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


@jit(nopython=True)
def cal_all_value(seq,PPM_temp):
    res_dic = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    all_values = np.zeros(len(seq)+1-len(PPM_temp))
    for i in range(len(seq)+ 1 - len(PPM_temp)):
        value = 1
        for k in range(len(PPM_temp)):
            value *= PPM_temp[k][res_dic[seq[i + k]]]
        all_values[i] = value
    return all_values


def reverse_seq(read):
    read = list(read)
    read.reverse()
    lettercomplemt = {'A': 'T','T': 'A','G': 'C','C': 'G','N':'N'}
    read = [lettercomplemt[letter] for letter in read]
    read = "".join(read)
    return read


def get_motif_feature(seq, PPM_temp):
    check_seq = 'TCTGGCCTAACTGGCCGCTTGACG' + seq + 'CACTGCGGCTCCTGCGATAGAGGG' #sequence before and after enhancer on plasmid
    all_value_forw = cal_all_value(check_seq,PPM_temp)
    check_seq_rev = reverse_seq(check_seq)
    all_value_rev = cal_all_value(check_seq_rev,PPM_temp)
    all_value_rev = all_value_rev[::-1]
    return all_value_forw, all_value_rev


for TF_name in ['ELF1','HNF1A','HNF4A']:
    TF_ppm_temp = TF_ppm[TF_name]
    max_ppm = 1
    for item in TF_ppm_temp:
        max_ppm*=max(item)
    bottom_motif_features_forw = []
    bottom_motif_features_rev = []
    with open(TF_name + '_masked_bottom_seq.fasta') as f_in:
        for lines in f_in:
            if '>' in lines:
                lines = f_in.readline()
                seq = lines.split()[0]
                value_forw, value_rev = get_motif_feature(seq, TF_ppm[TF_name])
                value_forw=value_forw/max_ppm
                value_rev=value_rev/max_ppm
                bottom_motif_features_forw.append(value_forw)
                bottom_motif_features_rev.append(value_rev)
    
    bottom_motif_features_forw = np.array(bottom_motif_features_forw)
    bottom_motif_features_rev = np.array(bottom_motif_features_rev)
    np.save(TF_name+'_bottom_forw.npy',bottom_motif_features_forw)
    np.save(TF_name+'_bottom_rev_correforw.npy',bottom_motif_features_rev)

    top_motif_features_forw = []
    top_motif_features_rev = []
    with open(TF_name + '_masked_top_seq.fasta') as f_in:
        for lines in f_in:
            if '>' in lines:
                lines = f_in.readline()
                seq = lines.split()[0]
                value_forw, value_rev = get_motif_feature(seq, TF_ppm[TF_name])
                value_forw=value_forw/max_ppm
                value_rev=value_rev/max_ppm
                top_motif_features_forw.append(value_forw)
                top_motif_features_rev.append(value_rev)

    top_motif_features_forw = np.array(top_motif_features_forw)
    top_motif_features_rev = np.array(top_motif_features_rev)
    np.save(TF_name+'_top_forw.npy',top_motif_features_forw)
    np.save(TF_name+'_top_rev_correforw.npy',top_motif_features_rev)


@jit(nopython=True)
def cal_simulation_result(PPM_no_N):
    x = PPM_no_N.shape[0]
    simulation_result = np.zeros(4**x)
    for i in range(4**x):
        num = i
        result_array = np.zeros(x, dtype=np.int64)
        index = x - 1
        while i > 0 and index >= 0:
            result_array[index] = i % 4
            i //= 4
            index -= 1
        result = 1.0
        for j in range(PPM_no_N.shape[0]):
            result *= PPM_no_N[j, result_array[j]]   
        simulation_result[num]=result
    return simulation_result



for TF_name in ['ELF1','HNF1A','HNF4A']:

    TF_ppm_temp = TF_ppm[TF_name]
    max_ppm = 1
    for item in TF_ppm_temp:
        max_ppm*=max(item)
    PPM_no_N = TF_ppm_temp[:,0:4]
    simulation_result = cal_simulation_result(PPM_no_N)
    simulation_result = simulation_result/max_ppm
    total_num = 4**(PPM_no_N.shape[0])
    partition_values = np.zeros(7)
    for i,partition_temp in enumerate([0.05,0.01,0.001,0.0001,0.00001,0.000001,0.0000001]):
        k = int(total_num*partition_temp)
        partition_value_temp = np.partition(simulation_result, -k)[-k]
        partition_values[i] = partition_value_temp
    np.save('partition_values_' + TF_name + '.npy',partition_values)


fig = plt.figure(figsize=(7, 5),constrained_layout=True)
gs = gridspec.GridSpec(2, 3, height_ratios=[2, 10], hspace=0.02,wspace=0.2)

all_p_val = [0.05,0.01,0.001,0.0001,0.00001,0.000001,0.0000001]
for ax_num,TF_name in enumerate(['ELF1','HNF1A','HNF4A']):
    all_thresh = np.load('partition_values_' + TF_name + '.npy')
    thresh_num = 2
    thresh_temp=all_thresh[thresh_num]
    thresh_next=all_thresh[thresh_num+1]

    bottom_motif_features_forw = np.load(TF_name+'_bottom_forw.npy')
    bottom_motif_features_rev = np.load(TF_name+'_bottom_rev_correforw.npy')

    top_motif_features_forw = np.load(TF_name+'_top_forw.npy')
    top_motif_features_rev = np.load(TF_name+'_top_rev_correforw.npy')

    bottom_feature = np.maximum(bottom_motif_features_forw, bottom_motif_features_rev)
    top_feature = np.maximum(top_motif_features_forw, top_motif_features_rev)


    bottom_feature_0 = np.copy(bottom_feature)
    top_feature_0 = np.copy(top_feature)
    bottom_feature_0[bottom_feature_0<thresh_temp]=0
    top_feature_0[top_feature_0<thresh_temp]=0
    bottom_feature_0 = np.mean(bottom_feature_0,axis=1)
    top_feature_0 = np.mean(top_feature_0,axis=1)
    feature_temp = (bottom_feature>thresh_temp)&(bottom_feature<thresh_next)
    bottom_feature = np.sum(feature_temp,axis=1)
    feature_temp = (top_feature>thresh_temp)&(top_feature<thresh_next)
    top_feature = np.sum(feature_temp,axis=1)
    bottom_feature = np.squeeze(bottom_feature)
    top_feature = np.squeeze(top_feature)

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
    bottom_mean = bottom_feature.mean()
    top_mean = top_feature.mean()
    plt.text(0.5, 1.13, TF_name, ha='center', va='center', transform=ax.transAxes)
    plt.text(0.22, 1.03, f'mean:{bottom_mean:.4f}', ha='center', va='center', transform=ax.transAxes, color='k')
    plt.text(0.78, 1.03, f'mean:{top_mean:.4f}', ha='center', va='center', transform=ax.transAxes, color='k')
    plt.text(0.25, -0.08, 'n='+str(len(bottom_feature)), ha='center', va='center', transform=ax.transAxes, color='k')
    plt.text(0.75, -0.08, 'n='+str(len(top_feature)), ha='center', va='center', transform=ax.transAxes, color='k')
    ax.set_xlabel('')
    ax.set_ylabel('Weak binding sites count')


plt.savefig('final_together_count_wilcox_'+str(all_p_val[thresh_num])+'.png')
plt.close()