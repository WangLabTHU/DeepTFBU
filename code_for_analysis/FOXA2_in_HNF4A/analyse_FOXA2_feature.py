import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def cal_max_pos(seq,PPM_temp):
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


seq_mot = 'CAAAGTCCA' #HNF4A core TFBS
with open('../Supplementary_Table1_MPRA_result_3TF.tsv','r') as f,open('./all_HNF4A_seq.tsv','w') as f_out:
    for lines in f:
        line = lines.split()
        if ('aim' in line[2])or('pos' in line[2])or('neg' in line[2]):
            seq = line[0]
            value = float(line[1])
            if len(seq)==168:
                if seq[int(168/2)-int(len(seq_mot)/2):int(168/2)-int(len(seq_mot)/2)+int(len(seq_mot))]==seq_mot:
                    f_out.write(lines)


TF_PPM_temp = [[0.35757745,0.04742269,0.14700576,0.44799411], # PPM of FOXA2
                [0.07159941,0.01116905,0.90142398,0.01580756],
                [0.01425503,0.00777027,0.00720953,0.97076516],
                [0.93001034,0.03594847,0.01007427,0.02396692],
                [0.85088097,0.08555691,0.02473365,0.03882847],
                [0.94682876,0.01269869,0.01069986,0.02977269],
                [0.00919692,0.75135131,0.01374388,0.22570789],
                [0.94092,0.01177175,0.01138267,0.03592559]]

TF_PPM_temp = np.concatenate((TF_PPM_temp, np.ones((len(TF_PPM_temp), 1)) * 0.25), axis=1)
max_ppm = 1
for item in TF_PPM_temp:
    max_ppm*=max(item)


info_dic = {}
with open('all_HNF4A_seq.tsv') as f_in:
    for lines in f_in:
        line = lines.split()
        seq = line[0]
        check_seq = 'NNNN' + seq + 'NNN'
        info_seq = cal_max_pos(check_seq,TF_PPM_temp)
        check_seq_rev = reverse_seq(check_seq)
        info_seq_rev = cal_max_pos(check_seq_rev,TF_PPM_temp)
        info_dic[seq] = [info_seq/max_ppm,info_seq_rev/max_ppm]


###for expression values
all_values = []
with open('all_HNF4A_seq.tsv') as f_in:
    for lines in f_in:
        line = lines.split()
        all_values.append(float(line[1]))

sorted_lst = sorted(all_values, reverse=True)
top10percent_index = int(len(sorted_lst) * 0.1)
top_value = sorted_lst[top10percent_index - 1]
bottom10percent_index = int(len(sorted_lst) * 0.9)
bottom_value = sorted_lst[bottom10percent_index - 1]

print(top_value)
print(bottom_value)

plt.figure(figsize=(10, 6))

top_values = []
bottom_values = []
with open('all_HNF4A_seq.tsv') as f_in:
    for lines in f_in:
        line = lines.split()
        seq = line[0]
        x = np.linspace(1,168,168)
        x_rev = np.linspace(168,1,168)
        if float(line[1])<bottom_value:
            plt.subplot(3, 1, 1)
            plt.plot(x, info_dic[seq][0], color='#8eb8d9',alpha=0.5)
            # plt.subplot(2, 1, 2)
            plt.plot(x_rev, -info_dic[seq][1], color='#8eb8d9',alpha=0.5)

            bottom_values.append(max(np.hstack((info_dic[seq][0],info_dic[seq][1]))))
        
with open('all_HNF4A_seq.tsv') as f_in:
    for lines in f_in:
        line = lines.split()
        seq = line[0]
        x = np.linspace(1,168,168)
        x_rev = np.linspace(168,1,168)
        if float(line[1])>top_value:
            plt.subplot(3, 1, 1)
            plt.plot(x, info_dic[seq][0], color='#ee7b6c',alpha=0.5)
            # plt.subplot(2, 1, 2)
            plt.plot(x_rev, -info_dic[seq][1], color='#ee7b6c',alpha=0.5)

            top_values.append(max(np.hstack((info_dic[seq][0],info_dic[seq][1]))))

plt.subplot(3, 1, 1)
plt.ylim([-1,1])
TtestResult = stats.ttest_ind(top_values, bottom_values, alternative="greater")
t_statistic = TtestResult.statistic
p_value_exp = TtestResult.pvalue

mean1 = np.mean(top_values)
mean2 = np.mean(bottom_values)
std1 = np.std(top_values, ddof=1)
std2 = np.std(bottom_values, ddof=1)
n1 = len(top_values)
n2 = len(bottom_values)
pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
effect_size = (mean1 - mean2) / pooled_std

print('for exp:'+str(p_value_exp)+'\t'+str(t_statistic)+'\t'+str(effect_size))


###for predicted values
all_values = []
with open('all_HNF4A_seq.tsv') as f_in:
    for lines in f_in:
        line = lines.split()
        pred = line[2].split('_')[0]
        all_values.append(float(pred))

sorted_lst = sorted(all_values, reverse=True)
top10percent_index = int(len(sorted_lst) * 0.1)
top_value = sorted_lst[top10percent_index - 1]
bottom10percent_index = int(len(sorted_lst) * 0.9)
bottom_value = sorted_lst[bottom10percent_index - 1]
print(top_value)
print(bottom_value)

top_values = []
bottom_values = []
with open('all_HNF4A_seq.tsv') as f_in:
    for lines in f_in:
        line = lines.split()
        seq = line[0]
        x = np.linspace(1,168,168)
        x_rev = np.linspace(168,1,168)
        if float(line[2].split('_')[0])<bottom_value:
            plt.subplot(3, 1, 2)
            plt.plot(x, info_dic[seq][0], color='#8eb8d9',alpha=0.5)
            plt.plot(x_rev, -info_dic[seq][1], color='#8eb8d9',alpha=0.5)
            bottom_values.append(max(np.hstack((info_dic[seq][0],info_dic[seq][1]))))

with open('all_HNF4A_seq.tsv') as f_in:
    for lines in f_in:
        line = lines.split()
        seq = line[0]
        x = np.linspace(1,168,168)
        x_rev = np.linspace(168,1,168)
        if float(line[2].split('_')[0])>top_value:
            plt.subplot(3, 1, 2)
            plt.plot(x, info_dic[seq][0], color='#ee7b6c',alpha=0.5)
            plt.plot(x_rev, -info_dic[seq][1], color='#ee7b6c',alpha=0.5)
            top_values.append(max(np.hstack((info_dic[seq][0],info_dic[seq][1]))))

         
plt.subplot(3, 1, 2)
plt.ylim([-1,1])
TtestResult = stats.ttest_ind(top_values, bottom_values, alternative="greater")
t_statistic = TtestResult.statistic
p_value_pred = TtestResult.pvalue

mean1 = np.mean(top_values)
mean2 = np.mean(bottom_values)
std1 = np.std(top_values, ddof=1)
std2 = np.std(bottom_values, ddof=1)
n1 = len(top_values)
n2 = len(bottom_values)
pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
effect_size = (mean1 - mean2) / pooled_std
print('for predicted:'+str(p_value_pred)+'\t'+str(t_statistic)+'\t'+str(effect_size))


all_values = []
with open('all_HNF4A_seq.tsv') as f_in:
    for lines in f_in:
        line = lines.split()
        if '_aim_1_' in line[2]:
            continue
        pred = line[2].split('_')[0]
        all_values.append(float(pred))
sorted_lst = sorted(all_values, reverse=True)
top10percent_index = int(len(sorted_lst) * 0.1)
top_value = sorted_lst[top10percent_index - 1]
bottom10percent_index = int(len(sorted_lst) * 0.9)
bottom_value = sorted_lst[bottom10percent_index - 1]

print(top_value)
print(bottom_value)

top_values = []
bottom_values = []
with open('all_HNF4A_seq.tsv') as f_in:
    for lines in f_in:
        line = lines.split()
        if '_aim_1_' in line[2]:
            continue
        seq = line[0]
        x = np.linspace(1,168,168)
        x_rev = np.linspace(168,1,168)
        if float(line[2].split('_')[0])<bottom_value:
            plt.subplot(3, 1, 3)
            plt.plot(x, info_dic[seq][0], color='#8eb8d9',alpha=0.5)
            plt.plot(x_rev, -info_dic[seq][1], color='#8eb8d9',alpha=0.5)
            bottom_values.append(max(np.hstack((info_dic[seq][0],info_dic[seq][1]))))

with open('all_HNF4A_seq.tsv') as f_in:
    for lines in f_in:
        line = lines.split()
        if '_aim_1_' in line[2]:
            continue
        seq = line[0]
        x = np.linspace(1,168,168)
        x_rev = np.linspace(168,1,168)
        if float(line[2].split('_')[0])>top_value:
            plt.subplot(3, 1, 3)
            plt.plot(x, info_dic[seq][0], color='#ee7b6c',alpha=0.5)
            plt.plot(x_rev, -info_dic[seq][1], color='#ee7b6c',alpha=0.5)
            top_values.append(max(np.hstack((info_dic[seq][0],info_dic[seq][1]))))
          
plt.subplot(3, 1, 3)
plt.ylim([-1,1])
TtestResult = stats.ttest_ind(top_values, bottom_values, alternative="greater")
t_statistic = TtestResult.statistic
p_value_pred_nocollaps = TtestResult.pvalue
mean1 = np.mean(top_values)
mean2 = np.mean(bottom_values)
std1 = np.std(top_values, ddof=1)
std2 = np.std(bottom_values, ddof=1)
n1 = len(top_values)
n2 = len(bottom_values)
pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
effect_size = (mean1 - mean2) / pooled_std
print('for pred nocollaps:'+str(p_value_pred_nocollaps)+'\t'+str(t_statistic)+'\t'+str(effect_size))
print(p_value_exp)
print(p_value_pred)
print(p_value_pred_nocollaps)


plt.savefig('FOXA2_exp_pred_noreppred.png')
plt.close()




