import numpy as np
import pickle


def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


if __name__ == '__main__':
    TF_list = ['GATA2']
    all_chr = np.array(load_variavle('./data_dir/step0_K562_chrs.pkl'))
    all_start = np.array(load_variavle('./data_dir/step0_K562_start.pkl'))
    all_end = np.array(load_variavle('./data_dir/step0_K562_end.pkl'))
    for TF_name in TF_list:
        unselectable = np.zeros(len(all_chr))
        all_values = load_variavle('./data_dir/step0_K562_max_for_'+TF_name+'.pkl')
        chip_data = open('./data_dir/K562_ChIP_'+TF_name+'.bed')
        chip_region = 0
        for lines in chip_data:
            line = lines.split()
            chr_temp = line[0]
            start_temp = int(line[1])
            end_temp = int(line[2])
            chip_region+=1
            unselectable[(all_chr==chr_temp)&((start_temp<=all_end)&(end_temp>=all_start))]=1
        new_value = all_values-unselectable
        selected_index = np.argsort(-new_value)
        out_file = open('./data_dir/step1_K562_negative_'+TF_name+'.bed','w')
        for index in selected_index:
            if new_value[index]>0:
                out_file.write(all_chr[index] + '\t' + str(all_start[index]) + '\t' + str(all_end[index]) + '\t' + str(new_value[index])+'\n')
        out_file.close()






