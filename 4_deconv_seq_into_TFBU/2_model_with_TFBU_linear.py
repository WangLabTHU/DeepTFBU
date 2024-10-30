import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd

def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


if __name__ == '__main__':
    os.makedirs('./step2_model_with_TFBU_linear', exist_ok=True)
    max_env_values = load_variavle('./data_dir/step1_max_env_values_final.pkl')
    max_value_pos = load_variavle('./data_dir/step1_max_value_pos_final.pkl')
    max_core_values = load_variavle('./data_dir/step1_max_core_values_final.pkl')
    max_core_normed = load_variavle('./data_dir/step1_max_core_normed_final.pkl')

    with open('./data_dir/MPRA_exp_mean.tsv') as f_in:
        all_exp = []
        for lines in f_in:
            line = lines.strip().split('\t')
            seq = line[0]
            exp = float(line[1])
            all_exp.append(exp)

    all_exp = np.array(all_exp)
    all_exp = np.array(np.log10(all_exp))
    np.random.seed(42)
    np.random.shuffle(max_env_values)
    np.random.seed(42)
    np.random.shuffle(max_value_pos)
    np.random.seed(42)
    np.random.shuffle(max_core_values)
    np.random.seed(42)
    np.random.shuffle(max_core_normed)
    np.random.seed(42)
    np.random.shuffle(all_exp)

    lr=0.001
    for mode in ['env_conc_core_conc_envcore','env_conc_core','env_core','core_norm','env_score']:
        for tr_num in range(10):
            if mode == 'env_score':
                motif_score = np.array(max_env_values)
                input_size = 198
            elif mode == 'core_norm':
                motif_score = np.array(max_core_normed)
                input_size = 198
            elif mode == 'env_core':
                motif_score = np.array(max_core_normed)*np.array(max_env_values)
                input_size = 198
            elif mode == 'env_conc_core':
                motif_score = np.concatenate((np.array(max_env_values),np.array(max_core_normed)),axis=1)
                input_size = 198*2
            elif mode == 'env_conc_core_conc_envcore':
                motif_score = np.concatenate((np.array(max_env_values),np.array(max_core_normed)),axis=1)
                motif_score = np.concatenate((motif_score,np.array(max_core_normed)*np.array(max_env_values)),axis=1)
                input_size = 198*3

            motif_score_tensor = torch.tensor(motif_score, dtype=torch.float32)
            exp_tensor = torch.tensor(all_exp, dtype=torch.float32)
            dataset_size = len(exp_tensor)
            train_motif = motif_score_tensor[0:int(dataset_size*0.8)]
            train_exp = exp_tensor[0:int(dataset_size*0.8)]
            val_motif = motif_score_tensor[int(dataset_size*0.8):int(dataset_size*0.9)]
            val_exp = exp_tensor[int(dataset_size*0.8):int(dataset_size*0.9)]
            test_motif = motif_score_tensor[int(dataset_size*0.9):dataset_size]
            test_exp = exp_tensor[int(dataset_size*0.9):dataset_size]

            scaler = StandardScaler()
            train_exp = scaler.fit_transform(train_exp.cpu().reshape(-1, 1))
            train_exp = torch.tensor(train_exp, dtype=torch.float32)

            batch_size = 128
            train_dataset = TensorDataset(train_motif, train_exp)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            val_dataset = TensorDataset(val_motif, val_exp)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            test_dataset = TensorDataset(test_motif, test_exp)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


            class LinearRegressionModel(nn.Module):
                def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
                    super(LinearRegressionModel, self).__init__()
                    self.linear = nn.Linear(input_size, output_size)

                def forward(self, x):
                    x = self.linear(x)
                    return x

            hidden_size1 = 256
            hidden_size2 = 128
            output_size = 1
            model = LinearRegressionModel(input_size, hidden_size1, hidden_size2, output_size).cuda()
            criterion = nn.MSELoss().cuda()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            best_val_correlation = -1.0
            best_model_state = None

            num_epochs = 1000
            for epoch in range(num_epochs):
                model.train()
                for batch_motif, batch_exp in train_loader:
                    batch_motif = batch_motif.cuda()
                    batch_exp = batch_exp.cuda()

                    optimizer.zero_grad()
                    outputs = model(batch_motif)
                    loss = criterion(outputs, batch_exp)
                    loss.backward()
                    optimizer.step()

                model.eval()
                label_list = []
                pred_list = []
                with torch.no_grad():
                    for batch_motif, batch_exp in val_loader:
                        batch_motif = batch_motif.cuda()
                        outputs = model(batch_motif)

                        pred_result = outputs.cpu().numpy()
                        pred_list.extend(list(pred_result))

                        label_list.extend(list(batch_exp.numpy()))

                val_predicted_exp = np.array(pred_list)
                val_predicted_exp = scaler.inverse_transform(val_predicted_exp.reshape(-1, 1)).flatten()


                val_correlation, _ = pearsonr(val_predicted_exp.flatten(), label_list)
                if val_correlation > best_val_correlation:
                    best_val_correlation = val_correlation
                    best_model_state = model.state_dict()

                print(f'Epoch [{epoch+1}/{num_epochs}], Validation Correlation: {val_correlation:.4f}')

            torch.save(best_model_state, './step2_model_with_TFBU_linear/model_'+mode + '_' +str(lr)+ '_'+ str(tr_num) +'.pth')
            best_model = LinearRegressionModel(input_size, hidden_size1, hidden_size2, output_size).cuda()
            best_model.load_state_dict(best_model_state)
            best_model.eval()
            label_list = []
            pred_list = []
            with torch.no_grad():
                for batch_motif, batch_exp in test_loader:
                    batch_motif = batch_motif.cuda()
                    outputs = best_model(batch_motif)

                    pred_result = outputs.cpu().numpy()
                    pred_list.extend(list(pred_result))

                    label_list.extend(list(batch_exp.numpy()))

            predicted_exp = np.array(pred_list)
            predicted_exp = scaler.inverse_transform(predicted_exp.reshape(-1, 1)).flatten()

            pearson = pearsonr(predicted_exp.flatten(), label_list)[0]

            with open('./step2_model_with_TFBU_linear/repeat_log_linear.txt','a') as log_out:
                log_out.write(str(lr) + '\t' + str(mode) + '\t' + str(tr_num) + '\t' + str(pearson) + '\n')
            plt.scatter(predicted_exp, test_exp,alpha=0.5, s=2)
            plt.xlabel('pred_result')
            plt.ylabel('exp')
            plt.title('r = %.4f' % pearson)
            # plt.show()
            plt.savefig('./step2_model_with_TFBU_linear/result_'+mode + '_' +str(lr)+ '_'+ str(tr_num) +'.png')
            plt.close()


            del model, best_model
            torch.cuda.empty_cache()


    ## plot result
    data = pd.read_csv('./step2_model_with_TFBU_linear/repeat_log_linear.txt', header=None, delimiter='\t', names=['Learning Rate', 'Method', 'Experiment', 'Result'])
    learning_rates = data['Learning Rate'].unique()

    lr=0.001
    filtered_data = data[data['Learning Rate'] == lr]
    fig, ax = plt.subplots(figsize=(8, 5))
    method_order = filtered_data['Method'].unique()

    boxplot_data = []
    means = []
    stds = []
    labels = []

    for method in method_order:
        group = filtered_data[filtered_data['Method'] == method]
        means.append(group['Result'].mean())
        stds.append(group['Result'].std())
        boxplot_data.append(group['Result'].tolist())
        if method == 'env_conc_core_conc_envcore':
            method_log = 'all three feature'
        if method == 'env_conc_core':
            method_log = 'TFBS-context & core TFBS'            
        if method == 'env_core':
            method_log = 'TFBU'  
        if method == 'core_norm':
            method_log = 'core TFBS'  
        if method == 'env_score':
            method_log = 'TFBS-context'  
        labels.append(method_log)



    x = range(len(labels))
    ax.bar(x, means, yerr=stds, capsize=5, width=0.3,color='#ED6F6D', align='center', alpha=0.7,edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim([0.6,0.85])
    ax.set_xlabel('Feature')
    ax.set_ylabel('Person correlation')
    ax.set_title('Model effect with different feature')
    plt.tight_layout()
    plt.savefig('./step2_model_with_TFBU_linear/final_result.png')
    plt.close()













