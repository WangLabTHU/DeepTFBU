from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import umap
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader


TF_list = ['ELF1','HNF1A','HNF4A']


### The commented-out code is used to generate DNAshape features, which is sourced from https://rohslab.cmb.usc.edu/DNAshape/, and we do not have the rights to disclose it publicly.
### We have provided the DNA shape features required for analysis; the un-commented sections of the code can be executed directly
# top_ratio = 0.1
# seq_mot_dic = {'ELF1':'CAGGAAGTG', 'HNF1A':'GTTAATGATTAAC', 'HNF4A':'CAAAGTCCA'} #consensus sequence of used motif
# for TF_temp in TF_list:
#     seq_mot = seq_mot_dic[TF_temp]
#     temp_seqs = []
#     temp_values = []
#     temp_masked = []
#     with open('../identify_obvious_TFBS/Lib118_without_motif.tsv','r') as f:
#         for lines in f:
#             line = lines.split()
#             if ('aim' in line[2])or('pos' in line[2])or('neg' in line[2]):
#                 seq = line[0]
#                 value = float(line[1])
#                 if len(seq)==168:
#                     if seq[int(168/2)-int(len(seq_mot)/2):int(168/2)-int(len(seq_mot)/2)+int(len(seq_mot))]==seq_mot:
#                         temp_seqs.append(seq)
#                         temp_values.append(value)
#     sorted_lst = sorted(temp_values)
#     seq_num = len(sorted_lst)

#     sorted_lst = sorted(temp_values)
#     seq_num = len(sorted_lst)

#     with open(f'./{TF_temp}_top.txt.s','w') as f_out:
#         for i,(value, seq) in enumerate(zip(temp_values,temp_seqs)):
#             if value>sorted_lst[-1-int(seq_num*top_ratio)]:
#                 f_out.write(str(np.log10(value))+'\t' + seq + '\n')

#     with open(f'./{TF_temp}_bottom.txt.s','w') as f_out:
#         for i,(value, seq) in enumerate(zip(temp_values,temp_seqs)):
#             if value<sorted_lst[int(seq_num*top_ratio)]:
#                 f_out.write(str(np.log10(value))+'\t' + seq + '\n')


file_types = ['bottom','top']
for TF_name in TF_list:
    for type_temp in file_types:
        # os.system('./Feature_encoding.pl ' + TF_name + '_' + type_temp + '.txt.s')

        root_dir = './DNAshape_features/'
        seq_len = 168
        libsvm_input = root_dir + TF_name + '_' + type_temp + '_4shape.libsvm'
        label_input = root_dir + TF_name + '_' + type_temp + '_4shape.label'
        dict_file = root_dir + TF_name + '_' + type_temp + '_4shape.npy'

        with open(label_input,'r') as f:
            label = []
            for item in f:
                label.append(item)
                
        with open(libsvm_input,'r') as f:
            record = []
            for item in f:
                record.append(item.split())
        
        MGW_array = np.zeros((len(record),seq_len,1))
        Roll_array = np.zeros((len(record),seq_len,1))
        twist_array = np.zeros((len(record),seq_len,1))
        ProT_array = np.zeros((len(record),seq_len,1))    

        for i,item in enumerate(record):
            for j,subitem in enumerate(item):
                if j > 0 and j < (162+3):
                    MGW_array[i,j+1] = subitem.split(":")[1]
                elif j >= (162+3) and j <= (323+6):
                    Roll_array[i,(j+1) - (162+3)] = subitem.split(":")[1]
                elif j >= (324+6) and j <= (485+9):
                    twist_array[i,(j+1) - (324+6)] = subitem.split(":")[1]
                elif j >= (486+9) and j <= (646+12):
                    ProT_array[i,(j+1) - (485+9)] = subitem.split(":")[1]
                
        info_array = np.concatenate((MGW_array,Roll_array,twist_array,ProT_array),axis=2)
        np.save(TF_name + '_' + type_temp + '_(MGW,Roll,twist,ProT).npy',info_array)


class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError
    def __add__(self, other):
        return ConcatDataset([self, other])


class LoadData(Dataset):
    def __init__(self, seq_shapes):
        self.seq_shapes = seq_shapes
    def __getitem__(self, item):
        seq_shape_i1 = self.seq_shapes[item, :, :].copy()
        seq_len = np.size(self.seq_shapes, 2)
        pos1 = np.random.randint(0, seq_len)
        pos2 = np.random.randint(0, seq_len)
        seq_shape_i1[:, 0: seq_len - pos1] = self.seq_shapes[item, :, pos1: seq_len]
        seq_shape_i1[:, seq_len - pos1: seq_len] = self.seq_shapes[item, :, 0: pos1]
        seq_shape_i1 = torch.from_numpy(seq_shape_i1).float().to('cuda')
        seq_shape_i2 = self.seq_shapes[item, :, :].copy()
        seq_shape_i2[:, 0: seq_len - pos2] = self.seq_shapes[item, :, pos2: seq_len]
        seq_shape_i2[:, seq_len - pos2: seq_len] = self.seq_shapes[item, :, 0: pos2]
        seq_shape_i2 = torch.from_numpy(seq_shape_i2).float().to('cuda')
        return torch.cat((seq_shape_i1, seq_shape_i2), dim=0)
    def __len__(self):
        return np.size(self.seq_shapes, 0)


class encoder(nn.Module):
    def __init__(self, seq_len=165):
        super(encoder, self).__init__()
        seed=42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        self.conv1 = nn.Conv1d(4, 512, kernel_size=7)
        self.conv2 = nn.Conv1d(4, 512, kernel_size=13)
        self.conv3 = nn.Conv1d(4, 512, kernel_size=19)
        self.conv4 = nn.Conv1d(4, 512, kernel_size=25)
        self.conv5 = nn.Conv1d(4, 512, kernel_size=31)
        self.mp1 = nn.MaxPool1d(kernel_size=seq_len - 7 + 1)
        self.mp2 = nn.MaxPool1d(kernel_size=seq_len - 13 + 1)
        self.mp3 = nn.MaxPool1d(kernel_size=seq_len - 19 + 1)
        self.mp4 = nn.MaxPool1d(kernel_size=seq_len - 25 + 1)
        self.mp5 = nn.MaxPool1d(kernel_size=seq_len - 31 + 1)
        self.project = nn.Linear(512*4, 64)
    def forward(self, x):
        x1 = self.mp1(self.conv1(x)).squeeze()
        x2 = self.mp2(self.conv2(x)).squeeze()
        x3 = self.mp3(self.conv3(x)).squeeze()
        x4 = self.mp4(self.conv4(x)).squeeze()
        x_c = torch.cat((x1, x2, x3, x4), dim=1)
        x_p = self.project(x_c)
        return x_p


def compute_loss(y_pred, lamda=0.05):
    idxs = torch.arange(0, y_pred.shape[0], device='cuda')
    y_true = idxs + 1 - idxs % 2 * 2
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    similarities = similarities - torch.eye(y_pred.shape[0], device='cuda') * 1e12
    similarities = similarities / lamda
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


def train(seq_shapes, model, optimizer, save_model):
    model.train()
    D = LoadData(seq_shapes)
    dataloader = DataLoader(D, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(42))
    size = len(dataloader.dataset)
    for batch, data in tqdm(enumerate(dataloader)):
        d = data.reshape([-1, 4, 168])
        pred = model(d)
        loss = compute_loss(pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss, current = loss.item(), batch * int(len(data) / 2)
    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    torch.save(model, save_model)
    print("Saved PyTorch Model State to {}".format(save_model))



def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_data_constitutive(high_emb, low_emb, picture_name):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.kdeplot(high_emb[:, 0], high_emb[:, 1], shade=False,
                colors='#ee7b6c',
                thresh=0.4,
                cmap=None, linewidths=1, alpha=0.7)
    sns.kdeplot(low_emb[:, 0], low_emb[:, 1], shade=False,
                colors='#8eb8d9',
                thresh=0.4,
                cmap=None, linewidths=1, alpha=0.7)
    plt.scatter(high_emb[:, 0], high_emb[:, 1], c='#ee7b6c', alpha=0.2, s=6,
                    linewidth=0, label='High Activity')
    plt.scatter(low_emb[:, 0], low_emb[:, 1], c='#8eb8d9', alpha=0.5, s=6,
                    linewidth=0, label='Low Activity')
    plt.legend(loc='upper right')
    plt.tick_params(
        axis='x',
        which='both',
        bottom=False, 
        top=False,
        labelbottom=False)
    plt.tick_params(
        axis='y',
        which='both',
        left=False,
        right=False,
        labelleft=False)
    plt.xlabel('Umap 1')
    plt.ylabel('Umap 2')
    plt.title(picture_name.split('_')[-1].split('.')[0]+' DNAshape feature')
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    plt.savefig(picture_name+'.png')
    

for TF_name in TF_list:
    setup_seed(42)
    seq_len = 168
    train_encoder = True
    save_model = './encoder_' + TF_name + '_constitutive.pth'
    lib_path_bottom = './' + TF_name + '_bottom_(MGW,Roll,twist,ProT).npy'
    lib_array_bottom = np.load(lib_path_bottom, allow_pickle=True).squeeze()
    lib_path_top = './' + TF_name + '_top_(MGW,Roll,twist,ProT).npy'
    lib_array_top = np.load(lib_path_top, allow_pickle=True).squeeze()
    lib_array = np.vstack((lib_array_bottom,lib_array_top))
    seq_shapes = np.transpose(lib_array, (0,2,1))
    

    # train encoder using contrastive learning
    enc_model = encoder(seq_len=seq_len).cuda()
    if train_encoder == True:
        optimizer = torch.optim.AdamW(enc_model.parameters(), lr=1e-5)
        for i in range(100):
            train(seq_shapes, enc_model, optimizer, save_model)

    enc_model = torch.load(save_model).cuda()
    shape_enc = torch.from_numpy(seq_shapes).float().cuda()
    with torch.no_grad():
        shape_enc = enc_model(shape_enc).cpu().numpy()


    umap_model = umap.UMAP(n_components=2,random_state=42)
    umap_input = np.vstack((shape_enc))
    umap_result = umap_model.fit_transform(umap_input)
    n_cut = len(lib_array_bottom)
    low_emb = umap_result[0: n_cut, :]
    high_emb = umap_result[n_cut::, :]

    # np.savetxt('0_'+ TF_name + '_lowemb.txt',low_emb)
    # np.savetxt('0_'+ TF_name + '_highemb.txt',high_emb)

    picture_name = './DNA_shape_'+TF_name+''
    plot_data_constitutive(high_emb, low_emb, picture_name)









