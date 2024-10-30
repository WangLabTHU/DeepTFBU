from torch.utils.data import Dataset
import pickle

def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


class ProbDataset(Dataset):
    def __init__(self, seq, label):
        self.seq = seq
        self.label = label        
        self.lenth = len(self.label)

    def __getitem__(self, idx):
        x = self.seq[idx]
        y = self.label[idx]
        return x,y


    def __len__(self):
        return self.lenth