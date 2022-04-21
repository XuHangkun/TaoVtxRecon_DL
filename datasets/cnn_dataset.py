import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import pandas as pd

class CNNDataset(Dataset):

    def __init__(self, h5_file, h5_info, information = "all"):
        self.h5_file = h5py.File(h5_file, "r")
        csv_info = pd.read_csv(h5_info, header=None)
        self.groupname = np.asarray(csv_info.iloc[:,0])
        self.datainfo = np.asarray(csv_info.iloc[:,1])
        self.length  = len(csv_info)
        self.information = information

    def set_length(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        dset = self.h5_file[self.groupname[idx]][str(self.datainfo[idx])]
        vertex = dset.attrs["vertex"]
        edep = dset.attrs["edep"]
        sipm_info = np.array(dset)
        if self.information == "charge":
            sipm_info = sipm_info[0].reshape([1,*sipm_info.shape[1:]])
        elif self.information == "time":
            sipm_info = sipm_info[1].reshape([1,*sipm_info.shape[1:]])
        return torch.from_numpy(sipm_info).to(torch.float32), torch.from_numpy(vertex).to(torch.float32), torch.tensor(edep).to(torch.float32)

def test():
    dataset = CNNDataset("../dataset/valid_dataset.h5", "../dataset/valid_dataset.csv", information = "charge")
    print(len(dataset))
    imgs, vertex, _ = dataset[1]
    print(f"SiPM info shape : {imgs.shape};\nVertex : {vertex}")

if __name__ == "__main__":
    test()