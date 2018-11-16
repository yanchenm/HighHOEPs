import torch.utils.data as data


class LinearDataset(data.Dataset):
    def __init__(self, X, y):
        self.data = X
        self.labels = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index],self.labels[index]
