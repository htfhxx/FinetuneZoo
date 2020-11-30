import torch
from torch.utils.data import Dataset


class LcqmcDataset(Dataset):
    def __init__(self, data, max_len, padding_idx):
        super(LcqmcDataset, self).__init__()
        self.data_size = len(data["text"])

        max_data_len = 1
        for line in data["text"]:
            if len(line)>max_data_len:
                max_data_len = len(line)
        max_len = min(max_len, max_data_len+1)
        print('real max_len: %d'%(max_len))

        self.text = torch.ones((self.data_size, max_len), dtype=torch.int64) * padding_idx
        self.label = torch.ones((self.data_size, 1), dtype=torch.int64) * padding_idx
        self.mask = torch.ones((self.data_size, max_len), dtype=torch.int64) * padding_idx


        text_length = list()

        for idx in range(self.data_size):
            content_len = min(len(data["text"][idx]), max_len)
            self.text[idx][:content_len] = torch.tensor(data["text"][idx][:content_len], dtype=torch.int64)
            self.mask[idx][:content_len] = 1
            text_length.append(content_len)
            self.label[idx] = torch.tensor(data["label"][idx], dtype=torch.int64)

        self.text_length = torch.tensor(text_length)
        self.mask = torch.tensor(self.mask)


    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        ret_data = {
            "text": self.text[idx],
            "mask": self.mask[idx],
            "text_length": self.text_length[idx],
            "label": self.label[idx]
        }
        return ret_data


