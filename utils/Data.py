import torch
from torch.utils.data import Dataset


class SentencePairDataset(Dataset):
    def __init__(self, data, max_len, padding_idx):
        super(SentencePairDataset, self).__init__()

        self.data_size = len(data["input_ids"])
        self.input_ids = torch.ones((self.data_size, max_len), dtype=torch.int64) * padding_idx
        self.token_type_ids = torch.ones((self.data_size, max_len), dtype=torch.int64) * padding_idx
        self.mask_ids = torch.ones((self.data_size, max_len), dtype=torch.int64) * padding_idx
        self.labels = torch.ones((self.data_size, 1), dtype=torch.int64) * padding_idx
        text_length = list()

        for idx in range(self.data_size):
            content_len = min(len(data["input_ids"][idx]), max_len)
            self.input_ids[idx][:content_len] = torch.tensor(data["input_ids"][idx][:content_len], dtype=torch.int64)
            self.token_type_ids[idx][:content_len] = torch.tensor(data["token_type_ids"][idx][:content_len], dtype=torch.int64)
            self.mask_ids[idx][:content_len] = torch.tensor(data["mask_ids"][idx][:content_len], dtype=torch.int64)
            text_length.append(content_len)
            self.labels[idx] = torch.tensor(data["labels"][idx], dtype=torch.int64)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        ret_data = {
            "input_ids": self.input_ids[idx],
            "token_type_ids": self.token_type_ids[idx],
            "mask_ids": self.mask_ids[idx],
            "labels": self.labels[idx]
        }
        return ret_data


