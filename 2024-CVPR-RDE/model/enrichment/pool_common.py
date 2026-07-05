import torch
from torch.utils.data import Dataset


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _pool_transform(img_size):
    import torchvision.transforms as T

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    return T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


class _PoolImageDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from utils.iotools import read_image

        record = self.records[index]
        image = read_image(record["img_path"])
        if self.transform is not None:
            image = self.transform(image)
        return record["pid"], record["image_id"], image


class _PoolTextDataset(Dataset):
    def __init__(self, records, tokenizer, text_length, truncate):
        self.records = records
        self.tokenizer = tokenizer
        self.text_length = text_length
        self.truncate = truncate

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from datasets.bases import tokenize

        record = self.records[index]
        caption = tokenize(
            record["caption"],
            tokenizer=self.tokenizer,
            text_length=self.text_length,
            truncate=self.truncate,
        )
        return record["pid"], record["query_index"], caption
