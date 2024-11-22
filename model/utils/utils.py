import logging
import pickle
import random
import torch
import torch.backends
import torch.utils
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.utils.data


def load_dataloader(file_path):
    """
    从 .npz 文件加载 DataLoader。
    :param file_path: 保存文件路径
    :return: 还原的 DataLoader 对象
    """
    # 加载文件数据
    data = np.load(file_path)

    # 还原 Dataset
    keys = list(data.keys())
    tensors = [torch.tensor(data[key]) for key in keys if key not in ("indices", "batch_size", "shuffle")]
    dataset = TensorDataset(*tensors)

    # 还原 DataLoader 的配置信息
    indices = data.get("indices")
    subset = torch.utils.data.Subset(dataset,indices)
    batch_size = int(data['batch_size'])
    shuffle = bool(data['shuffle'])

    # 创建 DataLoader
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)




def save_dataloader(dataloader, file_path,shuffle):
    """
    将 DataLoader 的底层 Dataset 和配置信息保存到 .npz 文件。
    :param dataloader: 要保存的 DataLoader 对象
    :param file_path: 保存文件路径
    """
    dataset = dataloader.dataset
    indices = dataset.indices
    dataset = dataset.dataset

    if not isinstance(dataset, torch.utils.data.TensorDataset):
        raise TypeError()
    tensors = [t.numpy() for t in dataset.tensors]

    # 保存 Dataset 和 DataLoader 的配置信息
    np.savez(file_path, *tensors, indices=indices, batch_size=dataloader.batch_size, shuffle=shuffle)
    print(f"DataLoader 已保存到：{file_path}")



def save_dict(path, dict):
    with open(path,"wb") as file:
        pickle.dump(dict, file)

def load_dict(path):
    with open(path,"rb") as file:
        return pickle.load(file)




def random_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def convert_dict(d):
    new_d = {}
    for k, v in d.items():
        if isinstance(k, str) or k in ["-1", "-2"]:
            k = int(k)
        if isinstance(v, str) and v.isdigit():
            v = int(v)
        new_d[k] = v
    return new_d


def get_device(gpu=-1):
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu))
    else:
        device = torch.device("cpu")   
    logging.info(f"Calculating with {device}")
    
    return device