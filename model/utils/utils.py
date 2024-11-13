import pickle
import torch
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