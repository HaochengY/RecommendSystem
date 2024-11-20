import os
import pickle
import random
import numpy as np
import pyarrow.parquet as pq
import torch.nn as nn
import torch
from sklearn.preprocessing import LabelEncoder
import torch.utils
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data
from tqdm import tqdm
import torch.multiprocessing as mp
from model.utils.utils import load_dataloader, load_dict, save_dataloader, save_dict

class DataRecorder:
    def __init__(self,
                 dataset_name="Criteo_x1", embedding_dim=12, batch_size=4096):
        mp.set_start_method("spawn", force=True)
        self.random_all(2021)  # 设置一切随机数种为 2021

        self.existed_datarecoder_path = f"/home/yanghc03/dataset/{dataset_name}/emb_{embedding_dim}"
        self.parquet_path_linux = f"/home/yanghc03/dataset/{dataset_name}/data.parquet"
        self.dataset_name = dataset_name
        self.parquet_table = None
        self.label_name = "label"
        self.encoder_type = "labelencoder"
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim

        self.read_parquet(self.parquet_path_linux)
        self.label_encoders = {}
        self.label_tensor = None
        self.encoded_tensor = None
        self.feature_name_list = []
        self.feature_num = 0
        self.vocab_sizes = {}
        self.embedding_dict = {}
        self.embedding_to_1_dim_dict = {}
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.num_sample = 0
        self.input_dim = 0
        dataloader_path = os.path.join(self.existed_datarecoder_path, "train_loader_dataset.npz")
        if os.path.exists(dataloader_path) and os.path.exists(os.path.join(self.existed_datarecoder_path, "embedding_dict.pkl")):
            with tqdm(total=3, desc=f"完整的dataloader和embedding dict数据存在，直接导入,可以跳过编码，导入中...", unit=" steps") as pbar:
                self.train_loader = load_dataloader(os.path.join(self.existed_datarecoder_path, "train_loader_dataset.npz"))
                pbar.update(1)
                self.val_loader = load_dataloader(os.path.join(self.existed_datarecoder_path, "val_loader_dataset.npz"))
                pbar.update(1)
                self.test_loader = load_dataloader(os.path.join(self.existed_datarecoder_path, "test_loader_dataset.npz"))
                pbar.update(1)
            print("导入完成！")
        else:
            print(f"dataloader和embedding dict数据不完整，需要计算...")
            self.encoded_data = self.encode_process()
            self.embedding_schema()
        if os.path.exists(dataloader_path):
            print("完整的dataloader数据存在，直接导入,可以跳过dataset")
        else:
            self.create_dataset()
        del self.parquet_table
        if self.num_sample != 0:
            print("储存其他变量中")
            self.save_other_variables()
        else:
            print("其他变量没有初始化，导入中")
            self.load_other_variables(os.path.join(self.existed_datarecoder_path, "other_var.pkl"))
        print("============================================")
        print(f"Loading Dataset {self.dataset_name} Successfully!")
        print(f"Sample number: {self.num_sample}")
        print(f"Feature number: {self.feature_num}")
        print("============================================")


    def load_other_variables(self, input_file):
        with open(input_file, 'rb') as file:
            data = pickle.load(file)
        keys = []
        for key, value in data.items():
            setattr(self, key, value)
            keys.append(key)
        print(f"类变量{keys}已从 {input_file} 恢复")
    def save_other_variables(self):

        blacklist = ["label_tensor", "encoded_tensor", "train_loader", "val_loader", "test_loader", "dataset"]
        class_variables = vars(self)
        # 过滤掉黑名单中的变量
        filtered_variables = {key: value for key, value in class_variables.items() if key not in blacklist}

        # 保存到 Pickle 文件
        with open(os.path.join(self.existed_datarecoder_path, 'other_var.pkl'), 'wb') as file:
            pickle.dump(filtered_variables, file)

        print(f"过滤后的类变量已保存到 {os.path.join(self.existed_datarecoder_path, 'other_var.pkl')}")
    @staticmethod
    def random_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def create_dataset(self):
        dataloader_path = os.path.join(self.existed_datarecoder_path, "train_loader_dataset.npz")
        if os.path.exists(dataloader_path):
            print("完整的dataloader数据存在，直接导入")
            self.train_loader = load_dataloader(os.path.join(self.existed_datarecoder_path, "train_loader_dataset.npz"))
            self.val_loader = load_dataloader(os.path.join(self.existed_datarecoder_path, "val_loader_dataset.npz"))
            self.test_loader = load_dataloader(os.path.join(self.existed_datarecoder_path, "test_loader_dataset.npz"))
        else:
            print("dataloader数据不存在，需要计算")
            if self.encoded_tensor.numel() == 0 or self.label_tensor.numel() == 0:
                raise ValueError("create_dataset 的时候有空值")
            dataset_path = os.path.join(self.existed_datarecoder_path, "dataset.npz")
            if os.path.exists(dataset_path):
                print(f"{dataset_path}文件存在，加载中...")
                data = np.load(dataset_path)
                tensors = [torch.tensor(data[key]) for key in data.files]
                self.dataset = TensorDataset(*tensors)
                print("加载成功")
            else:
                print(f"{dataset_path}文件不存在，计算中...")
                self.dataset = TensorDataset(self.encoded_tensor, self.label_tensor)
                tensors = [t.cpu().numpy() for t in self.dataset.tensors]
                np.savez(dataset_path, *tensors)
                print(f"成功存储搭配{dataset_path}")

            total_size = len(self.dataset)
            train_size = int(0.7 * total_size)
            val_size = int(0.15 * total_size)
            test_size = total_size - train_size - val_size
            # 按照时间顺序划分
            train_dataset = torch.utils.data.Subset(self.dataset, range(0, train_size))
            val_dataset = torch.utils.data.Subset(self.dataset, range(train_size, train_size + val_size))
            test_dataset = torch.utils.data.Subset(self.dataset, range(train_size + val_size, total_size))
            
        
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)
            self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)
            self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)
            print("储存dataloader中...")
            save_dataloader(self.train_loader,os.path.join(self.existed_datarecoder_path, "train_loader_dataset.npz"),True)
            save_dataloader(self.val_loader,os.path.join(self.existed_datarecoder_path, "val_loader_dataset.npz"),False)
            save_dataloader(self.test_loader,os.path.join(self.existed_datarecoder_path, "test_loader_dataset.npz"),False)
            print("储存成功")
        self.num_sample = len(self.dataset)
        print(self.num_sample)
        self.input_dim = self.feature_num * self.embedding_dim



    def read_parquet(self, parquet_path):
        print("============================================")
        print(f"Loading Dataset from {self.parquet_path_linux}...")
        self.parquet_table = pq.ParquetFile(parquet_path)
        self.num_row_groups = self.parquet_table.num_row_groups

    def encode_process(self):
        """
        执行编码操作，检查是否已存在编码后的列文件。
        如果存在，直接加载；否则重新编码并保存。
        """
        encoded_dir = os.path.join(self.existed_datarecoder_path, "encoded_columns")  # 子文件夹路径
        os.makedirs(encoded_dir, exist_ok=True)  # 创建子文件夹

        label_file_path = os.path.join(self.existed_datarecoder_path, "label_tensor.pt")  # 标签文件路径

        # 初始化存储变量
        encoded_data = {}
        all_encoded_cols = []

        # 处理标签列
        if os.path.exists(label_file_path):
            print("检测到已存在的标签文件，直接加载...")
            self.label_tensor = torch.load(label_file_path)
            print("加载成功")
        else:
            print("未检测到标签文件，开始处理标签列...")
            label_col = []
            for rg_idx in range(self.num_row_groups):
                label_batch = self.parquet_table.read_row_group(rg_idx, columns=[self.label_name]).column(0).to_pylist()
                label_col.append(label_batch)
            self.label_tensor = torch.tensor(np.concatenate(label_col), dtype=torch.float32).unsqueeze(1)
            torch.save(self.label_tensor, label_file_path)  # 保存标签文件



        encoded_tensor_path = os.path.join(self.existed_datarecoder_path, "encoded_tensor.pt")
        if os.path.exists(encoded_tensor_path):
            print(f"检测到已存在的完整的编码文件{encoded_tensor_path},加载中...")
            self.encoded_tensor = torch.load(encoded_tensor_path)
            print("加载成功")

        else:
            print("未检测到完整的编码文件，开始梳理标签列...")

            total_steps = self.num_row_groups
            for col in self.parquet_table.schema.names:
                if col == self.label_name:
                    continue  # 标签列已处理，跳过

                col_file_path = os.path.join(encoded_dir, f"{col}.pt")  # 当前列的编码结果文件路径

                if os.path.exists(col_file_path):
                    print(f"检测到已存在的编码文件：{col}.pt")
                    encoded_col = torch.load(col_file_path).numpy()  # 加载编码结果
                    all_encoded_cols.append(encoded_col)
                else:
                    encoder = LabelEncoder()
                    encoded_col_batches = []
                    with tqdm(total=total_steps, desc=f"未检测到编码文件：{col}.pt，开始编码...", unit=" steps") as pbar:

                        for rg_idx in range(self.num_row_groups):
                            col_data_batch = self.parquet_table.read_row_group(rg_idx, columns=[col]).column(0).to_pylist()
                            encoder_batch = encoder.fit_transform(col_data_batch)
                            encoded_col_batches.append(encoder_batch)
                            pbar.update(1)  # 更新进度条

                        encoded_col = np.concatenate(encoded_col_batches)
                        torch.save(torch.tensor(encoded_col), col_file_path)  # 保存编码结果
                        encoded_data[col] = encoded_col
                        self.label_encoders[col] = encoder
                        all_encoded_cols.append(encoded_col)

                if col == "sponsor_id":  # 特殊处理 sponsor_id 列
                    self.sponsor_id_tensor = torch.tensor(encoded_col, dtype=torch.float32).unsqueeze(1)

            # 将所有编码后的列堆叠为 Tensor
            self.encoded_tensor = torch.tensor(np.stack(all_encoded_cols, axis=1), dtype=torch.long)
            print(f"正在保存encoded_tensor...")
            torch.save(self.encoded_tensor, encoded_tensor_path)  # 保存整体编码结果
            print(f"编码完成，已保存到：{encoded_tensor_path}")

        # 更新属性
        columns = self.parquet_table.schema.names
        columns.remove(self.label_name)
        self.feature_name_list = columns
        self.feature_num = len(columns)

        return encoded_data

    def embedding_schema(self):
        """
        只是给出schema而不执行embedding，所以不用分片
        :return:
        """
       
        if os.path.exists(os.path.join(self.existed_datarecoder_path, "embedding_dict.pkl")):
            print("找到字典信息，加载中....")
            self.embedding_dict = load_dict(os.path.join(self.existed_datarecoder_path, "embedding_dict.pkl"))
            self.embedding_to_1_dim_dict = load_dict(os.path.join(self.existed_datarecoder_path, "embedding_to_1_dim_dict.pkl"))
            print("加载成功")

        else:
            print("没找到字典信息，Generating Embedding Schema....")
            for col, data in self.encoded_data.items():
                vocab_size = len(np.unique(data))
                self.vocab_sizes[col] = vocab_size
                self.embedding_dict[col] = nn.Embedding(vocab_size, self.embedding_dim)
                self.embedding_to_1_dim_dict[col] = nn.Embedding(vocab_size, 1)
            save_dict(os.path.join(self.existed_datarecoder_path, "embedding_dict.pkl"), self.embedding_dict)
            save_dict(os.path.join(self.existed_datarecoder_path, "embedding_to_1_dim_dict.pkl"), self.embedding_to_1_dim_dict)
            print("储存成功")


        print("Finished!")
        print("============================================")


