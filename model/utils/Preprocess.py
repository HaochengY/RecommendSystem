from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pickle
import h5py
import random
import time
import fastavro
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch.nn as nn
import torch
from sklearn.preprocessing import LabelEncoder
import torch.utils
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.utils.data
from tqdm import tqdm
import torch.multiprocessing as mp
from joblib import dump, load

class DataRecorder:
    def __init__(self,
                 dataset_name="Criteo_x1", embedding_dim=12, batch_size=256):
        mp.set_start_method("spawn", force=True)
        self.random_all(42)  # 设置一切随机数种为 42

        self.existed_datarecoder_path = f"/home/yanghc03/dataset/{dataset_name}/emb_{embedding_dim}.pkl"
        self.parquet_path_linux = f"/home/yanghc03/dataset/{dataset_name}/data_demo.parquet"
        self.dataset_name = dataset_name
        self.parquet_table = None
        self.label_name = "label"
        self.encoder_type = "labelencoder"
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim

        if os.path.exists(self.existed_datarecoder_path):
            # 为了重复实验超参数而省略每次的加载过程
            existed_datarecorder = self.load(self.existed_datarecoder_path)
            self.__dict__.update(existed_datarecorder.__dict__)
            print(f"在指定路径{self.existed_datarecoder_path}: 找到datarecorder并加载成功")

        else:
            print(f"在指定路径{self.existed_datarecoder_path}: 未找到datarecorder，开始初始化计算")
            self.read_parquet(self.parquet_path_linux)
            self.label_encoders = {}
            self.label_tensor = None
            self.encoded_tensor = None
            self.feature_name_list = []
            self.feature_num = 0
            self.encoded_data = self.encode_process()
            self.vocab_sizes = {}
            self.embedding_dict = {}
            self.embedding_to_1_dim_dict = {}
            self.embedding_schema()
            self.train_loader = None
            self.val_loader = None
            self.test_loader = None
            self.num_sample = 0
            self.input_dim = 0
            self.create_dataset()
            del self.parquet_table
            self.save(self.existed_datarecoder_path, chunk_size=10_000_000)  # 分块大小 10MB

            print(f"初始化成功，datarecorder已经成功储存到:{self.existed_datarecoder_path}")

        print("============================================")
        print(f"Loading Dataset {self.dataset_name} Successfully!")
        print(f"Sample number: {self.num_sample}")
        print(f"Feature number: {self.feature_num}")
        print("============================================")
        for k, v in self.__dict__.items():
            print(f"变量名:{k}, 类型:{type(v)}")
        raise KeyboardInterrupt
    
    def save(self, filepath, chunk_size=1_000_000):
        """
        分块保存整个类对象
        :param filepath: 保存文件的路径（基础路径）
        :param chunk_size: 分块大小（字节数）
        """
        with open(filepath, "wb") as f:
            pickled_data = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
            total_size = len(pickled_data)
            # 分块写入
            for i in range(0, total_size, chunk_size):
                chunk = pickled_data[i:i + chunk_size]
                f.write(chunk)
                print(f"已写入第 {i // chunk_size + 1} 块，共 {total_size // chunk_size + 1} 块")


    @classmethod
    def load(cls, filepath):
        """
        从分块文件加载整个类对象
        :param filepath: 文件路径
        :return: 重建的类对象
        """
        with open(filepath, "rb") as f:
            pickled_data = f.read()  # 读取所有分块
            obj = pickle.loads(pickled_data)  # 反序列化为对象
        print(f"对象已从 {filepath} 加载成功")
        return obj


    @staticmethod
    def random_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def create_dataset(self):
        if self.encoded_tensor.numel() == 0 or self.label_tensor.numel() == 0:
            raise ValueError("create_dataset 的时候有空值")
        self.dataset = TensorDataset(self.encoded_tensor, self.label_tensor)

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
        self.num_sample = len(self.dataset)
        self.input_dim = self.feature_num * self.embedding_dim



    def read_parquet(self, parquet_path):
        print("============================================")
        print(f"Loading Dataset from {self.parquet_path_linux}...")
        self.parquet_table = pq.ParquetFile(parquet_path)
        self.num_row_groups = self.parquet_table.num_row_groups

    @staticmethod
    def encode_column(encoder_type, parquet_path, col, row_group_indices):
        """
        对单个列进行编码
        """
        torch.cuda.set_device(0)

        parquet_table = pq.ParquetFile(parquet_path)
        if encoder_type == "labelencoder":
            encoder = LabelEncoder()
        encoded_col = []
        for rg_idx in row_group_indices:
            col_data_batch = parquet_table.read_row_group(rg_idx, columns=[col]).column(0).to_pylist()
            encoder_batch = encoder.fit_transform(col_data_batch)
            encoded_col.append(encoder_batch)
        encoded_data = np.concatenate(encoded_col)
        torch.cuda.empty_cache()
        return col, encoded_data, encoder



    def encode_process(self):
        """
        执行编码操作
        :return:
        """

        encoded_data = {}
        columns = self.parquet_table.schema.names
        all_encoded_cols = []
        total_steps = len(columns) * self.num_row_groups  # 总步数等于列数 * 分组数
        file_path = self.parquet_path_linux
        with tqdm(total=total_steps, desc="Encoding Columns", unit=" steps") as pbar:
            with ProcessPoolExecutor(max_workers=1) as executor:
                futures = []
                for col in columns:
                    if col == self.label_name:
                        label_col = []
                        for rg_idx in range(self.num_row_groups):
                            label_batch = self.parquet_table.read_row_group(rg_idx, columns=[col]).column(0).to_pylist()
                            label_col.append(label_batch)
                            pbar.update(1)  # 更新进度条
                        self.label_tensor = torch.tensor(np.concatenate(label_col), dtype=torch.float32).unsqueeze(1)
                        continue
                    futures.append(executor.submit(DataRecorder.encode_column, self.encoder_type, file_path, col, range(self.num_row_groups)))

                for future in as_completed(futures):
                    col, encoded_col, encoder = future.result()
                    if col == "sponsor_id":
                        self.sponsor_id_tensor = torch.tensor(encoded_col, dtype=torch.float32).unsqueeze(1)
                    encoded_data[col] = encoded_col
                    self.label_encoders[col] = encoder
                    all_encoded_cols.append(encoded_col)
                    pbar.update(self.num_row_groups)

            executor.shutdown(wait=True)

        self.encoded_tensor = torch.tensor(np.stack(all_encoded_cols, axis=1), dtype=torch.long)
        columns.remove(self.label_name)
        self.feature_name_list = columns
        self.feature_num = len(columns)
        return encoded_data


    def embedding_schema(self):
        """
        只是给出schema而不执行embedding，所以不用分片
        :return:
        """
        print("Generating Embedding Schema....")
        for col, data in self.encoded_data.items():
            vocab_size = len(np.unique(data))
            self.vocab_sizes[col] = vocab_size
            self.embedding_dict[col] = nn.Embedding(vocab_size, self.embedding_dim)
            self.embedding_to_1_dim_dict[col] = nn.Embedding(vocab_size, 1)
        print("Finished!")
        print("============================================")


