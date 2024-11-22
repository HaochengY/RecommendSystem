import json
import os
import logging

import pyarrow.parquet as pq

import torch.nn as nn
import tqdm
from sklearn.preprocessing import LabelEncoder
import torch.utils
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data
from tqdm import tqdm
import torch.multiprocessing as mp
from model.utils.Tokenizer import Tokenizer
from model.utils.utils import  get_device, random_all
import polars as pl

class DataRecorder:
    def __init__(self,test=False, embedding_dim=1, batch_size=128, gpu=0):
        random_all(2021)  # 设置一切随机数种为 2021
        self.test = test


        self.feature_map = {}
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.load_dataset_config() 
        self.device = get_device(gpu)
        train_ddf, valid_ddf, test_ddf = self.generate_encoder()

        self.generate_tensor(train_ddf, valid_ddf, test_ddf)

    
    def generate_tensor(self, train_ddf, valid_ddf, test_ddf):
        self.train_sample_num = train_ddf.height
        self.valid_sample_num = valid_ddf.height

        self.train_loader = self.transform(train_ddf, shuffle=False)
        self.valid_loader = self.transform(valid_ddf)
        self.test_loader = self.transform(test_ddf)
        
        

    def transform(self, ddf, shuffle=False):
        self.features = self.categorical_col + self.numerical_col
        self.feature_num = len(self.features)
        self.input_dim = self.feature_num * self.embedding_dim
        tensor_dict = {col:torch.tensor(ddf[col].to_list()) for col in ddf.columns}
        if self.test:
            label_dict = {col:torch.tensor(ddf[col].to_list()) for col in ["clk"]}
        else:
            label_dict = {col:torch.tensor(ddf[col].to_list()) for col in ["label"]}

        dataset = TensorDictDataset(tensor_dict, label_dict, self.device)
        dataloader = DataLoader(dataset, batch_size=self.batch_size,persistent_workers=True,
                                 shuffle=shuffle, pin_memory=True,
                                 num_workers=16, prefetch_factor=8)
        return dataloader

    def generate_encoder(self):
        all_exist = all(os.path.exists(path) for path in [os.path.join(self.root_path, "checkpoint", 'feature_map.json'),
                                                          os.path.join(self.root_path, "checkpoint", "encoded_column", "train", "encoded_table.parquet"),
                                                          os.path.join(self.root_path, "checkpoint", "encoded_column", "test", "encoded_table.parquet"),
                                                          os.path.join(self.root_path, "checkpoint", "encoded_column", "valid", "encoded_table.parquet")])
        if all_exist:
            self.logger.info("检测到已编码后的数据完整")
            train_ddf = pl.from_arrow(pq.read_table(os.path.join(self.root_path, "checkpoint", "encoded_column", "train", "encoded_table.parquet")))  
            test_ddf = pl.from_arrow(pq.read_table(os.path.join(self.root_path, "checkpoint", "encoded_column", "test", "encoded_table.parquet")))
            valid_ddf = pl.from_arrow(pq.read_table(os.path.join(self.root_path, "checkpoint", "encoded_column", "valid", "encoded_table.parquet")))
            with open(os.path.join(self.root_path, "checkpoint",'feature_map.json'), 'r', encoding='utf-8') as json_file:
                self.feature_map = json.load(json_file)
        else:
            self.ogger.info("检测到已编码后的数据不存在或不完整，编译中...")
            train_ddf, test_ddf, valid_ddf = self.preprocess_all() 
            self.processed_path = os.path.join(self.root_path, "train_data")
            tokenizer = Tokenizer(self.feature_map, self.embedding_dim, train_ddf, self.categorical_col, self.root_path)
            train_ddf = tokenizer.fit(train_ddf, "train")
            valid_ddf = tokenizer.fit(valid_ddf, "valid")
            test_ddf = tokenizer.fit(test_ddf, "test")
            
        return train_ddf, valid_ddf, test_ddf



        

    def preprocess(self, file_name):
        path = os.path.join(self.root_path, file_name)
        ddf = pl.read_parquet(source=path)
        self.logger.info(f"读取原始Parquet文件from {path}")
        if set(ddf.columns) != set(self.all_col):
            logging.exception(f"路径文件列不符合预期要求, 预期列:{set(self.all_col)}, 路径文件列：{set(ddf.columns)}")
            raise ValueError(f"路径文件列不符合预期要求, 预期列:{set(self.all_col)}, 路径文件列：{set(ddf.columns)}")
        return ddf




    def load_dataset_config(self):
        if self.test:
            self.root_path = "/home/yanghc03/fuxictr-main/FuxiCTR/data/tiny_parquet" 
            self.numerical_col = []
            self.categorical_col = ["userid", "adgroup_id", "pid", "cate_id", "campaign_id", "customer",
                            "brand", "cms_segid", "cms_group_id", "final_gender_code", "age_level",
                            "pvalue_level", "shopping_level", "occupation"]
            self.label_col = "clk"
            self.dataset_name = "tiny_parquet"
        
        else:
            self.root_path = "/home/yanghc03/dataset/Criteo_x1"
            self.numerical_col = ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12", "I13"]
            self.categorical_col = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", 
                                "C14", "C15", "C16","C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24",
                                    "C25", "C26"]
            self.label_col = "label"
            self.dataset_name = "Criteo_x1"


        self.all_col = self.numerical_col + self.categorical_col + [self.label_col]

        for col in self.categorical_col:
            self.feature_map[col]={"type":"categorical"}
        for col in self.numerical_col:
            self.feature_map[col]={"type":"numerical"}

        self.logger = logging.getLogger("my_logger")
        self.logger.setLevel(logging.INFO)

        # 创建一个文件 handler（写入模式）
        file_handler = logging.FileHandler(f'/home/yanghc03/recommended_system/RecommendSystem/checkpoint/{self.dataset_name}/Log.log',
                                           mode="a")
        file_handler.setLevel(logging.INFO)

        # 创建一个 stream handler（输出到控制台）
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        # 创建 formatter 并添加到 handler
        formatter = logging.Formatter('%(asctime)s,%(msecs)d %(levelname)s [%(name)s] %(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        # 添加 handler 到 logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)


        self.logger.info(f'Load data from root dir {self.root_path}')


    def preprocess_all(self):
        train_ddf = self.preprocess("train.parquet")
        test_ddf = self.preprocess("test.parquet")
        valid_ddf = self.preprocess("valid.parquet")
        return train_ddf, test_ddf, valid_ddf



from torch.utils.data import Dataset, DataLoader
class TensorDictDataset(Dataset):
    def __init__(self,feature_dict, label_dict, device):
        super(TensorDictDataset, self).__init__()
        self.feature_dict = {key: tensor for key, tensor in feature_dict.items()}
        self.label_dict = {key: tensor for key, tensor in label_dict.items()}
 

        self.keys = list(feature_dict.keys())
        self.label_keys = list(label_dict.keys())
        self.length = len(next(iter(feature_dict.values())))

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 获取特征和标签
        features = {key: self.feature_dict[key][idx] for key in self.keys}
        labels = {key: self.label_dict[key][idx] for key in self.label_keys}
        
        return features, labels




            


