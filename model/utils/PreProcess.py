import json
import os
import shutil

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
from model.utils.utils import  random_all
import polars as pl

class DataRecorder:
    def __init__(self,test=True, embedding_dim=4, batch_size=128, gpu=False):
        random_all(2021)  # 设置一切随机数种为 2021

        self.test = test
        if gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.dataset_name = "Criteo_x1"
        self.feature_map = {}
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.root_path = "/home/yanghc03/fuxictr-main/FuxiCTR/data/tiny_parquet" if test else "/home/yanghc03/dataset/Criteo_x1"
        train_ddf, valid_ddf, test_ddf = self.generate_encoder()

        self.generate_tensor(train_ddf, valid_ddf, test_ddf)

    
    def generate_tensor(self, train_ddf, valid_ddf, test_ddf):
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
        dataloader = DataLoader(dataset,batch_size=self.batch_size, shuffle=shuffle)
        return dataloader

    def generate_encoder(self):
        all_exist = all(os.path.exists(path) for path in [os.path.join(self.root_path, "checkpoint", 'feature_map.json'),
                                                          os.path.join(self.root_path, "checkpoint", "encoded_column", "train", "encoded_table.parquet"),
                                                          os.path.join(self.root_path, "checkpoint", "encoded_column", "test", "encoded_table.parquet"),
                                                          os.path.join(self.root_path, "checkpoint", "encoded_column", "valid", "encoded_table.parquet")])
        if all_exist:
            print("检测到已编码后的数据完整")
            self.load_dataset_config(if_preprocess=False) 
            train_ddf = pl.from_arrow(pq.read_table(os.path.join(self.root_path, "checkpoint", "encoded_column", "train", "encoded_table.parquet")))  
            test_ddf = pl.from_arrow(pq.read_table(os.path.join(self.root_path, "checkpoint", "encoded_column", "test", "encoded_table.parquet")))
            valid_ddf = pl.from_arrow(pq.read_table(os.path.join(self.root_path, "checkpoint", "encoded_column", "valid", "encoded_table.parquet")))
            with open(os.path.join(self.root_path, "checkpoint",'feature_map.json'), 'r', encoding='utf-8') as json_file:
                self.feature_map = json.load(json_file)
        else:
            print(f"检测到已编码后的数据不存在或不完整，编译中...")
            train_ddf, test_ddf, valid_ddf = self.load_dataset_config() 
            self.processed_path = os.path.join(self.root_path, "train_data")
            tokenizer = Tokenizer(self.feature_map, self.embedding_dim, train_ddf, self.categorical_col, self.root_path)
            train_ddf = tokenizer.fit(train_ddf, "train")
            valid_ddf = tokenizer.fit(valid_ddf, "valid")
            test_ddf = tokenizer.fit(test_ddf, "test")
            
        return train_ddf, valid_ddf, test_ddf



        

    def preprocess(self, file_name):
        path = os.path.join(self.root_path, file_name)
        ddf = pl.read_parquet(source=path)
        # if self.test:
        #     ddf = ddf.select(["userid", "adgroup_id", "pid", "cate_id", "campaign_id", "customer",
        #                           "brand", "cms_segid", "cms_group_id", "final_gender_code", "age_level",
        #                           "pvalue_level", "shopping_level", "occupation", "clk"])
        if set(ddf.columns) != set(self.all_col):
            raise ValueError(f"路径文件列不符合预期要求, 预期列:{set(self.all_col)}, 路径文件列：{set(ddf.columns)}")
        return ddf

        




    def load_dataset_config(self, if_preprocess=True):
        if self.test:
            self.numerical_col = []
            self.categorical_col = ["userid", "adgroup_id", "pid", "cate_id", "campaign_id", "customer",
                            "brand", "cms_segid", "cms_group_id", "final_gender_code", "age_level",
                            "pvalue_level", "shopping_level", "occupation"]
            # self.categorical_col = ['userid']
            self.label_col = "clk"
        
        else:
            self.numerical_col = ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12", "I13"]
            self.categorical_col = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", 
                                "C14", "C15", "C16","C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24",
                                    "C25", "C26"]
            self.label_col = "label"
            
        self.all_col = self.numerical_col + self.categorical_col + [self.label_col]

        for col in self.categorical_col:
            self.feature_map[col]={"type":"categorical"}
        for col in self.numerical_col:
            self.feature_map[col]={"type":"numerical"}
        if if_preprocess:
            train_ddf = self.preprocess("train.parquet")
            test_ddf = self.preprocess("test.parquet")
            valid_ddf = self.preprocess("valid.parquet")
            return train_ddf, test_ddf, valid_ddf



from torch.utils.data import Dataset, DataLoader
class TensorDictDataset(Dataset):
    def __init__(self,feature_dict, label_dict, device):
        super(TensorDictDataset, self).__init__()
        self.feature_dict = feature_dict
        self.label_dict = label_dict
        self.keys = list(feature_dict.keys())
        self.label_keys = list(label_dict.keys())
        self.device = device
        self.length = len(next(iter(feature_dict.values())))

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        features = {key: self.feature_dict[key][idx].to(self.device)
                     for key in self.keys}
        labels = {key: self.label_dict[key][idx].to(self.device)
                   for key in self.label_keys}
        return features, labels




            


