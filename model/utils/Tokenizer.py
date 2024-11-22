from collections import Counter
import glob
import json
import logging
import os
import shutil
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import pyarrow.parquet as pq
from tqdm import tqdm
import pyarrow as pa
from model.utils.utils import convert_dict
"""
-2 是OOV
-1 是PAD（目前没有用）
"""


class Tokenizer:
    def __init__(self, feature_map, embedding_dim, train_ddf, categorical_col, root_path):
        self.logger = logging.getLogger('my_logger')
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.root_path = root_path
        self.checkpoint = os.path.join(self.root_path, 'checkpoint')
        self.categorical_col = categorical_col
        self.build_vocab(train_ddf)
        


    def build_vocab(self, ddf):
        checkpoint_for_encoding_map =  os.path.join(self.checkpoint, 'encoding_map')
        if not os.path.exists(checkpoint_for_encoding_map):
            os.makedirs(checkpoint_for_encoding_map) 
        self.encoding_maps = {}
        # 创建一个 tqdm 对象
        pbar = tqdm(total=len(self.categorical_col), desc="Building Vocab...", unit="column")
        for col_name in self.categorical_col:
            cp_path = os.path.join(checkpoint_for_encoding_map, f'encoding_map_{col_name}.json')
            pbar.set_description(f"Processing {col_name}")  # 更新进度条描述
            if os.path.exists(cp_path): 
                with open(cp_path, 'r', encoding='utf-8') as json_file:
                    encoding_map = json.load(json_file)
                    encoding_map = convert_dict(encoding_map)
                    vocab_size = len(encoding_map)
            else:              
                encoding_map, vocab_size = self.encode_categorical(col_name, ddf)
                with open(cp_path, 'w', encoding='utf-8') as json_file:
                    json.dump(encoding_map, json_file, indent=4, ensure_ascii=False)
            self.encoding_maps[col_name] = encoding_map
            self.feature_map[col_name].update({"vocab_size": vocab_size})
            pbar.update(1)  
        pbar.close()  
        self.logger.info(f"vocab 搭建完成, feature_map:{self.feature_map}")

        with open(os.path.join(self.checkpoint, 'feature_map.json'), 'w', encoding='utf-8') as json_file:
            json.dump(self.feature_map, json_file, indent=4, ensure_ascii=False)
        self.logger.info(f"feature_map成功写入{os.path.join(self.checkpoint, 'feature_map.json')}")

        with open(os.path.join(self.checkpoint, 'encoding_maps.json'), 'w', encoding='utf-8') as json_file:
            json.dump(self.encoding_maps, json_file, indent=4, ensure_ascii=False)
        self.logger.info(f"vocab成功写入{os.path.join(self.checkpoint, 'encoding_maps.json')}")





    def encode_categorical(self, col_name, ddf):
        values = ddf.select(pl.col(col_name)).get_column(col_name).to_list()
        word_counts = []
        for k, v in Counter(values).items():
            word_counts.append((str(k), np.int64(v)))
        word_counts = sorted(word_counts, key=lambda x: (-x[1], x[0]))
        words = [int(k) for k, _ in word_counts]
        encoding_map={}
        encoding_map[-1] = 0
        encoding_map.update(dict((token, idx) for idx, token in enumerate(words, 1)))
        vocab_size = len(encoding_map)
        encoding_map.update({-2:vocab_size})
        return encoding_map, vocab_size + 1

    def encode_numerical(self):
        pass             

    def fit(self, ddf, partten):
        checkpoint_for_encoded_columns =  os.path.join(os.path.join(self.checkpoint, 'encoded_column'), partten)
        existed_encoded_table = os.path.join(checkpoint_for_encoded_columns, "encoded_table.parquet")
        if os.path.exists(existed_encoded_table):
            self.logger.info(f"编码后的{partten}数据存在，调取from {existed_encoded_table}")
        else:
            self.logger.info(f"编码后的{partten}数据不存在或不完整，计算中...")

            if not os.path.exists(checkpoint_for_encoded_columns):
                os.makedirs(checkpoint_for_encoded_columns) 
            pbar = tqdm(total=len(self.categorical_col), desc="Encoding columns", unit="column")
            for col_name in self.categorical_col:
                cp_path = os.path.join(checkpoint_for_encoded_columns, f'encoded_col_{col_name}.json')
                if os.path.exists(cp_path): 
                    with open(cp_path, 'r', encoding='utf-8') as json_file:
                        new_col = json.load(json_file)
                else:
                    encoding_map = self.encoding_maps[col_name]
                    default_value = encoding_map.get(-2)
                    if not default_value:
                        print(encoding_map)
                        raise ValueError
                    new_col = []
                    data_list = ddf.select(pl.col(col_name)).to_series().to_list()
                    for _, v in enumerate(tqdm(data_list, desc=f"Mapping {col_name}", unit="item")):
                        if v in encoding_map:
                            mapped_value = encoding_map[v]
                        else:
                            mapped_value = default_value
                        new_col.append(mapped_value)
                    with open(cp_path, 'w', encoding='utf-8') as json_file:
                        json.dump(new_col, json_file, indent=4, ensure_ascii=False)
                pbar.update(1)
                try:
                    temp = new_col
                    new_col=[int(x) for x in new_col]
                except Exception as e:
                    print(temp[:100])
                    print(new_col[:100])
                    raise e
                ddf = ddf.with_column(pl.Series(name=col_name, values=new_col))
            pbar.close()


            parquet_files = glob.glob(os.path.join(checkpoint_for_encoded_columns, '*.parquet'))
            
            def create_parquet(ddf):
                self.logger.info(f"单个列的parquet文件不存在,重新计算并储存至 {checkpoint_for_encoded_columns}")
                for col in ddf.columns:
                    try:
                        cp_path = os.path.join(checkpoint_for_encoded_columns, f'encoded_col_{col}.parquet')
                        ddf.select(pl.col(col)).to_parquet(cp_path)
                    except Exception as e:
                        logging.exception(f"Error saving column {col}: {e}")
                        raise ValueError(f"Error saving column {col}: {e}")
                    
            if not parquet_files: create_parquet(ddf)

            parquet_files = glob.glob(os.path.join(checkpoint_for_encoded_columns, '*.parquet'))
            if len(parquet_files) == 0:
                raise ValueError("没找到parquet_files")
            self.logger.info(f"单个列的parquet文件加载成功from {checkpoint_for_encoded_columns}")
            
            columns = []
            for file in parquet_files:
                table = pq.read_table(file)  # 读取 Parquet 文件
                columns.append(table)
            ddf = pa.Table.from_arrays([col[column_name] 
                                                for col in columns for column_name in col.column_names],
                                                names=[col.column_names[0] for col in columns])
            pq.write_table(ddf, existed_encoded_table)
            self.logger.info(f"完整的parquet文件合并成功，储存到{existed_encoded_table}")
        ddf = pq.read_table(existed_encoded_table)
        # TODO: 有bug
        ddf = pl.from_arrow(ddf)   
        return ddf


