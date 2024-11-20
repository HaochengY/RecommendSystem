import glob
import json
import os
import shutil
import polars as pl
import torch
import torch.nn as nn
import pyarrow.parquet as pq
from tqdm import tqdm
import pyarrow as pa
from model.utils.utils import convert_dict



class Tokenizer:
    def __init__(self, feature_map, embedding_dim, train_ddf, categorical_col, root_path):
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
            self.feature_map[col_name].update({"vocab_size": vocab_size+1})
            pbar.update(1)  
        pbar.close()  
        print("vocab 搭建完成")
        json_file_path = os.path.join(self.checkpoint, 'feature_map.json')
        # 将字典写入 JSON 文件
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(self.feature_map, json_file, indent=4, ensure_ascii=False)





    def encode_categorical(self, col_name, ddf):
        unique_values = ddf.select(pl.col(col_name)).get_column(col_name).unique().sort().to_list()
        unique_values.append(-1)
        unique_values.append(-2)
        vocab_size = len(unique_values)
        encoding_map={value:idx+1 for idx, value in enumerate(unique_values)}
        return encoding_map, vocab_size

    def encode_numerical(self):
        pass             

    def fit(self, ddf, partten):
        checkpoint_for_encoded_columns =  os.path.join(os.path.join(self.checkpoint, 'encoded_column'), partten)
        existed_encoded_table = os.path.join(checkpoint_for_encoded_columns, "encoded_table.parquet")
        if os.path.exists(existed_encoded_table):
            print(f"{existed_encoded_table}存在，直接调取...")
            # ddf = pq.read_table(existed_encoded_table)
        else:
            print(f"{existed_encoded_table}不存在，需要计算...")
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
                    # 使用 tqdm 包裹内部循环
                    for _, v in enumerate(tqdm(data_list, desc=f"Mapping {col_name}", unit="item")):
                        if v in encoding_map:
                            mapped_value = encoding_map[v]
                        else:
                            mapped_value = default_value
                        new_col.append(mapped_value)
                    with open(cp_path, 'w', encoding='utf-8') as json_file:
                        json.dump(new_col, json_file, indent=4, ensure_ascii=False)
                    print(f"encoded_col已储存到{cp_path}")
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
                print("单个列的parquet文件不存在")
                for col in ddf.columns:
                    try:
                        cp_path = os.path.join(checkpoint_for_encoded_columns, f'encoded_col_{col}.parquet')
                        ddf.select(pl.col(col)).to_parquet(cp_path)
                        print(f"column {col} saved 成功储存到{cp_path}")
                    except Exception as e:
                        raise ValueError(f"Error saving column {col}: {e}")
                    
            if not parquet_files: create_parquet(ddf)

            parquet_files = glob.glob(os.path.join(checkpoint_for_encoded_columns, '*.parquet'))
            if len(parquet_files) == 0:
                raise ValueError("没找到parquet_files")
            print("单个列的parquet文件存在")
            columns = []
            for file in parquet_files:
                table = pq.read_table(file)  # 读取 Parquet 文件
                columns.append(table)
            ddf = pa.Table.from_arrays([col[column_name] 
                                                for col in columns for column_name in col.column_names],
                                                names=[col.column_names[0] for col in columns])
            pq.write_table(ddf, existed_encoded_table)
            print(f"合并完成并保存为{existed_encoded_table}")
        ddf = pq.read_table(existed_encoded_table)
        # TODO: 有bug
        ddf = pl.from_arrow(ddf)   
        return ddf


