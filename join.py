

# import pyarrow.csv as pv
# import pyarrow.parquet as pq
# import pyarrow as pa
# from tqdm import tqdm

# csv_files = ['/home/yanghc03/python/RecommendSystem/dataset/hdfs_data/train.csv','/home/yanghc03/python/RecommendSystem/dataset/hdfs_data/valid.csv','/home/yanghc03/python/RecommendSystem/dataset/hdfs_data/test.csv']
# output_file = '/home/yanghc03/dataset/Criteo_x1/data.parquet'



# # 初始化进度条
# pbar = tqdm(total=len(csv_files), desc="Processing CSV files")

# # 合并所有CSV文件
# tables = []
# for file in csv_files:
#     # 使用Pyarrow读取CSV
#     table = pv.read_csv(file)
#     tables.append(table)
#     pbar.update(1)  # 更新进度条

# pbar.close()

# # 合并为一个Pyarrow Table
# combined_table = pa.concat_tables(tables).head(100)

# # 写入Parquet文件（使用snappy压缩以优化存储）
# pq.write_table(combined_table, output_file, compression="snappy")


import pyarrow.csv as pv
import pyarrow.parquet as pq

# 文件路径（替换为你的CSV文件路径）
csv_file = '/home/yanghc03/python/RecommendSystem/dataset/hdfs_data/train.csv'
output_file = '/home/yanghc03/dataset/Criteo_x1/data_demo.parquet'

# 读取CSV的前1000行
table = pv.read_csv(csv_file, read_options=pv.ReadOptions(skip_rows=0, column_names=None, autogenerate_column_names=False, block_size=None)).slice(0, 1000)

# 保存为Parquet格式
pq.write_table(table, output_file, compression="snappy")

print(f"前1000行已保存为 {output_file}")