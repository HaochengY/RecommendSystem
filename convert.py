import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

def convert_csv_to_parquet(input_csv, output_parquet, chunk_size=None):
    """
    Convert a large CSV file to a single Parquet file using chunks.

    Args:
        input_csv (str): Path to the input CSV file.
        output_parquet (str): Path to the output Parquet file.
        chunk_size (int, optional): Number of rows to process at a time for large files.
    """
    if chunk_size:
        print(f"Converting CSV to Parquet in chunks of {chunk_size} rows...")

        # Open a ParquetWriter
        writer = None

        # Get total rows for progress bar
        total_rows = sum(1 for _ in open(input_csv, 'r')) - 1  # Exclude header row
        num_chunks = (total_rows + chunk_size - 1) // chunk_size  # Total chunks

        with tqdm(total=num_chunks, desc="Processing chunks", unit="chunk") as pbar:
            for chunk in pd.read_csv(input_csv, chunksize=chunk_size):
                # Convert each chunk to Arrow Table
                table = pa.Table.from_pandas(chunk)
                if writer is None:
                    writer = pq.ParquetWriter(output_parquet, table.schema, compression='snappy')
                writer.write_table(table)
                pbar.update(1)
            
            if writer:
                writer.close()  # Close the writer after all chunks are written
    else:
        print("Converting CSV to Parquet without chunking...")
        df = pd.read_csv(input_csv).head(1000)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_parquet, compression='snappy')
    
    print(f"Conversion completed: {output_parquet}")

if __name__ == "__main__":
    convert_csv_to_parquet("/home/yanghc03/python/RecommendSystem/dataset/hdfs_data/train.csv", "/home/yanghc03/dataset/Criteo_x1/demo/train.parquet")
    convert_csv_to_parquet("/home/yanghc03/python/RecommendSystem/dataset/hdfs_data/test.csv", "/home/yanghc03/dataset/Criteo_x1/demo/test.parquet")
    convert_csv_to_parquet("/home/yanghc03/python/RecommendSystem/dataset/hdfs_data/valid.csv", "/home/yanghc03/dataset/Criteo_x1/demo/valid.parquet")
