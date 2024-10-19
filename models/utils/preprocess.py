import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split


def import_data():
    data = pd.read_csv("../../dataset/train_data.csv")
    print("Dataset loaded successfully.")
    encoders = {}
    embeddings = {}
    col_list = data.columns.tolist()[1:-1]
    # 为每个类别型特征列创建LabelEncoder和Embedding层
    for column in col_list:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        encoders[column] = le  # 存储LabelEncoder
        num_embeddings = len(le.classes_)
        embeddings[column] = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=12)  # 存储Embedding层

    input_data = {column: torch.tensor(data[column].values, dtype=torch.long) for column in col_list}

    embedded_features = {}
    for column, tensor in input_data.items():
        embedded_features[column] = embeddings[column](tensor)

    input_tensor = torch.cat([embedded_features[column] for column in col_list], dim=1)
    label_tensor = torch.tensor(data["label"].values, dtype=torch.float32).unsqueeze(1)
    num_feature = input_tensor.size()[1]

    dataset = TensorDataset(input_tensor, label_tensor)
    # 创建 DataLoader

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    # 进行随机拆分
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader, test_loader, num_feature
