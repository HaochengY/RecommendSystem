from collections import defaultdict
import datetime
import os
from time import sleep
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import log_loss, roc_auc_score
from tqdm import tqdm

from model.utils.Layers import Embedding_layer

class BaseModel(nn.Module):
    def __init__(self, 
                 model_name,
                 dataRecorder,
                 dataset_name="Criteo_x1",
                 patience=2,
                 num_epochs=30, 
                 learning_rate=0.001,
                 criterion_type="BCE", 
                 optimizer_type="Adam",
                 weight_decay=0,
                 record=False
                 ):
        super(BaseModel, self).__init__()
        self.start_time = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Calculating with {self.device}")
        self.dataRecorder = dataRecorder
        self.dataset_name = dataset_name
        self.input_dim = self.dataRecorder.input_dim
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.criterion_type = criterion_type
        self.optimizer_type = optimizer_type
        self.optimizer = None
        self.criterion = None
        self.embedding_layer = Embedding_layer(self.dataRecorder)
        self.embedding_layer = self.embedding_layer.to(self.device)
        self.early_stop = EarlyStop(patience)
        self.stop_epoch = None
        self.train_loss_records=[]
        self.val_loss_records=[]
        self.early_flag = False
        self.weight_decay = weight_decay
        self.record = record
        self.hidden_dim_list = []
        self.dropout_rate = 0
        self.normalization=None
        self.activation_func="ReLU"


    def initialize_optimizer(self):
        if self.optimizer_type == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == "Adam":
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == "AdamW":
            self.optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError("Invalid value for optimizer")
    def initialize_criterion(self):
        if self.criterion_type == "BCE":
            self.criterion = nn.BCELoss()  # 二元交叉熵损失
        else:
            raise ValueError("Invalid value for criterion")
        
    def intialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias:
                    nn.init.constant_(m.bias, 0)

                    

    def train_model(self):
        print(f"Training {self.model_name} Model...")
        if self.record:
            print("=================================================================")
            print(f"                      {self.model_name} Model Result            ")
            print("=================================================================")
            print("模型参数:")
            # print(f"数据集路径:{self.dataRecorder.parquet_path_linux}")
            print(f"批次大小: {self.dataRecorder.batch_size}, embedding维度:{self.dataRecorder.embedding_dim}, 优化器：{self.optimizer_type},学习率：{self.learning_rate}, 正则化系数：{self.weight_decay}")
            print(f"CPU/GPU: {self.device}, 损失函数：{self.criterion_type}")

        for epoch in range(self.num_epochs):
            self.train()
            sleep(0.1)
            train_loss = 0.0
            with tqdm(total=len(self.dataRecorder.train_loader), desc=f"Epoch {epoch+1}/{self.num_epochs} - batch {len(self.dataRecorder.train_loader)}", unit="batch") as pbar:
                for i, (inputs, labels) in enumerate(self.dataRecorder.train_loader):
                    inputs = inputs
                    labels = next(iter(labels.values()))
                    self.optimizer.zero_grad()
                    outputs = self.forward(inputs)
                    loss = self.criterion(outputs.squeeze(1), labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
                    self.optimizer.step()
                    train_loss += loss.item()
                    print(f"train_loss:{train_loss}")
                    pbar.update(1)

            avg_train_loss = train_loss / len(self.dataRecorder.train_loader)
            print(f'Training Loss: {avg_train_loss}')
            self.train_loss_records.append(avg_train_loss)

            self.eval()  
            val_labels = []
            val_preds = []
            with torch.no_grad():  
                with tqdm(total=len(self.dataRecorder.valid_loader), desc=f"Validation {epoch+1}/{self.num_epochs}", unit="batch") as pbar:
                    for inputs, labels in self.dataRecorder.valid_loader:
                        inputs = inputs
                        labels = next(iter(labels.values()))
                        outputs = self.forward(inputs)
                        val_labels.extend(labels.cpu().numpy().reshape(-1))
                        val_preds.extend(outputs.data.cpu().numpy().reshape(-1))
                        pbar.update(1)
            val_logloss = log_loss(val_labels, val_preds, eps=1e-7)
            val_auc= roc_auc_score(val_labels, val_preds)
            print(f'Validation LogLoss: {val_logloss:.6f}, AUC: {val_auc:.6f}')
            self.stop_epoch = epoch + 1
            if self.early_stop.should_early_stop(val_logloss):
                self.early_flag = True
                print(f"Early Stopping at Epoch {epoch + 1}")
                break


    def show_evaluation_results(self):

        train_auc_score, train_loss = self.evaluate_model(self.dataRecorder.train_loader)
        test_auc_score, test_loss = self.evaluate_model(self.dataRecorder.test_loader)

        print("=================================================================")
        print("Evaluating...")
        print("=================================================================")
        print(f"                      {self.model_name} Model Result                            ")
        print("=================================================================")
        print(f'Train Loss:{train_loss:.6f}, AUC: {train_auc_score:.6f}')
        print(f'Test Loss:{test_loss:.6f}, AUC: {test_auc_score:.6f}')
        print(f"Cost time {int(time.time()-self.start_time)} seconds")
        print("=================================================================")


        today_date = datetime.datetime.now().strftime('%Y%m%d')
        exact_time = datetime.datetime.now().strftime('%H%M%S')
        if self.record:
            root_dir = f'/home/yanghc03/recommended_system/RecommendSystem/result/{self.dataset_name}/{self.model_name}/{today_date}/{exact_time}'
            os.makedirs(root_dir, exist_ok=True)
            print(f"根目录: {root_dir} 创建成功")
            dict_file_path = os.path.join(root_dir, f"model_state_dict.pth")
            torch.save(self.state_dict(), dict_file_path)
            print(f"模型储存至{dict_file_path}")
        # 定义日志信息
        if self.early_flag:
            epoch_inf = f"{self.stop_epoch}/{self.num_epochs} (Early Stop)"
        else:
            epoch_inf = f"{self.stop_epoch}/{self.num_epochs}"
        if self.record:
            log_messages = [
                "=================================================================",
                f"                      {self.model_name} Model Result            ",
                "=================================================================",
                f"模型参数:",
                f"批次大小: {self.dataRecorder.batch_size}, embedding维度：{self.dataRecorder.embedding_dim}, 优化器：{self.optimizer_type}, 学习率：{self.learning_rate}, 正则化系数：{self.weight_decay}",
                f"CPU/GPU: {self.device}, 迭代次数：{epoch_inf}, 损失函数：{self.criterion_type}"]
            
            if self.model_name in ["DNN", "DeepFM"]:
                log_messages.append(f"DNN部分网络层数:{self.hidden_dim_list}, dropout比例:{self.dropout_rate}, 归一化方法:{self.normalization}, 激活函数：{self.activation_func}")
            log_messages.extend([
                f"数据集名称：{self.dataRecorder.dataset_name}, 训练集样本个数: {len(self.dataRecorder.train_loader)}, 特征个数：{self.dataRecorder.feature_num}",
                "=================================================================",
                "训练集损失记录: ",
                f"{self.train_loss_records}", 
                "验证集损失记录: ",
                f"{self.val_loss_records}",
                "=================================================================",
                f'Train AUC: {train_auc_score:.4f}',
                f'Test AUC: {test_auc_score:.4f}',
                f"Cost time {int(time.time()-self.start_time)} seconds",
                f"Model training completed on {today_date} {exact_time}.",
                f"Model储存到 {dict_file_path}",
                " "
            ])
            log_file_path = os.path.join(f'/home/yanghc03/recommended_system/RecommendSystem/result/{self.dataset_name}/{self.model_name}', "log.txt")

            with open(log_file_path, 'a+') as log_file:
                for message in log_messages:
                    log_file.write(message + '\n')  

            print(f"Log messages have been saved to {log_file_path}")

       

    def evaluate_model(self, data_loader):
        self.eval()  # 设置模型为评估模式
        all_labels = []
        all_preds = []
        with torch.no_grad():  # 在评估阶段不计算梯度
            with tqdm(total=len(data_loader), desc="Evaluating", unit="batch") as pbar:
                for inputs, labels in data_loader:
                    inputs = inputs
                    labels = next(iter(labels.values()))
                    outputs = self.forward(inputs)
                    all_labels.extend(labels.cpu().numpy().reshape(-1))
                    all_preds.extend(outputs.data.cpu().numpy().reshape(-1))
                    pbar.update(1)
        # print(all_labels)
        # print(all_preds)
        # raise KeyboardInterrupt
        average_loss = log_loss(all_labels, all_preds, eps=1e-7)
        auc_score = roc_auc_score(all_labels, all_preds)
        return auc_score, average_loss
        
    


class EarlyStop:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def should_early_stop(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False

    