from collections import defaultdict
import datetime
import os
from time import sleep
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


from RecommendSystem.model.utils.Layers import Embedding_layer
from RecommendSystem.model.utils.Preprocess import DataRecorder


class BaseModel(nn.Module):
    def __init__(self, 
                 model_name, 
                 dataset_name="Criteo_x1",
                 batch_size=256,
                 patience=5, 
                 embedding_dim=12, 
                 num_epochs=30, 
                 learning_rate=0.01,
                 criterion_type="BCE", 
                 optimizer_type="AdamW",
                 weight_decay=0, 
                 record=True):
        super(BaseModel, self).__init__()
        self.start_time = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Calculating with {self.device}")
        self.dataRecorder = DataRecorder(dataset_name=dataset_name,
                                        embedding_dim=embedding_dim, 
                                        batch_size=batch_size)
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
            self.criterion = nn.BCEWithLogitsLoss()  # 二元交叉熵损失
            # TODO: 加入其他损失函数
        else:
            raise ValueError("Invalid value for criterion")
        
    def intialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias:
                    nn.init.constant_(m.bias, 0)

    def train_model(self):
        print(f"Training {self.model_name} Model...")
        if self.record:
            print("=================================================================")
            print(f"                      {self.model_name} Model Result            ")
            print("=================================================================")
            print("模型参数:")
            print(f"数据集路径:{self.dataRecorder.parquet_path_linux}")
            print(f"批次大小: {self.dataRecorder.batch_size}, embedding维度:{self.dataRecorder.embedding_dim}, 优化器：{self.optimizer_type},学习率：{self.learning_rate}, 正则化系数：{self.weight_decay}")
            print(f"CPU/GPU: {self.device}, 损失函数：{self.criterion_type}")

            
    
        if self.model_name in ["DNN", "DeepFM"]:
            print(f"DNN部分网络层数:{self.hidden_dim_list}, dropout比例:{self.dropout_rate}, 归一化方法:{self.normalization}")
            print(f"数据集名称：{self.dataRecorder.dataset_name}, 样本个数: {self.dataRecorder.num_sample}, 特征个数：{self.dataRecorder.feature_num}")
            print("=================================================================")



        scaler = GradScaler()
        for epoch in range(self.num_epochs):
            self.train()
            sleep(0.1)
            train_loss = 0.0
            with tqdm(total=len(self.dataRecorder.train_loader),desc=f"Epoch {epoch+1}/{self.num_epochs}", unit="batch") as pbar:
                for i, (inputs, labels) in enumerate(self.dataRecorder.train_loader):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    self.optimizer.zero_grad()
                    with autocast():
                        outputs = self.forward(inputs)
                        loss = self.criterion(outputs, labels)
  
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)  # 更新权重
                    scaler.update()
                    train_loss += loss.item()
                    pbar.update(1)

            avg_train_loss = train_loss / len(self.dataRecorder.train_loader)
            print(f'Training Loss: {avg_train_loss}')
            self.train_loss_records.append(avg_train_loss)
            # 在验证集上评估模型性能
            self.eval()  # 设置模型为评估模式
            val_loss = 0.0
            with torch.no_grad():  # 在评估阶段不计算梯度
                with tqdm(total=len(self.dataRecorder.val_loader),desc=f"Validation {epoch+1}/{self.num_epochs}", unit="batch") as pbar:
                    for inputs, labels in self.dataRecorder.val_loader:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        outputs = self.forward(inputs)
                        outputs = torch.sigmoid(outputs)
                        loss = self.criterion(outputs, labels)
                        val_loss += loss.item()
                        # pbar.set_postfix({"Val Loss": loss.item()})
                        pbar.update(1)
            average_val_loss = val_loss / len(self.dataRecorder.val_loader)
            print(f'Validation Loss: {average_val_loss}')
            self.val_loss_records.append(average_val_loss)

            self.stop_epoch = epoch + 1
            if self.early_stop.should_early_stop(val_loss):
                self.early_flag = True
                print(f"Early Stopping at Epoch {epoch + 1}")
                break

    def show_evaluation_results(self):

        train_auc_score = self.evaluate_model(self.dataRecorder.train_loader)
        test_auc_score = self.evaluate_model(self.dataRecorder.test_loader)

        print("=================================================================")
        print("Evaluating...")
        print("=================================================================")
        print(f"                      {self.model_name} Model Result                            ")
        print("=================================================================")
        print(f'Train AUC: {train_auc_score:.4f}')
        print(f'Test AUC: {test_auc_score:.4f}')
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
                f"负采样比例:{self.dataRecorder.down_sample}%",
                f"数据集路径:{self.dataRecorder.parquet_path_linux}",
                f"批次大小: {self.dataRecorder.batch_size}, embedding维度：{self.dataRecorder.embedding_dim}, 优化器：{self.optimizer_type}, 学习率：{self.learning_rate}, 正则化系数：{self.weight_decay}",
                f"CPU/GPU: {self.device}, 迭代次数：{epoch_inf}, 损失函数：{self.criterion_type}"]
            
            if self.model_name in ["DNN", "DeepFM"]:
                log_messages.append(f"DNN部分网络层数:{self.hidden_dim_list}, dropout比例:{self.dropout_rate}, 归一化方法:{self.normalization}, 激活函数：{self.activation_func}")
            log_messages.extend([
                f"数据集名称：{self.dataRecorder.dataset_name}, 样本个数: {self.dataRecorder.num_sample}, 特征个数：{self.dataRecorder.feature_num}",
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
            log_file_path = os.path.join(f'/home/yanghc03/python_from_github/RecommendSystem/result/{self.model_name}', "log.txt")

            with open(log_file_path, 'a+') as log_file:
                for message in log_messages:
                    log_file.write(message + '\n')  

            print(f"Log messages have been saved to {log_file_path}")

       

    def evaluate_model(self, data_loader):
        self.eval()  # 设置模型为评估模式
        total_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():  # 在评估阶段不计算梯度
            with tqdm(total=len(data_loader),desc="Evaluating", unit="batch") as pbar:
            
                for inputs, labels in data_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.forward(inputs)
                    outputs = torch.sigmoid(outputs)
                    loss = nn.BCELoss()(outputs, labels)
                    total_loss += loss.item()

                    # 存储预测和标签用于后续计算 AUC
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(outputs.detach().cpu().numpy())
                    pbar.update(1)
        # 计算 AUC 和 F1 分数
        auc_score = roc_auc_score(all_labels, all_preds)
        # gauc_score = BaseModel.calculate_gauc(all_labels, all_preds, all_sponsor_ids)
        return auc_score
    
    @staticmethod
    def calculate_gauc(labels, preds, sponsor_ids):
        """
        计算加权GAUC
        """
        gauc_total = 0.0
        weight_sum = 0
        sponsor_groups = defaultdict(list)

        # 将数据按sponsor_id进行分组
        for label, pred, sponsor_id in zip(labels, preds, sponsor_ids):
            sponsor_id = int(sponsor_id)
            label = int(label)
            sponsor_groups[sponsor_id].append((label, pred))

        # 计算每个sponsor_id组的AUC，并按点击数加权
        for sponsor_id, group in sponsor_groups.items():
            group_labels, group_preds = zip(*group)
            if len(set(group_labels)) > 1:  # 确保AUC可计算
                auc = roc_auc_score(group_labels, group_preds)
                weight = sum(group_labels)  # 以点击数作为权重
                gauc_total += auc * weight
                weight_sum += weight

        gauc = gauc_total / weight_sum if weight_sum > 0 else 0.0
        return gauc




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

    