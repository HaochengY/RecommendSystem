import torch
import torch.nn as nn
import torch.optim as optim
from jinja2 import optimizer
from sklearn.metrics import roc_auc_score, f1_score

from models.utils.preprocess import DataRecorder, Embedding_layer


class BaseModel(nn.Module):
    def __init__(self, model_name, num_epochs=5, learning_rate=0.01,
                 criterion_type="BCE", optimizer_type="SGD"):
        super(BaseModel, self).__init__()
        self.dataRecorder = DataRecorder()
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.criterion_type = criterion_type
        self.optimizer_type = optimizer_type
        self.optimizer = None
        self.criterion = None
        self.embedding_layer = Embedding_layer(self.dataRecorder)

    def initialize_optimizer(self):
        if self.optimizer_type == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "Adam":
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError("Invalid value for optimizer")

    def initialize_criterion(self):
        if self.criterion_type == "BCE":
            self.criterion = nn.BCELoss()  # 二元交叉熵损失
            # TODO: 加入其他损失函数
        else:
            raise ValueError("Invalid value for criterion")

    def train_model(self):
        print("Training...")
        for epoch in range(self.num_epochs):
            self.train()
            train_loss = 0.0
            for i, (inputs, labels) in enumerate(self.dataRecorder.train_loader):
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward(retain_graph=True)
                self.optimizer.step()  # 更新权重
                train_loss += loss.item()

            print(f'============ Epoch {epoch + 1} ================')
            print(f'Training Loss: {train_loss / len(self.dataRecorder.train_loader)}')
            # 在验证集上评估模型性能
            self.eval()  # 设置模型为评估模式
            val_loss = 0.0
            with torch.no_grad():  # 在评估阶段不计算梯度
                for inputs, labels in self.dataRecorder.val_loader:
                    outputs = self.forward(inputs)
                    outputs = torch.sigmoid(outputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
            print(f'Validation Loss: {val_loss / len(self.dataRecorder.val_loader)}')




    def show_evaluation_results(self):
        print("=================================================================")
        print("Evaluating...")
        print("=================================================================")
        print(f"                      {self.model_name} Model Result                            ")
        print("=================================================================")
        train_loss, train_auc_score, train_f1 = self.evaluate_model(self.dataRecorder.train_loader)
        print(f'Train Loss: {train_loss:.4f}, Train AUC: {train_auc_score:.4f}, Train F1-Score: {train_f1:.4f}')
        test_loss, test_auc_score, test_f1 = self.evaluate_model(self.dataRecorder.test_loader)
        print(f'Test Loss: {test_loss:.4f}, Test AUC: {test_auc_score:.4f}, Test F1-Score: {test_f1:.4f}')
        print("=================================================================")


    def evaluate_model(self, data_loader):
        self.eval()  # 设置模型为评估模式
        total_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():  # 在评估阶段不计算梯度
            for inputs, labels in data_loader:
                outputs = self.forward(inputs)
                outputs = torch.sigmoid(outputs)
                loss = nn.BCELoss()(outputs, labels)
                total_loss += loss.item()

                # 存储预测和标签用于后续计算 AUC 和 F1
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.detach().cpu().numpy())

        # 计算 AUC 和 F1 分数
        auc_score = roc_auc_score(all_labels, all_preds)
        preds_binary = [1 if p >= 0.5 else 0 for p in all_preds]
        f1 = f1_score(all_labels, preds_binary)
        loss = total_loss / len(data_loader)
        return loss, auc_score, f1

