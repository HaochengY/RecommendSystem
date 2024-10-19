import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score


def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    criterion = nn.BCELoss()  # 二元交叉熵损失
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # 随机梯度下降优化器
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)
            optimizer.step()  # 更新权重
            train_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {train_loss / len(train_loader)}')

        # 在验证集上评估模型性能
        model.eval()  # 设置模型为评估模式
        val_loss = 0.0
        with torch.no_grad():  # 在评估阶段不计算梯度
            for inputs, labels in val_loader:
                outputs = model(inputs)
                outputs = torch.sigmoid(outputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()


        print(f'Validation Loss: {val_loss / len(val_loader)}')
    return model


def evaluate_model(model, data_loader):
    model.eval()  # 设置模型为评估模式
    total_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():  # 在评估阶段不计算梯度
        for inputs, labels in data_loader:
            outputs = model(inputs)
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
