import torch
import torch.nn as nn
import torch.optim as optim



def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    criterion = nn.BCELoss()  # 二元交叉熵损失
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # 随机梯度下降优化器
    for epoch in range(num_epochs):
            model.train()  # 设置模型为训练模式
            train_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()  # 清空梯度
                outputs = model(inputs)  # 前向传播
                loss = criterion(outputs, labels)  # 计算损失
                loss.backward(retain_graph=True)  # 反向传播
                optimizer.step()  # 更新权重
                train_loss += loss.item()

            print(f'Epoch {epoch+1}, Loss: {train_loss/len(train_loader)}')

            # 在验证集上评估模型性能
            model.eval()  # 设置模型为评估模式
            val_loss = 0.0
            correct = 0
            with torch.no_grad():  # 在评估阶段不计算梯度
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    # 将输出转换为预测类别
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted.squeeze() == labels).sum().item()

            accuracy = correct / len(val_loader.dataset)
            print(f'Validation Loss: {val_loss/len(val_loader)}, Accuracy: {accuracy:.4f}')
    return model

def evaluate_model(model, data_loader):
    model.eval()  # 设置模型为评估模式
    total_loss = 0.0
    correct = 0
    total_samples = 0
    with torch.no_grad():  # 在评估阶段不计算梯度
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = nn.BCELoss()(outputs, labels)
            total_loss += loss.item()
            # 将输出转换为预测类别
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted.squeeze() == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = correct / total_samples
    loss = total_loss / len(data_loader)
    return loss, accuracy