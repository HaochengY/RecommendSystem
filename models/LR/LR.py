import torch
import torch.nn as nn


from models.utils.preprocess import import_data
from models.utils.train_and_test import train_model, evaluate_model


class LRModel(nn.Module):
    def __init__(self, num_features):
        super(LRModel, self).__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))  # 使用sigmoid激活函数
        return y_pred




if __name__ == "__main__":
    train_loader, val_loader, test_loader, num_feature = import_data()
    lr_model = LRModel(num_feature)
    trained_model = train_model(lr_model,train_loader,val_loader,num_epochs=30,learning_rate=0.01)
    # 在测试集上评估模型性能
    test_loss, test_accuracy = evaluate_model(trained_model, test_loader)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy:.4f}')
