import torch
from torch import nn

from models.utils.preprocess import import_data
from models.utils.train_and_test import train_model, evaluate_model

class DNN(nn.Module):
    def __init__(self, feature_dim, hidden_dim=4):
        super(DNN, self).__init__()


    def forward(self, x):


if  __name__ == "__main__":
    train_loader, val_loader, test_loader, num_feature = import_data()
    model = DNN(num_feature)
    trained_model = train_model(model,train_loader,val_loader,num_epochs=10, learning_rate=0.01)
    # 在测试集上评估模型性能
    test_loss, test_accuracy = evaluate_model(trained_model, test_loader)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy:.4f}')

