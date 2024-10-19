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

    print("=================================================================")
    print("                         Model Result                            ")
    print("=================================================================")
    train_loss, train_auc_score, train_f1 = evaluate_model(trained_model, test_loader)
    print(f'Train Loss: {train_loss:.4f}, Train AUC: {train_auc_score:.4f}, Train F1-Score: {train_f1:.4f}')
    test_loss, test_auc_score, test_f1 = evaluate_model(trained_model, train_loader)
    print(f'Test Loss: {test_loss:.4f}, Test AUC: {test_auc_score:.4f}, Test F1-Score: {test_f1:.4f}')
    print("=================================================================")
