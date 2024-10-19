import torch
from torch import nn

from models.utils.preprocess import import_data
from models.utils.train_and_test import train_model, evaluate_model
class FM(nn.Module):
    def __init__(self, num_feature, k_dim=4):
        super(FM, self).__init__()
        self.linear = nn.Linear(in_features=num_feature, out_features=1)
        self.v = nn.Parameter(torch.randn(num_feature, k_dim)) # n * k n个特征，k个隐维度

    def forward(self, x):
        # x: m * n m个样本，n个特征
        first_order_output = self.linear(x) # m * 1
        square_of_sum = torch.pow((x @ self.v), 2) # m * k
        sum_of_square = torch.pow(x, 2) @ torch.pow(self.v, 2) # m * k
        second_order_output = 0.5 * torch.sum((square_of_sum - sum_of_square), dim=1,keepdim=True) # m * 1
        output = first_order_output + second_order_output
        return torch.sigmoid(output)



if  __name__ == "__main__":
    train_loader, val_loader, test_loader, num_feature = import_data()
    model = FM(num_feature, k_dim=4)
    trained_model = train_model(model,train_loader,val_loader,num_epochs=10,learning_rate=0.01)


    print("=================================================================")
    print("                         Model Result                            ")
    print("=================================================================")
    train_loss, train_auc_score, train_f1 = evaluate_model(trained_model, test_loader)
    print(f'Train Loss: {train_loss:.4f}, Train AUC: {train_auc_score:.4f}, Train F1-Score: {train_f1:.4f}')
    test_loss, test_auc_score, test_f1 = evaluate_model(trained_model, train_loader)
    print(f'Test Loss: {test_loss:.4f}, Test AUC: {test_auc_score:.4f}, Test F1-Score: {test_f1:.4f}')
    print("=================================================================")
