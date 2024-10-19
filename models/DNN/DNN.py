import torch
from torch import nn

from models.utils.preprocess import import_data
from models.utils.train_and_test import train_model, evaluate_model


class MLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation_func, normalization):
        super(MLPBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  # 修改为不同的输入和输出维度

        if activation_func == "ReLU":
            self.activation_func = nn.ReLU()
        elif activation_func == "PReLU":
            self.activation_func = nn.PReLU

        # 归一化选择
        self.normalization = normalization
        if normalization == "bn":
            self.norm_layer = nn.BatchNorm1d(output_dim)
        elif normalization == "ln":
            self.norm_layer = nn.LayerNorm(output_dim)
        else:
            self.norm_layer = None

        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        out = self.fc(x)
        if self.norm_layer is not None:  # 如果有归一化层，进行归一化
            out = self.norm_layer(out)
        out = self.activation_func(out)
        out = self.dropout(out)
        return out

class DNN(nn.Module):
    def __init__(self, hidden_dim_list, activation_func="ReLU", normalization="bn"):
        super(DNN, self).__init__()
        layers = []  # DNN 可以设置为多个隐藏层
        # 根据 layerlist 动态创建 MLPBlock
        for i in range(len(hidden_dim_list) - 1):
            input_dim = hidden_dim_list[i]
            output_dim = hidden_dim_list[i + 1]
            layers.append(MLPBlock(input_dim, output_dim, activation_func, normalization))
        self.dnn = nn.Sequential(*layers) # 将层打包成网络
        """
        nn.Sequential 是 PyTorch 中的一个模块，它允许将多个神经网络层按照它们的顺序组合在一起。
        通过 nn.Sequential，我们可以更加方便地定义一个前馈神经网络，
        而不需要在 forward() 方法中手动定义每一层的计算顺序。
        """

    def forward(self, x):
        output = self.dnn(x)
        return torch.sigmoid(output)


if  __name__ == "__main__":
    train_loader, val_loader, test_loader, num_feature = import_data()
    layer_list = [num_feature]
    layer_list.extend([512, 256])
    layer_list.append(1)
    model = DNN(layer_list)
    trained_model = train_model(model,train_loader,val_loader,num_epochs=10, learning_rate=0.1)

    print("=================================================================")
    print("                         Model Result                            ")
    print("=================================================================")
    train_loss, train_auc_score, train_f1 = evaluate_model(trained_model, test_loader)
    print(f'Train Loss: {train_loss:.4f}, Train AUC: {train_auc_score:.4f}, Train F1-Score: {train_f1:.4f}')
    test_loss, test_auc_score, test_f1 = evaluate_model(trained_model, train_loader)
    print(f'Test Loss: {test_loss:.4f}, Test AUC: {test_auc_score:.4f}, Test F1-Score: {test_f1:.4f}')
    print("=================================================================")



