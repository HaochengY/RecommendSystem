import torch
import torch.nn as nn

from models.utils.Layers import LinearRegression_layer
from models.utils.BaseModel import BaseModel


class LR(BaseModel):
    def __init__(self, optimizer_type="SGD", criterion_type="BCE"):
        super(LR, self).__init__(model_name="LR",
                                 optimizer_type=optimizer_type,
                                 criterion_type=criterion_type,
                                 )
        # 调用父类
        self.lr_layer = LinearRegression_layer(self.dataRecorder)
        self.initialize_criterion()
        self.initialize_optimizer()

    def forward(self, x):
        output = self.lr_layer(x)
        output = torch.sigmoid(output)  # 使用sigmoid激活函数
        return output


if __name__ == "__main__":
    model = LR()
    model.train_model()
    model.show_evaluation_results()
