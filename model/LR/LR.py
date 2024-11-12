import torch
import torch.nn as nn

from model.utils.BaseModel import BaseModel
from model.utils.Layers import LinearRegression_layer


class LR(BaseModel):
    def __init__(self,embedding_dim=12,gamma=0,lr=0.001):
        super(LR, self).__init__(model_name="LR",
                                 embedding_dim=embedding_dim,
                                 num_epochs=30,
                                 weight_decay=gamma,
                                 learning_rate=lr,
                                 record=False)
        # 调用父类
        self.lr_layer = LinearRegression_layer(self.dataRecorder).to(self.device)
        self.intialize_weights()
        self.initialize_criterion()
        self.initialize_optimizer()

    def forward(self, x):
        x = x.to(self.device)
        output = self.lr_layer(x)
        return output


if __name__ == "__main__":
    for gamma in [1e-5, 5e-5, 1e-6, 5e-6, 1e-7, 5e-7, 1e-8]:
            model = LR(gamma=gamma)
            model.train_model()
            model.show_evaluation_results()
        
