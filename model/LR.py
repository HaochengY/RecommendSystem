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
                                 record=True)
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
    for emb_dim in [12, 16, 20]:
        for gamma in [0, 0.1, 0.2]:
            for lr in [0.001, 0.0001]:
                model = LR(embedding_dim=emb_dim, gamma=gamma, lr=lr)
                model.train_model()
                model.show_evaluation_results()
            
