import torch
import torch.nn as nn


from model.utils.BaseModel import BaseModel
from model.utils.Layers import LinearRegression_layer
from model.utils.PreProcess import DataRecorder

class LR(BaseModel):
    def __init__(self,dataRecorder, gamma,lr):
        super(LR, self).__init__(model_name="LR",
                                 num_epochs=5,
                                 weight_decay=gamma,
                                 learning_rate=lr,
                                 record=True,
                                 dataRecorder=dataRecorder)
        # 调用父类
        self.lr_layer = LinearRegression_layer(self.dataRecorder).to(self.device)
        # self.intialize_weights()
        self.initialize_criterion()
        self.initialize_optimizer()

    def forward(self, x):
        output = self.lr_layer(x)
        output = torch.sigmoid(output)
        return output


if __name__ == "__main__":
    dataRecorder = DataRecorder(embedding_dim=4, batch_size=128)
    model = LR(dataRecorder, gamma=0, lr=1e-3)
    model.train_model()
    model.show_evaluation_results()

