import torch
import torch.nn as nn
from models.utils.Layers import LinearRegression_layer
from models.utils.BaseModel import BaseModel


class LR(BaseModel):
    def __init__(self, num_epochs=20, criterion_type="BCE", batch_size=256, learning_rate=0.01,
                  weight_decay=0.01, record=True,down_sample=5):
        super(LR, self).__init__(model_name="LR",
                                 num_epochs=num_epochs,
                                 criterion_type=criterion_type,
                                 learning_rate=learning_rate,
                                 batch_size=batch_size,
                                 weight_decay=weight_decay,
                                 record=record,
                                 down_sample=down_sample)
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
        model = LR(learning_rate=0.001, weight_decay=0.2,down_sample=10)
        model.train_model()
        model.show_evaluation_results()
        
