import torch
import torch.nn as nn

# from RecommendSystem.model.utils.BaseModel import BaseModel
# from RecommendSystem.model.utils.Layers import LinearRegression_layer
# from RecommendSystem.model.utils.Preprocess import DataRecorder


from model.utils.BaseModel import BaseModel
from model.utils.Layers import LinearRegression_layer
from model.utils.Preprocess import DataRecorder

class LR(BaseModel):
    def __init__(self,dataRecorder, gamma,lr=0.001):
        super(LR, self).__init__(model_name="LR",
                                 num_epochs=30,
                                 weight_decay=gamma,
                                 learning_rate=lr,
                                 record=True,
                                 dataRecorder=dataRecorder)
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
    dataRecorder = DataRecorder(embedding_dim=12, batch_size=4096)
    for gamma in [1e-5, 5e-5, 1e-6, 5e-6, 1e-7, 5e-7, 1e-8]:
        model = LR(dataRecorder, gamma=gamma)
        model.train_model()
        model.show_evaluation_results()

