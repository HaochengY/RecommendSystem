import logging
import torch
import torch.nn as nn


from model.utils.BaseModel import BaseModel
from model.utils.Layers import LinearRegression_layer
from model.utils.PreProcess import DataRecorder

class LR(BaseModel):
    def __init__(self, dataRecorder,lr, gpu, regular=None):
        super(LR, self).__init__(model_name="LR",
                                 num_epochs=10,
                                 learning_rate=lr,
                                 record=True,
                                 dataRecorder=dataRecorder,
                                 gpu = gpu,
                                 emb_regular=regular,
                                 net_regular=regular)
        # 调用父类
        self.lr_layer = LinearRegression_layer(self.dataRecorder).to(self.device)
        # self.intialize_weights()
        self.initialize_criterion()
        self.initialize_optimizer()
        self.reset_parameters()
        self.model_to_device()


    def forward(self, x):
        x = self.features_to_divice(x)
        output = self.lr_layer(x)
        output = torch.sigmoid(output)
        return output


if __name__ == "__main__":
    logger = logging.getLogger('my_logger')
    gpu = 0
    dataRecorder = DataRecorder(embedding_dim=1, batch_size=4096, gpu=gpu)
    model = LR(dataRecorder, regular=5e-07, lr=0.001, gpu=gpu)
    model.fit()
    logger.info("======== Validation evaluation ========")
    model.evaluate(dataRecorder.valid_loader)
    logger.info("======== Test evaluation ========")
    model.evaluate(dataRecorder.test_loader)
    logger.info("\n\n\n\n\n")



    

