import torch
import torch.nn as nn
from models.utils.BaseModel import BaseModel
from models.utils.Layers import LinearRegression_layer


class FFM(BaseModel):
    def __init__(self,learning_rate, weight_decay,embedding_dim,down_sample):
        super(FFM, self).__init__(model_name="FFM",
                                 embedding_dim=embedding_dim,
                                 optimizer_type="AdamW",
                                 criterion_type="BCE",
                                 learning_rate=learning_rate,
                                 weight_decay=weight_decay,
                                 down_sample=down_sample)
        self.emb_dim = self.dataRecorder.embedding_dim
        self.feature_num = self.dataRecorder.feature_num
        self.field_num = self.feature_num
        self.linear = LinearRegression_layer(self.dataRecorder).to(self.device)

        self.embedding_dict = self.dataRecorder.embedding_dict
        for feature_name, embedding_layer in self.embedding_dict.items():
            self.embedding_dict.update({feature_name: [embedding_layer.to(self.device) for _ in range(self.field_num)]})
        # 在FFM中重新定义embeddingLayer，因为他要生成field
        self.feature_name_list = self.dataRecorder.feature_name_list
        self.field_index = torch.arange(0, self.field_num).to(self.device)  # 每个特征所属的field
        # 我这里就设置了每个都有一个 第i个特征的field就是第i个filed，也可以自己定义

        self.initialize_criterion()
        self.initialize_optimizer()

    def forward(self, x):
        x = x.to(self.device)

        linear_part = self.linear(x)
        # 二阶交互部分 (FFM)
        second_part = self.ffm_interaction(x)
        output = linear_part + second_part
        # (batch_size,1) + (batch_size,1) = (batch_size,1)
        return torch.sigmoid(output)


    def ffm_interaction(self, x):
        batch_size = x.size(0)
        interaction_part = torch.zeros(batch_size, 1, device=self.device)

        for i in range(self.feature_num):
            for j in range(i + 1, self.feature_num):  # 交互特征
                feature_i = self.feature_name_list[int(self.field_index[i])]
                feature_j = self.feature_name_list[int(self.field_index[j])]
                try:
                    vi_fj = self.embedding_dict[feature_i][j](x[:, i])  # 特征 i 在字段 j 下的隐向量

                    vj_fi = self.embedding_dict[feature_j][i](x[:, j])  # 特征 j 在字段 i 下的隐向量
                except:
                    print(i, j)
                    raise KeyError
                # 每个v的维度：(bs, emb_dim)
                interaction = torch.sum(vi_fj * vj_fi, dim=-1, keepdim=True)
                # interaction (bs,)
                interaction_part += interaction
        return interaction_part
if __name__ == "__main__":
    for weight_decay in [0.2,0.3]:
        for embedding_dim in [8,12, 16, 20]:
            model = FFM(learning_rate=0.001, weight_decay=weight_decay,embedding_dim = embedding_dim,down_sample=10)
            model.train_model()
            model.show_evaluation_results()