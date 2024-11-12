import torch
from torch import nn

from models.utils.Layers import MLPBlock, LinearRegression_layer
from models.utils.BaseModel import BaseModel


class AFM(BaseModel):
    def __init__(self, hidden_dim_list, attention_dim=8,
                 optimizer_type="SGD", criterion_type="BCE",
                 activation_func="ReLU", normalization=None):
        super(AFM, self).__init__(model_name="AFM",
                                  optimizer_type=optimizer_type,
                                  criterion_type=criterion_type)
        self.emb_dim = self.dataRecorder.embedding_dim
        self.feature_num = self.dataRecorder.feature_num
        self.attention_dim = attention_dim

        self.attention = nn.Sequential(nn.Linear(self.emb_dim, self.attention_dim),
                                       # (bs, f*(f-1)/2, emb_dim) -> (bs, f*(f-1)/2, attention_dim)
                                       nn.ReLU(),
                                       nn.Linear(attention_dim, 1, bias=False),
                                       # (bs, f * (f - 1) / 2, attention_dim) -> (bs, f * (f - 1) / 2, 1)
                                       nn.Softmax(dim=1)).to(self.device)   # (bs, f * (f - 1) / 2, 1) 转化为概率，表示每个的重要性

        self.p = nn.Linear(self.emb_dim, 1).to(self.device)
        self.linear = LinearRegression_layer(self.dataRecorder).to(self.device)

        self.initialize_criterion()
        self.initialize_optimizer()

    def forward(self, x):
        x = x.to(self.device)
        lr_out = self.linear(x)
        emb_out = self.embedding_layer(x)
        pair_wise_output = self.pair_wise_interaction_layer(emb_out)
        # (bs, (fea_num * (fea_num-1)/2 , emb_dim)
        attention_score = self.attention(pair_wise_output)
        # (bs, (fea_num * (fea_num-1)/2 , 1)
        attention_sum = torch.sum(attention_score * pair_wise_output, dim=1)
        # 这里其实第三个维度是不匹配的，但是还能乘是由于广播机制
        afm_output = self.p(attention_sum)

        output = lr_out + afm_output
        return torch.sigmoid(output)

    def pair_wise_interaction_layer(self, emb_out):
        bs = emb_out.size()[0]
        emb_out = emb_out.view(bs, -1)
        interactions = []
        for i in range(self.feature_num):
            for j in range(i + 1, self.feature_num):
                interactions.append(emb_out[:, i * self.emb_dim:(i + 1) * self.emb_dim]
                                    * emb_out[:, j * self.emb_dim:(j + 1) * self.emb_dim])
                # (bs, emb_dim)
        return torch.stack(interactions, dim=1)


if __name__ == "__main__":
    model = AFM(hidden_dim_list=[1024, 512, 256])
    model.train_model()
    model.show_evaluation_results()


