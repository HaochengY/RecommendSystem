from collections import defaultdict
import datetime
import os
from time import sleep
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import log_loss, roc_auc_score
from tqdm import tqdm
import logging
from model.utils.Layers import Embedding_layer
from model.utils.utils import get_device
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self, 
                 model_name,
                 dataRecorder,
                 dataset_name="Criteo_x1",
                 patience=2,
                 num_epochs=30, 
                 learning_rate=0.001,
                 criterion_type="BCE", 
                 optimizer_type="Adam",
                 weight_decay=0,
                 record=False,
                 gpu=-1,
                 emb_regular=None,
                 net_regular=None):
        super(BaseModel, self).__init__()
        self.start_time = time.time()
        self.logger = logging.getLogger('my_logger')

        self.device = get_device(gpu)
        self.dataRecorder = dataRecorder
        self.dataset_name = dataset_name
        self.input_dim = self.dataRecorder.input_dim
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.criterion_type = criterion_type
        self.optimizer_type = optimizer_type
        self.optimizer = None
        self.criterion = None
        self.embedding_layer = Embedding_layer(self.dataRecorder)
        self.embedding_layer = self.embedding_layer
        self.checkpoint = os.path.join("/home/yanghc03/recommended_system/RecommendSystem/result",
                                       self.dataset_name,
                                       datetime.datetime.now().strftime('%Y%m%d'),
                                       datetime.datetime.now().strftime('%H%M%S'))
        self.stop_training = False
        self.record = record
        self.hidden_dim_list = []
        self.dropout_rate = 0
        self.normalization=None
        self.activation_func="ReLU"
        self.patience = patience
        self.reduce_lr_on_plateau = True
        self.emb_regular = emb_regular
        self.net_regular = net_regular

    def initialize_optimizer(self):
        if self.optimizer_type == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "Adam":
            self.optimizer = getattr(torch.optim, "Adam")(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "AdamW":
            self.optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError("Invalid value for optimizer")
    def initialize_criterion(self):
        if self.criterion_type == "BCE":
            self.criterion = getattr(F, 'binary_cross_entropy')  # 二元交叉熵损失
        else:
            raise ValueError("Invalid value for criterion")
        
    def reset_parameters(self):
        def reset_default_params(m):
            # initialize nn.Linear/nn.Conv1d layers by default
            if type(m) in [nn.Linear, nn.Conv1d]:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        def reset_custom_params(m):
            # initialize layers with customized reset_parameters
            if hasattr(m, 'reset_custom_params'):
                m.reset_custom_params()
        self.apply(reset_default_params)
        self.apply(reset_custom_params)
                    
    def model_to_device(self):
        self.to(device=self.device)

        

    def fit(self):
        if self.record:
            self.logger.info(f"模型:{self.model_name}, 批次大小: {self.dataRecorder.batch_size}, embedding维度:{self.dataRecorder.embedding_dim}, 优化器: {self.optimizer_type}")
            self.logger.info(f"学习率:{self.learning_rate}, net正则化系数:{self.net_regular}, emb正则化系数:{self.emb_regular},CPU/GPU: {self.device}, 损失函数: {self.criterion_type}")
            self.logger.info(f"训练样本个数:{self.dataRecorder.train_sample_num}, 验证样本个数:{self.dataRecorder.valid_sample_num}")
            self.logger.info(f"开始训练:{len(self.dataRecorder.train_loader)} batches/epoch")
            self.logger.info(f"=========== Epoch = 1 start ===========")
        self.best_auc = np.float("inf")
        for epoch in range(self.num_epochs):
            self.epoch_index = epoch
            self.train_epoch()
            if self.stop_training:
                break
            else:
                self.logger.info(f"=========== Epoch = {epoch + 1} end ===========")
        self.logger.info("训练结束.")
        self.logger.info("加载最优模型: {}".format(os.path.join(self.checkpoint,f"{self.model_name}.model")))
        self.load_weights(os.path.join(self.checkpoint,f"{self.model_name}.model"))


    def load_weights(self, checkpoint):
        self.to(self.device)
        state_dict = torch.load(checkpoint, map_location="cpu")
        self.load_state_dict(state_dict)




    def train_epoch(self):
        self.train()
        sleep(0.1)
        train_loss = 0.0
        with tqdm(total=len(self.dataRecorder.train_loader), desc=f"Epoch {self.epoch_index+1}/{self.num_epochs} - batch {len(self.dataRecorder.train_loader)}", unit="batch") as pbar:
            for i, (inputs, labels) in enumerate(self.dataRecorder.train_loader):
                loss = self.train_step(inputs, labels)
                train_loss += loss.item()
                pbar.update(1)
                if self.stop_training:
                    break
        self.logger.info(f'Train Loss: {train_loss / len(self.dataRecorder.train_loader):.6f}')
        train_loss = 0.0
        self.eval_step()

    def train_step(self, inputs, labels):
        self.optimizer.zero_grad()
        labels = self.labels_to_device(labels)
        outputs = self.forward(inputs)
        loss = self.criterion(outputs.squeeze(1), labels.float())
        loss += self.regularization_loss()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=10)
        self.optimizer.step()
        return loss
    
    def eval_step(self):
        self.logger.info(f'Evaluation @epoch {self.epoch_index+1} - batch {len(self.dataRecorder.train_loader)}: ')
        val_auc = self.evaluate(self.dataRecorder.valid_loader)
        self.checkpoint_and_earlystop(val_auc)
        self.train()
        
    def evaluate(self, data_loader):
        self.eval()  
        with torch.no_grad():  
            val_labels = []
            val_preds = []
            with tqdm(total=len(data_loader), desc=f"Evaluating...", unit="batch") as pbar:
                for inputs, labels in data_loader:
                    labels = self.labels_to_device(labels)
                    outputs = self.forward(inputs)
                    val_labels.extend(labels.cpu().numpy().reshape(-1))
                    val_preds.extend(outputs.data.cpu().numpy().reshape(-1))
                    pbar.update(1)
                val_logloss = log_loss(val_labels, val_preds, eps=1e-7)
                val_auc= roc_auc_score(val_labels, val_preds)
                self.logger.info(f'AUC: {val_auc:.6f} - Logloss: {val_logloss:.6f}')
                return val_auc

    def checkpoint_and_earlystop(self, val_auc, delta=1e-6):
        if val_auc < self.best_auc - delta:
            self.best_auc = val_auc
            self.counter = 0
            os.makedirs(self.checkpoint, exist_ok=True)
            self.save_weights()
        else:
            self.counter += 1
            if self.reduce_lr_on_plateau:
                current_lr = self.lr_decay()
                self.logger.info("降低学习率为: {:.6f}".format(current_lr))

            if self.counter >= self.patience:
                self.stop_training = True
                self.logger.info(f"======== Early stop at Epoch = {self.epoch_index} =========")

    def lr_decay(self, factor=0.1, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            reduced_lr = max(param_group["lr"] * factor, min_lr)
            param_group["lr"] = reduced_lr
        return reduced_lr

    def save_weights(self):
        dict_file_path = os.path.join(self.checkpoint, f"{self.model_name}.model")
        torch.save(self.state_dict(), dict_file_path)
        self.logger.info(f"Best Model with AUC = {self.best_auc:.6f} 储存至{dict_file_path}")

    
    
    def features_to_divice(self, inputs):
        X_dict = dict()
        for feature, _ in inputs.items():
            X_dict[feature] = inputs[feature].to(self.device)
        return X_dict
        
    def labels_to_device(self, labels):
        return next(iter(labels.values())).to(self.device)
    

    def regularization_loss(self):
        reg_term = 0
        if self.net_regular or self.emb_regular:
            emb_reg = [(2, self.emb_regular)]
            net_reg = [(2, self.net_regular)]
            for _, module in self.named_modules():
                for p_name, param in module.named_parameters():
                    if param.requires_grad:
                        if p_name in ["weight", "bias"]:
                            if type(module) == nn.Embedding:
                                if self.emb_regular:
                                    for emb_p, emb_lambda in emb_reg:
                                        reg_term += (emb_lambda / emb_p) * torch.norm(param, emb_p) ** emb_p
                            else:
                                if self.net_regular:
                                    for net_p, net_lambda in net_reg:
                                        reg_term += (net_lambda / net_p) * torch.norm(param, net_p) ** net_p
        return reg_term






