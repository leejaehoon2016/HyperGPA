import sys, json, os; sys.path.append("..")
import torch
from hyperfunction.hyper_model import HyperModel
from utils.util import mkdir, save_dict, save_score_model, fix_random_seed, cal_num_param, get_logger, IndexRandomGenerator
from utils.cal_score import RMSELoss, cal_score
from torch.utils.data import TensorDataset, DataLoader
from utils.get_data import get_data
from tensorboardX import SummaryWriter

import os
import time
os.environ['TZ'] =  'Asia/Seoul'
time.tzset()

class Trainer:
    def __init__(self, lstm_hparam, emb_process_hparam, lstm_process_hparam, rest_process_hparam, hyper_model_hparam, data_hparam, r_hparam):
        self.lstm_hparam, self.emb_process_hparam, self.lstm_process_hparam, self.rest_process_hparam, self.hyper_model_hparam, self.data_hparam, self.r_hparam = \
            lstm_hparam, emb_process_hparam, lstm_process_hparam, rest_process_hparam, hyper_model_hparam, data_hparam, r_hparam
        file_path = mkdir(r_hparam["save_loc"], data_hparam["d_folder"], data_hparam["d_name"], r_hparam["test_name"])
        for k,v in dict(vars(self)).items():
            save_dict(f"{file_path}/{k}.json",v)
        self.file_path = file_path
        fix_random_seed(r_hparam["seed"])

        if self.r_hparam["GPU_NUM"] < 0:
            self.device = torch.device('cpu')
        elif self.r_hparam["GPU_NUM"] == 0:
            self.device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(f'cuda:{self.r_hparam["GPU_NUM"]}' if torch.cuda.is_available() else 'cpu')
            
        self.writer = SummaryWriter(f"{self.file_path}/runs")
        self.logger = get_logger(self.file_path)

    def train(self):
        (train_data, val_data, test_data, pathx_dim_per_state, train_lstm_bacth_num, val_lstm_bacth_num, test_lstm_bacth_num, \
          front_num_states, back_num_states, state_edge_list, final_index, concat_info_front2back), self.prob_type = get_data(**self.data_hparam)
        out_dim = test_data[1][1].shape[-1]
        x_path_train_data = torch.stack([i[0] for i in train_data])
        final_index = torch.tensor(final_index, dtype=torch.long)
        num_batch = len(x_path_train_data)
        x_batch_idx = torch.arange(0,num_batch)
        self.x_path_train_data = DataLoader(TensorDataset(x_path_train_data, x_batch_idx, final_index), shuffle=True, batch_size=self.r_hparam["path_batch_size"])
        self.x_path_val_data     = val_data[0].unsqueeze(0)
        self.x_path_val_data_len = torch.tensor([self.x_path_val_data.size(2)]).long()
        
        self.x_path_test_data     = test_data[0].unsqueeze(0)
        self.x_path_test_data_len = torch.tensor([self.x_path_test_data.size(2)]).long()
        
        self.train_lstm_data = [[i for i in data[1]] for data in train_data]
        max_num_lstm_data = max([i[0].size(0) for i in self.train_lstm_data]) * back_num_states * self.r_hparam["path_batch_size"]
        num_iter_per_epoch = max_num_lstm_data // self.r_hparam["max_batch_size"] + (max_num_lstm_data % self.r_hparam["max_batch_size"] != 0)
        self.train_lstm_indices_dataloader = [[IndexRandomGenerator(train_lstm_bacth_num[b][s], num_iter_per_epoch) for s in range(back_num_states)] for b in range(num_batch)]

        self.val_lstm_data  = val_data[1]
        max_num_lstm_data = max([i[0].size(0) for i in self.val_lstm_data]) * back_num_states
        self.val_num_iter_per_epoch = max_num_lstm_data // self.r_hparam["max_batch_size"] + (max_num_lstm_data % self.r_hparam["max_batch_size"] != 0)
        self.val_lstm_indices_dataloader = [IndexRandomGenerator(val_lstm_bacth_num[s], self.val_num_iter_per_epoch, False) for s in range(back_num_states)]


        self.test_lstm_data  = test_data[1]
        max_num_lstm_data = max([i[0].size(0) for i in self.test_lstm_data]) * back_num_states
        self.test_num_iter_per_epoch = max_num_lstm_data // self.r_hparam["max_batch_size"] + (max_num_lstm_data % self.r_hparam["max_batch_size"] != 0)
        self.test_lstm_indices_dataloader = [IndexRandomGenerator(test_lstm_bacth_num[s], self.test_num_iter_per_epoch, False) for s in range(back_num_states)]

        # import pdb ; pdb.set_trace()
        self.model = HyperModel(path_data_dims = pathx_dim_per_state, lstm_data_dim = self.test_lstm_data[0].size(-1), lstm_out_dim=out_dim, 
                                data_edge=state_edge_list, front_num_states = front_num_states, back_num_states = front_num_states, concat_info_front2back = concat_info_front2back,
                                **self.lstm_hparam, num_batch=self.r_hparam["path_batch_size"], device=self.device, num_time_model = self.r_hparam["num_time_model"], prior_knowledge = self.r_hparam["prior"],
                                **self.emb_process_hparam, **self.lstm_process_hparam,**self.rest_process_hparam, **self.hyper_model_hparam)
        self.model = self.model.to(self.device)
        self.coef_dict = {}
        if self.emb_process_hparam["emb_process_args"]["emb_kinetic"]:
            self.coef_dict["e_ki_sq"] = self.r_hparam["e_ki_sq"]
            self.coef_dict["e_ki_quad"] = self.r_hparam["e_ki_quad"]
        if self.lstm_process_hparam["lstm_process_args"]["lstm_kinetic"]:
            self.coef_dict["z_ki_sq"] = self.r_hparam["z_ki_sq"]
            self.coef_dict["z_ki_quad"] = self.r_hparam["z_ki_quad"]

        # if self.data_hparam["d_folder"] == "ushcn":
        #     oparam = [{"params": [], "lr": 0.1,}, {"params": [], "lr": 0.0001}]
        #     for k,v in self.model.named_parameters():
        #         if k.split(".")[0] == "model_param_lst":
        #             oparam[0]["params"].append(v)
        #         else:
        #             oparam[1]["params"].append(v)

        #     optimizer = torch.optim.Adam(oparam, weight_decay= self.r_hparam["weight_decay"])
        # else:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.r_hparam["lr"], weight_decay= self.r_hparam["weight_decay"])
        
        
        
        # if self.data_hparam["d_folder"] == "ushcn":
        #     lr_s = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", min_lr=1e-5, patience=5)
        if self.r_hparam["loss_func"] == "rmse":
            loss_fn = RMSELoss()
        elif self.r_hparam["loss_func"] == "mse":
            loss_fn = torch.nn.MSELoss()
        elif self.r_hparam["loss_func"] == "l1":
            loss_fn = torch.nn.L1Loss()
        self.logger.info("Learning start")
        self.logger.info(f'# of params = {cal_num_param(self.model)}')
        
        iteration = -1
        saved_dict, model_dict = {}, { "model":{}}

        for epoch in range(self.r_hparam["epochs"]):
            [self.train_lstm_indices_dataloader[b][s].init()  for b in range(num_batch) for s in range(back_num_states)]
            all_loss = []
            for _ in range(num_iter_per_epoch):
                for path_x, batch_idx, final_index in self.x_path_train_data:
                    path_x = path_x.to(self.device)
                    iteration += 1
                    lstm_model_lst, lstm_model_lst2, attn_o, reg = self.model(path_x, final_index, True, True)
                    task_loss = \
                    [(self.cal_loss(lstm_model_lst[i][s], self.train_lstm_data[b][0], self.train_lstm_data[b][1], self.train_lstm_indices_dataloader[b][s].next(), s, loss_fn, self.device),
                      self.cal_loss(lstm_model_lst2[i][s], self.train_lstm_data[b][0], self.train_lstm_data[b][1], self.train_lstm_indices_dataloader[b][s].rep(), s, loss_fn, self.device),)
                        for s in range(back_num_states) for i, b in enumerate(batch_idx)]
                    task_loss2 = torch.stack([i[1] for i in task_loss])
                    task_loss = torch.stack([i[0] for i in task_loss])

                    # mask = torch.rand(len(task_loss)) >= 0.3
                    # task_loss = task_loss[mask]
                    if self.r_hparam["task2_ratio"] == 0:
                        task_loss = sum(task_loss) / len(task_loss)
                    else:
                        # if self.data_hparam["d_folder"] == "ushcn" and epoch <= 30:
                        #     le = attn_o.size(1)
                        #     task_loss = sum(task_loss) / len(task_loss) 
                        #     task_loss = task_loss - self.r_hparam["task2_ratio"]  *  attn_o[:,torch.arange(le),:,torch.arange(le)].sum()
                        # else:
                            task_loss = sum(task_loss) / len(task_loss) + self.r_hparam["task2_ratio"] * sum(task_loss2) / len(task_loss2)
                    # print(task_loss)
                    

                    self.writer.add_scalar(f"train/loss", task_loss, iteration)
                    for k,v in self.coef_dict.items():
                        self.writer.add_scalar(f"train/{k}", reg[k], iteration)
                        task_loss = task_loss + v * reg[k]
                    all_loss.append(task_loss.item())
                    optimizer.zero_grad()
                    task_loss.backward()
                    optimizer.step()
            train_loss = -sum(all_loss) / len(all_loss)
            self.model.eval()
            val_dict, val_y_stack, val_yhat_stack = self.cal_score("val",back_num_states, loss_fn, epoch)
            test_dict, test_y_stack, test_yhat_stack = self.cal_score("test",back_num_states, loss_fn, epoch)
            test_loss = test_dict['loss']
            val_loss  = val_dict['loss']
            if "nan" in str(test_loss):
                assert 0
            self.logger.info(f'{epoch} epoch : train_loss = {train_loss:.4f}  /  val_loss = {val_loss:.4f}  /  test_loss = {test_loss:.4f}' )
            change, saved_dict, model_dict = save_score_model(val_dict, test_dict, (val_y_stack, val_yhat_stack, test_y_stack, test_yhat_stack), saved_dict,model_dict, epoch)
            
            if change:
                torch.save(model_dict, f"{self.file_path}/model/model.pth")
                save_dict(f"{self.file_path}/score_info/score.json",saved_dict)
            self.model.train()
            
            # if self.data_hparam["d_folder"] == "ushcn":
            #     lr_s.step(val_loss)
            #     print(optimizer.param_groups[0]['lr'])
                # print(lr_s.get_last_lr())

                        
            

    def cal_score(self, name, tmp_num_states, loss_fn, epoch):
        if name == "test":
            lstm_indices_dataloader = self.test_lstm_indices_dataloader
            path_data, path_data_len = self.x_path_test_data, self.x_path_test_data_len
            lstm_data = self.test_lstm_data
            num_iter_per_epoch = self.test_num_iter_per_epoch
        else:
            lstm_indices_dataloader = self.val_lstm_indices_dataloader
            path_data, path_data_len = self.x_path_val_data, self.x_path_val_data_len
            lstm_data = self.val_lstm_data
            num_iter_per_epoch = self.val_num_iter_per_epoch
            

        [lstm_indices_dataloader[s].init() for s in range(tmp_num_states)]
        lstm_model_lst, _ = self.model(path_data.to(self.device), path_data_len, False)
        result = \
        [self.cal_yhat_y(lstm_model_lst[0][s], lstm_data[0][:,s], lstm_data[1][:,s], lstm_indices_dataloader[s].next(), s, self.device) for s in range(tmp_num_states)]
        
        real_y = torch.cat([i[0] for i in result],dim=0)
        hat_y  = torch.cat([i[1] for i in result],dim=0)
        real_y_stack = torch.stack([i[0] for i in result],dim=1)
        hat_y_stack = torch.stack([i[1] for i in result],dim=1)
        
        score_dic = cal_score(hat_y.cpu().detach().numpy(), real_y.cpu().detach().numpy(), self.prob_type)
        loss = loss_fn(real_y,hat_y)
        score_dic["loss"] = -loss.item()
        self.model.train()
        for k,v in score_dic.items():
            self.writer.add_scalar(f"{name}/{k}",v, epoch) 
        return score_dic, real_y_stack.cpu().detach().numpy(), hat_y_stack.cpu().detach().numpy()
            
            # save_dict(f"{self.file_path}/score_info/score.json",score_dic)
            # torch.save(model.state_dict(), f"{self.file_path}/model/model.pth")
            

    def cal_loss(self, lstm_model, x, y, batch_idxes, state, loss_fn, device):
        x = x[batch_idxes,state].to(device)
        y = y[batch_idxes,state].to(device)
        if self.lstm_hparam["lstm_name"].lower() in ["seq2seq", "geq2geq"]:
            y_hat = lstm_model(x,y=y)
        else:
            y_hat = lstm_model(x)
        if type(y_hat) is tuple:
            loss  = sum(y_hat[1:])
            y_hat = y_hat[0]
        else:
            loss = 0
        return loss_fn(y, y_hat) + loss
        
    def cal_yhat_y(self, lstm_model, x, y, batch_idxes, state, device): # for test
        # x = x[batch_idxes,state].to(device)
        # y = y[batch_idxes,state].to(device)
        x = x.to(device)
        y = y.to(device)
        y_hat = lstm_model(x,False)
        return y, y_hat