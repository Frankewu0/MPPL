from torch.utils.data import Dataset
import os
import numpy as np
import scipy.sparse as sps
import torch
# import dgl.backend as F
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import dgl
import numpy as np
import os
import random
import warnings
warnings.filterwarnings("ignore")
# local_rank = int(os.environ['LOCAL_RANK'])

import pandas as pd


def load_csv(path):
    data = pd.read_csv(path)

    scaffold_training = data[data['scaffold_train_test_label'] == 'train']
    scaffold_test = data[data['scaffold_train_test_label'] == 'test']
    scaffold_training = scaffold_training.reset_index()
    scaffold_test = scaffold_test.reset_index()

    # print(len(scaffold_training['Smiles_unify']))
    max_shape = 0 
    num = 0
    smi_list = []
    for i in range(len(scaffold_training['Smiles_unify'])):
        shape = len(scaffold_training['Smiles_unify'][i])
        smi_list.append(scaffold_training['Smiles_unify'][i])
        num += 1
    # print(f"number of the valuable data is {num}.")


    # print(len(scaffold_test['Smiles_unify']))
    max_shape = 0 
    num = 0
    test_smi_list = []
    for i in range(len(scaffold_test['Smiles_unify'])):
        shape = len(scaffold_test['Smiles_unify'][i])
        test_smi_list.append(scaffold_test['Smiles_unify'][i])
        num += 1
    return smi_list, test_smi_list, scaffold_training, scaffold_test
    

def list_min_max_norm_for_predicted_property(x_list, max_v, min_v):
    x_np = np.array(x_list)
    print(x_np.max(), ", ", x_np.min())
    x_np = (x_np - min_v) / (max_v - min_v)
    x_np = np.where(x_np<0.5, x_np-0.3, x_np+0.3)
    # x_np = np.where(x_np<0.5, x_np-0.2, x_np+0.2)
    x_np[x_np<0.] = 0.
    x_np[x_np>1.] = 1.
    # x_np = x_np * 2. - 1.
    x_list = x_np.tolist()
    return x_list
    

def list_min_max_norm(x_list, max_v, min_v):
    x_np = np.array(x_list)
    x_np = (x_np - min_v) / (max_v - min_v)  # [0, 1]
    x_np = x_np * 2. - 1.   # [-1, 1]
    
    # x_np = np.where(x_np<0.5, x_np-0.3, x_np+0.3)
    # x_np[x_np<0.] = 0.
    # x_np[x_np>1.] = 1.
    
    x_list = x_np.tolist()
    return x_list


class PharmaBenchDataset(Dataset):
    def __init__(self, task_name="water_sol_reg", mode="train", train_num=None):
        fp_path = f"/home/jovyan/prompts_learning/pretrain_data/{task_name}_{mode}_rdkfp1-7_512.npz"
        md_path = f"/home/jovyan/prompts_learning/pretrain_data/{task_name}_{mode}_molecular_descriptors.npz"
        # with open(smiles_path, 'r') as f:
        #     lines = f.readlines()
        #     self.smiles_list = [line.strip('\n') for line in lines]
        self.fps = torch.from_numpy(sps.load_npz(fp_path).todense().astype(np.float32))
        mds = np.load(md_path)['md'].astype(np.float32)
        mds = np.where(np.isnan(mds), 0, mds)
        self.mds = torch.from_numpy(mds)
        self.d_fps = self.fps.shape[1]
        self.d_mds = self.mds.shape[1]        
        
        csv_path = f"/home/jovyan/prompts_learning/12_2_data/pred/{task_name}_final_data.csv"
        if task_name == "bbb_cls":
            print("loaded bbb")
            csv_path = f"/home/jovyan/prompts_learning/bbb_cls_final_data_multi_class.csv"
        smi_list, test_smi_list, scaffold_training, scaffold_test = load_csv(csv_path)
        if mode == "train":
            self.smiles_list = smi_list
        elif mode == "test":
            self.smiles_list = test_smi_list
            
        
        # get correponding label from csv
        self.value_list = []
        self.logd_list = []
        self.logp_list = []
        self.pka_list = []
        self.pkb_list = []
        self.logsol_list = []
        self.wlogsol_list = []
        self.max_len = 0
        if mode == "train":
            pd_data = scaffold_training
        else:
            pd_data = scaffold_test
        
        self.max_value = 0.
        self.min_value = 999.
        
        # if train_num is None:
        for i in range(len(self.smiles_list)):
            idx = pd_data['Smiles_unify'][pd_data['Smiles_unify']==self.smiles_list[i]].index[0]
            self.value_list.append(pd_data['value'][idx])
            
            
            # For multiple prompts
            self.logd_list.append(pd_data['LogD_pred'][idx])
            self.logp_list.append(pd_data['LogP_pred'][idx])
            self.pka_list.append(pd_data['pKa_class_pred'][idx])
            self.pkb_list.append(pd_data['pKb_class_pred'][idx])
            self.logsol_list.append(pd_data['LogSol_pred'][idx])
            self.wlogsol_list.append(pd_data['wLogSol_pred'][idx])
            
        # Normalize these predicted property
        # self.logd_list = list_min_max_norm_for_predicted_property(self.logd_list, 8.7890625, -3.41796875)
        # self.logp_list = list_min_max_norm_for_predicted_property(self.logp_list, 12.796875, -5.53125)
        # self.logsol_list = list_min_max_norm_for_predicted_property(self.logsol_list, 3.017578125, -0.54931640625)
        # self.wlogsol_list = list_min_max_norm_for_predicted_property(self.wlogsol_list, 3.92578125, -12.3984375)
 
        

        if task_name == "water_sol_reg":
            self.value_list = list_min_max_norm(self.value_list, 10.141333333333334, -9.2177) # sol
            self.max_v, self.min_v = 10.141333333333334, -9.2177
            
            self.logd_list = list_min_max_norm_for_predicted_property(self.logd_list, 9.0 ,  -3.47265625)
            self.logp_list = list_min_max_norm_for_predicted_property(self.logp_list, 12.0 ,  -5.53515625)
            self.logsol_list = list_min_max_norm_for_predicted_property(self.logsol_list, 2.94921875 ,  -0.65283203125)
            self.wlogsol_list = list_min_max_norm_for_predicted_property(self.wlogsol_list, 4.5 ,  -13.9375)
        elif task_name == "ppb_reg":
            self.value_list = list_min_max_norm(self.value_list, 1., 0.) # ppb
            self.max_v, self.min_v = 1., 0. 
            
            self.logd_list = list_min_max_norm_for_predicted_property(self.logd_list, 6.7578125 ,  -1.42578125)
            self.logp_list = list_min_max_norm_for_predicted_property(self.logp_list, 7.3046875, -2.498046875)
            self.logsol_list = list_min_max_norm_for_predicted_property(self.logsol_list, 2.91015625 ,  -0.47705078125)
            self.wlogsol_list = list_min_max_norm_for_predicted_property(self.wlogsol_list, 4.01171875 ,  -10.640625)
        elif task_name == "cyp_2c9_reg":
            self.value_list = list_min_max_norm(self.value_list, 100., 0.0005) # cyp_2c9
            self.max_v, self.min_v = 100., 0.0005 
            
            self.logd_list = list_min_max_norm_for_predicted_property(self.logd_list, 5.328125 ,  -0.84521484375)
            self.logp_list = list_min_max_norm_for_predicted_property(self.logp_list, 7.640625 ,  -0.0003869533538818)
            self.logsol_list = list_min_max_norm_for_predicted_property(self.logsol_list, 2.728515625 ,  -0.33935546875)
            self.wlogsol_list = list_min_max_norm_for_predicted_property(self.wlogsol_list, 2.65625 ,  -10.3203125)
        elif task_name == "cyp_2d6_reg":
            self.value_list = list_min_max_norm(self.value_list, 100., 4.5e-5) # cyp_2d6
            self.max_v, self.min_v = 100., 4.5e-5 
            
            self.logd_list = list_min_max_norm_for_predicted_property(self.logd_list, 6.78515625 ,  -0.72802734375)
            self.logp_list = list_min_max_norm_for_predicted_property(self.logp_list, 9.5390625 ,  -0.0960693359375)
            self.logsol_list = list_min_max_norm_for_predicted_property(self.logsol_list, 2.849609375 ,  -0.34619140625)
            self.wlogsol_list = list_min_max_norm_for_predicted_property(self.wlogsol_list, 2.703125 ,  -10.2578125)
        elif task_name == "cyp_3a4_reg":
            self.value_list = list_min_max_norm(self.value_list, 100., 0.0003) # cyp_3a4
            self.max_v, self.min_v = 100., 0.0003
            
            self.logd_list = list_min_max_norm_for_predicted_property(self.logd_list, 5.40625 ,  -1.40234375)
            self.logp_list = list_min_max_norm_for_predicted_property(self.logp_list, 7.87109375 ,  0.36376953125)
            self.logsol_list = list_min_max_norm_for_predicted_property(self.logsol_list, 2.8515625 ,  -0.31298828125)
            self.wlogsol_list = list_min_max_norm_for_predicted_property(self.wlogsol_list, 3.69140625 ,  -10.8046875)
        elif task_name == "hum_mic_cl_reg":
            self.value_list = list_min_max_norm(self.value_list, 6.361727836017593, -2.5228787452803374) # hum_mic_cl 
            self.max_v, self.min_v = 6.361727836017593, -2.5228787452803374 
            
            self.logd_list = list_min_max_norm_for_predicted_property(self.logd_list, 5.328125, -1.158203125)
            self.logp_list = list_min_max_norm_for_predicted_property(self.logp_list, 8.28125 ,  -0.2034912109375)
            self.logsol_list = list_min_max_norm_for_predicted_property(self.logsol_list, 2.830078125 ,  -0.45263671875)
            self.wlogsol_list = list_min_max_norm_for_predicted_property(self.wlogsol_list, 3.5625 ,  -9.296875)
        elif task_name == "rat_mic_cl_reg":
            self.value_list = list_min_max_norm(self.value_list, 5.662757831681574, -2.0) # rat_mic_cl
            self.max_v, self.min_v = 5.662757831681574, -2.0
            
            self.logd_list = list_min_max_norm_for_predicted_property(self.logd_list, 7.77734375, -1.076171875)
            self.logp_list = list_min_max_norm_for_predicted_property(self.logp_list, 8.28125 ,  -1.8359375)
            self.logsol_list = list_min_max_norm_for_predicted_property(self.logsol_list, 2.818359375, -0.32568359375)
            self.wlogsol_list = list_min_max_norm_for_predicted_property(self.wlogsol_list, 3.5625, -9.296875)
        elif task_name == "mou_mic_cl_reg":
            self.value_list = list_min_max_norm(self.value_list, 5.934498451243568, -2.397940008672037) # mou_mic_cl
            self.max_v, self.min_v = 5.934498451243568, -2.397940008672037
            
            self.logd_list = list_min_max_norm_for_predicted_property(self.logd_list, 8.7890625, -3.41796875)
            self.logp_list = list_min_max_norm_for_predicted_property(self.logp_list, 12.796875, -5.53125)
            self.logsol_list = list_min_max_norm_for_predicted_property(self.logsol_list, 3.017578125, -0.54931640625)
            self.wlogsol_list = list_min_max_norm_for_predicted_property(self.wlogsol_list, 3.92578125, -12.3984375)
        elif task_name == "logd_reg":
            self.value_list = list_min_max_norm(self.value_list, 9.13, -4.8) # logd
            self.max_v, self.min_v = 9.13, -4.8
            
            self.logd_list = list_min_max_norm_for_predicted_property(self.logd_list, 8.375, -1.30078125)
            self.logp_list = list_min_max_norm_for_predicted_property(self.logp_list, 8.9453125, -1.908203125)
            self.logsol_list = list_min_max_norm_for_predicted_property(self.logsol_list, 2.8125 ,  -0.48681640625)
            self.wlogsol_list = list_min_max_norm_for_predicted_property(self.wlogsol_list, 3.4609375 ,  -11.4375)
            
        else:
            self.max_v, self.min_v = 0, 1
            num_p = 0
            num_n = 0
            for lab in self.value_list:
                if lab == 1: num_p += 1
                else: num_n += 1
            self.wp = num_p
            self.wn = num_n
            print(f"number of postive/negtive samples {num_p}/{num_n}.")
            
            if task_name == "bbb_cls":
                self.logd_list = list_min_max_norm_for_predicted_property(self.logd_list, 8.9453125 ,  -3.41796875)
                self.logp_list = list_min_max_norm_for_predicted_property(self.logp_list, 12.796875 ,  -5.53125)
                self.logsol_list = list_min_max_norm_for_predicted_property(self.logsol_list, 3.017578125, -0.56689453125)
                self.wlogsol_list = list_min_max_norm_for_predicted_property(self.wlogsol_list, 3.92578125 ,  -12.3984375)
            elif task_name == "ames_cls":
                self.logd_list = list_min_max_norm_for_predicted_property(self.logd_list, 8.96875 ,  -3.4375)
                self.logp_list = list_min_max_norm_for_predicted_property(self.logp_list, 12.828125 ,  -4.3359375)
                self.logsol_list = list_min_max_norm_for_predicted_property(self.logsol_list, 2.953125,  -0.52783203125)
                self.wlogsol_list = list_min_max_norm_for_predicted_property(self.wlogsol_list, 4.4296875, -13.2265625)
                

    def __len__(self):
        return len(self.value_list)
    
    def __getitem__(self, idx):
        label = self.value_list[idx]
        logd = self.logd_list[idx]
        logp = self.logp_list[idx]
        pka = self.pka_list[idx]
        pkb = self.pkb_list[idx]
        logsol = self.logsol_list[idx]
        wlogsol = self.wlogsol_list[idx]
        return self.smiles_list[idx], self.fps[idx], self.mds[idx],\
    label, logd, logp, pka, pkb, logsol, wlogsol