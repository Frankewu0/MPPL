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

    scaffold_training = data[data['label'] == 'train']
    scaffold_test = data[data['label'] == 'test']
    scaffold_training = scaffold_training.reset_index()
    scaffold_test = scaffold_test.reset_index()

    # print(len(scaffold_training['Smiles_unify']))
    max_shape = 0 
    num = 0
    smi_list = []
    for i in range(len(scaffold_training['smiles'])):
        shape = len(scaffold_training['smiles'][i])
        smi_list.append(scaffold_training['smiles'][i])
        num += 1
    # print(f"number of the valuable data is {num}.")


    # print(len(scaffold_test['Smiles_unify']))
    max_shape = 0 
    num = 0
    test_smi_list = []
    for i in range(len(scaffold_test['smiles'])):
        shape = len(scaffold_test['smiles'][i])
        test_smi_list.append(scaffold_test['smiles'][i])
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


class MolNetDataset(Dataset):
    def __init__(self, task_name="bace", mode="train", train_num=None):
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
        
        csv_path = f"/home/jovyan/molcularnet/finetune/{task_name}.csv"
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
            # For multiple prompts
            idx = pd_data['smiles'][pd_data['smiles']==self.smiles_list[i]].index[0]
            self.logd_list.append(pd_data['LogD_pred'][idx])
            self.logp_list.append(pd_data['LogP_pred'][idx])
            self.pka_list.append(pd_data['pKa_class_pred'][idx])
            self.pkb_list.append(pd_data['pKb_class_pred'][idx])
            self.logsol_list.append(pd_data['LogSol_pred'][idx])
            self.wlogsol_list.append(pd_data['wLogSol_pred'][idx])
 
        

        if task_name == "esol":
            for i in range(len(self.smiles_list)):
                idx = pd_data['smiles'][pd_data['smiles']==self.smiles_list[i]].index[0]
                self.value_list.append(pd_data['logSolubility'][idx])
            print("value:", max(self.value_list),",", min(self.value_list))
            self.value_list = list_min_max_norm(self.value_list, 1.144 , -11.6) 
            self.max_v, self.min_v = 1.144 , -11.6
            
            self.logd_list = list_min_max_norm_for_predicted_property(self.logd_list, 8.3203125 ,  -3.4453125)
            self.logp_list = list_min_max_norm_for_predicted_property(self.logp_list, 9.5390625 ,  -0.0960693359375)
            self.logsol_list = list_min_max_norm_for_predicted_property(self.logsol_list, 2.529296875 ,  -0.52978515625)
            self.wlogsol_list = list_min_max_norm_for_predicted_property(self.wlogsol_list, 1.1171875 ,  -10.609375)
        elif task_name == "freesolv":
            for i in range(len(self.smiles_list)):
                idx = pd_data['smiles'][pd_data['smiles']==self.smiles_list[i]].index[0]
                self.value_list.append(pd_data['freesolv'][idx])
            self.value_list = list_min_max_norm(self.value_list, 3.16 , -23.62) 
            self.max_v, self.min_v = 3.16 , -23.62
            
            self.logd_list = list_min_max_norm_for_predicted_property(self.logd_list, 5.203125 ,  -3.28125)
            self.logp_list = list_min_max_norm_for_predicted_property(self.logp_list, 8.4296875 ,  -3.052734375)
            self.logsol_list = list_min_max_norm_for_predicted_property(self.logsol_list, 2.365234375 ,  -0.1722412109375)
            self.wlogsol_list = list_min_max_norm_for_predicted_property(self.wlogsol_list, 1.060546875 ,  -10.6484375)
        elif task_name == "lipo":
            for i in range(len(self.smiles_list)):
                idx = pd_data['smiles'][pd_data['smiles']==self.smiles_list[i]].index[0]
                self.value_list.append(pd_data['lipo'][idx])
            
            self.value_list = list_min_max_norm(self.value_list, 4.5 , -1.48)
            self.max_v, self.min_v = 4.5 , -1.48
            
            self.logd_list = list_min_max_norm_for_predicted_property(self.logd_list, 4.71484375 ,  -1.44921875)
            self.logp_list = list_min_max_norm_for_predicted_property(self.logp_list, 7.63671875 ,  -3.564453125)
            self.logsol_list = list_min_max_norm_for_predicted_property(self.logsol_list, 2.947265625 ,  -0.43994140625)
            self.wlogsol_list = list_min_max_norm_for_predicted_property(self.wlogsol_list, 3.39453125 ,  -9.53125)
     
            
        else:
            if task_name == "bace":
                for i in range(len(self.smiles_list)):
                    idx = pd_data['smiles'][pd_data['smiles']==self.smiles_list[i]].index[0]
                    self.value_list.append(pd_data['Class'][idx])
                
                self.logd_list = list_min_max_norm_for_predicted_property(self.logd_list, 5.1953125 ,  -1.4375)
                self.logp_list = list_min_max_norm_for_predicted_property(self.logp_list, 6.60546875 ,  -2.037109375)
                self.logsol_list = list_min_max_norm_for_predicted_property(self.logsol_list, 2.896484375 ,  0.1392822265625)
                self.wlogsol_list = list_min_max_norm_for_predicted_property(self.wlogsol_list, 3.046875 ,  -6.56640625)
            elif task_name == "bbbp":
                for i in range(len(self.smiles_list)):
                    idx = pd_data['smiles'][pd_data['smiles']==self.smiles_list[i]].index[0]
                    self.value_list.append(pd_data['p_np'][idx])

                self.logd_list = list_min_max_norm_for_predicted_property(self.logd_list, 8.5 ,  -2.8125)
                self.logp_list = list_min_max_norm_for_predicted_property(self.logp_list, 9.9296875 ,  -5.53515625)
                self.logsol_list = list_min_max_norm_for_predicted_property(self.logsol_list, 2.845703125 ,  0.045501708984375)
                self.wlogsol_list = list_min_max_norm_for_predicted_property(self.wlogsol_list, 2.51953125 ,  -12.4296875)
            elif task_name == "clintox":
                for i in range(len(self.smiles_list)):
                    idx = pd_data['smiles'][pd_data['smiles']==self.smiles_list[i]].index[0]
                    self.value_list.append(pd_data['CT_TOX'][idx])
                print("value:", max(self.value_list),",", min(self.value_list))
                self.logd_list = list_min_max_norm_for_predicted_property(self.logd_list, 8.9453125 ,  -3.36328125)
                self.logp_list = list_min_max_norm_for_predicted_property(self.logp_list, 12.84375 ,  -4.94140625)
                self.logsol_list = list_min_max_norm_for_predicted_property(self.logsol_list, 2.96875 ,  -0.54345703125)
                self.wlogsol_list = list_min_max_norm_for_predicted_property(self.wlogsol_list, 4.43359375 ,  -12.609375)
            
            self.max_v, self.min_v = 0, 1
            num_p = 0
            num_n = 0
            for lab in self.value_list:
                if lab == 1: num_p += 1
                else: num_n += 1
            self.wp = num_p
            self.wn = num_n
            print(f"number of postive/negtive samples {num_p}/{num_n}.")

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