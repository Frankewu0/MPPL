import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import time
import gc
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm
# torch.autograd.set_detect_anomaly(True)
CUDA_LAUNCH_BLOCKING=1

# fixed random seed for reproduction
seed = 10
seed = 1024
# seed = 3407
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
print('Random seed :', seed)

from collections import OrderedDict
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag



def print_loss(loss, loss_name):
    print(f"{loss_name}: {loss.detach().cpu().numpy():.4f}; ", end='', flush=True)
    # print('\r', end='', flush=True)


##########################################
#########  construct dataloader  ######### 
##########################################
from pharmabench_pipeline import PharmaBenchDataset
from KPGT.src.data.featurizer import Vocab, N_BOND_TYPES, N_ATOM_TYPES
from KPGT.src.data.collator_tune import Collator_pretrain
from KPGT.src.model_config import config_dict
config = config_dict['base']

vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
collator = Collator_pretrain(vocab, max_length=config['path_length'], n_virtual_nodes=2, candi_rate=config['candi_rate'], fp_disturb_rate=config['fp_disturb_rate'], md_disturb_rate=config['md_disturb_rate'])



task_mode = "regression"
# task_name = "logd_reg"
# task_name = "water_sol_reg" 
# task_name = "ppb_reg"
# task_name = "cyp_3a4_reg"
# task_name = "cyp_2c9_reg" 
# task_name = "cyp_2d6_reg" 
# task_name = "hum_mic_cl_reg" 
# task_name = "rat_mic_cl_reg" 
task_name = "mou_mic_cl_reg" 

task_mode = "classification"
# task_name = "bbb_cls"
task_name = "ames_cls"
train_dataset = PharmaBenchDataset(task_name, mode="train")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, drop_last=True, collate_fn=collator)
test_dataset = PharmaBenchDataset(task_name, mode="test")
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=False, collate_fn=collator)
max_v, min_v = test_dataset.max_v, test_dataset.min_v
if task_mode == "classification":
    wp, wn = train_dataset.wp, test_dataset.wn


##########################################
######  build model and optimizer  ####### 
##########################################
dev = "cuda" if torch.cuda.is_available() else "cpu"
# from CLIP import clip
# clip_model, preprocess = clip.load(name="ViT-B/16", device="cpu", download_root="/home/jovyan/clip_download_root")

from KPGT.src.model.light import LiGhTPredictor as LiGhT
kpgt = LiGhT(
        d_node_feats=config['d_node_feats'],
        d_edge_feats=config['d_edge_feats'],
        d_g_feats=config['d_g_feats'],
        d_fp_feats=train_dataset.d_fps,
        d_md_feats=train_dataset.d_mds,
        d_hpath_ratio=config['d_hpath_ratio'],
        n_mol_layers=config['n_mol_layers'],
        path_length=config['path_length'],
        n_heads=config['n_heads'],
        n_ffn_dense_layers=config['n_ffn_dense_layers'],
        # input_drop=config['input_drop'],
        # attn_drop=config['attn_drop'],
        # feat_drop=config['feat_drop'],
        input_drop=0.0,
        # attn_drop=0.,
        # feat_drop=0.,
        attn_drop=0.1,
        feat_drop=0.1,
        # attn_drop=0.2,
        # feat_drop=0.2,
        n_node_types=vocab.vocab_size
    )# .to("cuda")
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
# kpgt.load_state_dict({k.replace('module.', ''): v for k, v in torch.load("/home/jovyan/prompts_learning/KPGT/src/models/base.pth").items()})
# print("Pre-trained weights of KPGT were loaded successfully!")

from model_zoo import MMPL_Finetune_Model
model = MMPL_Finetune_Model(kpgt).to(dev)

from copy import deepcopy
ema = deepcopy(model).to(dev)  # Create an EMA of the model for use after training
update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
requires_grad(ema, False)
ema.eval()

# best
lr = 5e-5
wd = 0.

print(f'Set of Optimizer: lr:{lr}, weight_decay:{wd}')
model_params = [
                # {'params': model.parameters(), 'lr': lr, "weight_decay": wd},
    {'params': model.mol_encoder.parameters(), 'lr': 5e-5, "weight_decay": wd},
    # {'params': model.mol_encoder.parameters(), 'lr': 1e-5, "weight_decay": wd}, # best
    {'params': model.text_proc.parameters(), 'lr': 1e-5, "weight_decay": wd},
    # {'params': model.head.parameters(), 'lr': 2e-4, "weight_decay": 1e-2},
    # {'params': model.head.parameters(), 'lr': 5e-5, "weight_decay": 0.},
    {'params': model.head.parameters(), 'lr': 1e-3, "weight_decay": 5e-2},
    
    # {'params': model.mpim.parameters(), 'lr': 5e-5, "weight_decay": wd},
    {'params': model.mpim.model.parameters(), 'lr': 5e-5, "weight_decay": wd},
    {'params': model.mpim.attn_model.parameters(), 'lr': 5e-5, "weight_decay": wd},
    {'params': model.mpim.prom_emb.parameters(), 'lr': 3e-5, "weight_decay": wd},
    {'params': model.mpim.mol_emb.parameters(), 'lr': 3e-5, "weight_decay": wd},
    {'params': model.mpim.positional_embedding, 'lr': 5e-5, "weight_decay": wd},
    
    {'params': model.mpim.moltex_attn.parameters(), 'lr': 1e-3, "weight_decay": 1e-2},
    # {'params': model.mpim.moltex_attn.parameters(), 'lr': 1e-3, "weight_decay": wd},
    
    {'params': model.mpim.attr_heads.parameters(), 'lr': 1e-3, "weight_decay": 1e-2}
               ]


optims = 'adan'
# optims = "sgd"
if optims == 'adan':
    from adan import Adan
    # optimizer = Adan(model_params, betas=(0.98, 0.92, 0.99), max_grad_norm=5.)
    optimizer = Adan(model_params, betas=(0.98, 0.92, 0.99), max_grad_norm=0.)
elif optims == 'sgd':
    optimizer = optim.SGD(model_params, momentum=0.9, weight_decay=wd)
elif optims == 'adamw':
    optimizer = optim.AdamW(model_params, betas=(0.9, 0.999), weight_decay=wd)
elif optims == 'adam':
    optimizer = optim.Adam(model_params, betas=(0.9, 0.999), weight_decay=wd)
print('Current Optimizer is', optims)


###################################################
########   build learning rate scheduler   ######## 
###################################################
# from KPGT.src.trainer.scheduler import PolynomialDecayLR
# scheduler = PolynomialDecayLR(optimizer, warmup_updates=1500, tot_updates=6225,lr=lr, end_lr=1e-9,power=1)
# cur_lr = scheduler.get_last_lr() 
# print(f"Current learning rate is {cur_lr}.")

# scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1./3., total_iters=5) # best
# scheduler2 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1., total_iters=5)
# scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
# scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2, scheduler3], milestones=[5, 10])
# cur_lr = scheduler.get_last_lr() 
# print(f"Current learning rate is {cur_lr}.")


##########################################
########   build loss criterion   ######## 
##########################################
# attr_loss = 'l1'
# attr_loss = 'mse'
attr_loss = 'bce'
# attr_loss = 'ce'

# red = 'sum'
red = 'mean'

print(f"attribution loss is {attr_loss}, and reduction method is {red}.")
if attr_loss == 'l1':
    # cri_attr = nn.L1Loss(reduction=red)
    cri_attr = nn.SmoothL1Loss(reduction=red)
elif attr_loss == 'mse':
    cri_attr = nn.MSELoss(reduction=red)
elif attr_loss == 'bce':
    cri_attr = nn.BCEWithLogitsLoss(reduction=red)
elif attr_loss == 'ce':
    cri_attr = nn.CrossEntropyLoss(reduction=red)


if task_mode == "regression":
    loss_func = torch.nn.MSELoss()
    best_metric = 99
elif task_mode == "classification":
    loss_func = torch.nn.BCEWithLogitsLoss()
    best_metric = 0.89
    
##########################################
######## start training our model ######## 
##########################################
gc.collect()
torch.cuda.empty_cache()
Epoch = 500
print("Let's start training!")

for e in range(0, Epoch):
    start = time.time()
    model.train()
    for step_id, batched_data in enumerate(tqdm(train_loader)):
        (_, batched_graph, fps, mds, _, _, _, label, logd, logp, pka, pkb, logsol, wlogsol) = batched_data
        batched_graph = batched_graph.to(dev)
        fps = fps.to(dev)
        mds = mds.to(dev)
        
        pred, attr_list = model([batched_graph, fps, mds])
        
        if task_mode == "classification":
            # loss_main = F.binary_cross_entropy_with_logits(pred, label.unsqueeze(-1).to(dev))
            loss_main = F.binary_cross_entropy_with_logits(pred, label.unsqueeze(-1).to(dev), reduction='none')
            for i, lab in enumerate(label):
                if lab == 0:
                    loss_main[i] = loss_main[i] * (wp/(wp+wn))  # best
                else:
                    loss_main[i] = loss_main[i] * (wn/(wp+wn))
            if red == 'mean':
                loss_main = loss_main.mean()
            elif red == 'sum':
                loss_main = loss_main.sum()
        else:
            loss_main = F.mse_loss(pred, label.unsqueeze(-1).to(dev))
            # loss_main = F.l1_loss(pred, label.unsqueeze(-1).to(dev))
            # loss_main = F.binary_cross_entropy_with_logits(pred, label.unsqueeze(-1).to(dev))
            
        loss_pka = F.binary_cross_entropy_with_logits(attr_list[2], pka.unsqueeze(-1).to(dev), reduction=red)
        loss_pkb = F.binary_cross_entropy_with_logits(attr_list[3], pkb.unsqueeze(-1).to(dev), reduction=red)
        loss_logd = cri_attr(attr_list[0], logd.unsqueeze(-1).to(dev))
        loss_logp = cri_attr(attr_list[1], logp.unsqueeze(-1).to(dev))
        loss_logsol = cri_attr(attr_list[4], logsol.unsqueeze(-1).to(dev))
        loss_wlogsol = cri_attr(attr_list[5], wlogsol.unsqueeze(-1).to(dev))

        loss_attr = loss_logd + loss_logp + loss_pka + loss_pkb # + loss_wlogsol + loss_logsol 
        # loss = loss_main*10. + loss_attr  
        # loss = loss_main*100. + loss_attr * 0.5
        loss = loss_main*100. + loss_attr * 0.7

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # update_ema(ema, model, 0.999)
        

    print(f"epoch: {e+1} / {Epoch}, loss:{loss.detach().cpu().numpy():.4f}")
    print_loss(loss_main, "Main")
    print_loss(loss_logd, "LogD")
    print_loss(loss_logp, "LogP")
    print_loss(loss_pka, "pKa")
    print_loss(loss_pkb, "pKb")
    print_loss(loss_logsol, "LogSol")
    print_loss(loss_wlogsol, "wLogSol")
    print()
                
    
    ##########################################
    ####### start evaluating our model #######
    ##########################################
    model.eval()
    print("evaluating...")
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for step_id, batched_data in enumerate(tqdm(test_loader)):
            (_, batched_graph, fps, mds, _, _, _, label, logd, logp, pka, pkb, logsol, wlogsol) = batched_data
            batched_graph = batched_graph.to(dev)
            fps = fps.to(dev)
            mds = mds.to(dev)
            y = label
            
            pred = model([batched_graph, fps, mds])
            # pred = ema([batched_graph, fps, mds])
            

            if task_mode == "regression":
                pred = (pred + 1.) / 2. * (max_v - min_v) + min_v
                y = (y + 1.) / 2. * (max_v - min_v) + min_v
                
                # pred = pred * (max_v - min_v) + min_v
                # y = y* (max_v - min_v) + min_v
            else:
                pred = torch.sigmoid(pred)

            y_true.append(y)
            y_pred.append(pred)

        y_true = torch.cat(y_true, dim=0).cpu().numpy()
        y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
        if task_mode == "regression":
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            mae = mean_absolute_error(y_true, y_pred)
            if best_metric > mae:
                best_metric = mae
                final_record = {"RMSE": rmse, "MAE": mae}
                torch.save(model.state_dict(), f'./ft_weight/{task_name}_Epoch{e+1}_val_mae_{best_metric:.5f}_rmse_{rmse:.5f}.pth') 
            print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        else:
            auc =  roc_auc_score(y_true, y_pred)
            th = 0.5
            y_pred[y_pred >= th] = 1.
            y_pred[y_pred < th] = 0.
            acc = average_precision_score(y_true, y_pred)
            if best_metric < auc:
                best_metric = auc
                torch.save(model.state_dict(), f'./ft_weight/{task_name}_Epoch{e+1}_val_auc_{best_metric:.5f}_acc_{acc:.5f}.pth')
                final_record = {"ACC": acc, "AUC": auc}
            print(f"ACC: {acc:.4f}, AUC: {auc:.4f}")
    # scheduler.step()
    # cur_lr = scheduler.get_last_lr() 
    # print(f"Current learning rate is {cur_lr}.")
    print()
print(final_record)
    
    