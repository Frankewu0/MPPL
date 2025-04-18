import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import time
import gc
from sklearn import metrics
from torch.cuda.amp import autocast, GradScaler
# torch.autograd.set_detect_anomaly(True)
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# fixed random seed for reproduction
seed = 10
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

from sklearn import metrics
def compute_AUC(y, pred, n_class=1):
    # compute one score
    if n_class == 1:
        auc = metrics.roc_auc_score(y, pred)

    # compute two-class
    elif n_class == 2:
        # pos = pred[:, 1]
        auc = metrics.roc_auc_score(y, pred)
    return auc


def print_loss(loss, loss_name):
    print(f"{loss_name}: {loss.detach().cpu().numpy():.4f}; ", end='', flush=True)
    # print('\r', end='', flush=True)


##########################################
#########  construct dataloader  ######### 
##########################################
 
from text_pretrain_datapipeline import MoleculeTextDataset
from KPGT.src.data.featurizer import Vocab, N_BOND_TYPES, N_ATOM_TYPES
from KPGT.src.data.collator_text import Collator_pretrain
from KPGT.src.model_config import config_dict
config = config_dict['base']

vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
collator = Collator_pretrain(vocab, max_length=config['path_length'], n_virtual_nodes=2, candi_rate=config['candi_rate'], fp_disturb_rate=config['fp_disturb_rate'], md_disturb_rate=config['md_disturb_rate'])
train_dataset = MoleculeTextDataset()
# train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8, drop_last=True, collate_fn=collator)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, drop_last=True, collate_fn=collator)


##########################################
######  build model and optimizer  ####### 
##########################################
dev = "cuda" if torch.cuda.is_available() else "cpu"
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
        input_drop=0.,
        attn_drop=0.,
        feat_drop=0.,
        n_node_types=vocab.vocab_size
    )# .to("cuda")
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
kpgt.load_state_dict({k.replace('module.', ''): v for k, v in torch.load("/home/jovyan/prompts_learning/KPGT/src/models/base.pth").items()})
print("Pre-trained weights of KPGT were loaded successfully!")

from model_zoo import MMPL_Pretrain_Model
model = MMPL_Pretrain_Model(kpgt).to(dev)


# from copy import deepcopy
# ema = deepcopy(model).to(dev)  # Create an EMA of the model for use after training
# update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
# requires_grad(ema, False)
# ema.eval()

# best
# lr = 1e-4
# wd = 0.

# best
lr = 5e-5
wd = 0.2

# best
lr = 1e-5
wd = 1e-6

print(f'Set of Optimizer: lr:{lr}, weight_decay:{wd}')
model_params = [
                # {'params': model.parameters(), 'lr': lr},
                
                # {'params': model.text_proc.parameters(), 'lr': 1e-3},
                {'params': model.text_proc.text_encoder.parameters(), 'lr': 1e-5},
                {'params': model.text_proc.tex_pad_token, 'lr': 1e-5},
                {'params': model.text_proc.model.parameters(), 'lr': 1e-5},
                {'params': model.text_proc.positional_embedding, 'lr': 1e-5},
                {'params': model.text_proc.text_proj.parameters(), 'lr': 3e-5},
                
                
                {'params': model.mol_encoder.mol_encoder.parameters(), 'lr': 1e-5},
                {'params': model.mol_encoder.positional_embedding, 'lr': 1e-5},
                {'params': model.mol_encoder.mol_proj.parameters(), 'lr': 3e-5},

                {'params': model.logit_scale, 'lr': 1e-5}
               ]


optims = 'adan'
# optims = "adam"
if optims == 'adan':
    from adan import Adan
    optimizer = Adan(model_params, betas=(0.98, 0.92, 0.99), weight_decay=wd, max_grad_norm=5., eps=1e-6)
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
# scheduler = PolynomialDecayLR(optimizer, warmup_updates=18630*1, tot_updates=18630*100,lr=lr, end_lr=1e-7,power=1)
# # scheduler = PolynomialDecayLR(optimizer, warmup_updates=1164*1, tot_updates=1164*100,lr=lr, end_lr=1e-7,power=1)
# cur_lr = scheduler.get_last_lr() 
# print(f"Current learning rate is {cur_lr}.")

scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1./3., total_iters=18630*1) # best
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=18630*120, eta_min=1e-7)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[18630*1])
cur_lr = scheduler.get_last_lr() 
print(f"Current learning rate is {cur_lr}.")



##########################################
######## start training our model ######## 
##########################################
gc.collect()
torch.cuda.empty_cache()
Epoch = 100
print("Let's start training!")
all_len = len(train_loader)
for e in range(0, Epoch):
    start = time.time()
    model.train()
    for step_id, batched_data in enumerate(train_loader):
        (_, batched_graph, fps, mds, _, _, _, text) = batched_data
        batched_graph = batched_graph.to(dev)
        fps = fps.to(dev)
        mds = mds.to(dev)
        text = text# .to(dev)
        
        loss = model([batched_graph, fps, mds], text)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        # update_ema(ema, model, 0.997)

        if not (step_id+1) % 200:
            print(f"epoch: {e+1} / {Epoch},step {step_id} / {all_len}, loss: {loss.detach().cpu().numpy():.8f}")
            gc.collect()
            torch.cuda.empty_cache()
                
    
    ##########################################
    ####### start evaluating our model #######
    ##########################################
    check_point = {
            'mol_encoder': model.mol_encoder.state_dict(),
            'text_proc': model.text_proc.state_dict(),
        }
    torch.save(check_point, f'./trained_weight/MMPL_KPGT_12_2_data_Epoch{e+1}.pth') 


    end = time.time()
    print(f"epoch: {e+1} end ; cost time: {(end - start)/60.:.4f} min")
    
    # scheduler.step()
    cur_lr = scheduler.get_last_lr() 
    print(f"Current learning rate is {cur_lr}.")