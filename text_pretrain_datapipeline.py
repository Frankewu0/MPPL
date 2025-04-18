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
import pandas as pd
from KPGT.src.data.featurizer import Vocab, N_BOND_TYPES, N_ATOM_TYPES
from KPGT.src.data.collator_text import Collator_pretrain
from KPGT.src.model.light import LiGhTPredictor as LiGhT
from KPGT.src.trainer.scheduler import PolynomialDecayLR
from KPGT.src.trainer.pretrain_trainer import Trainer
# from KPGT.src.trainer.evaluator import Evaluator
# from KPGT.src.trainer.result_tracker import Result_Tracker
from KPGT.src.model_config import config_dict
import warnings
warnings.filterwarnings("ignore")
# local_rank = int(os.environ['LOCAL_RANK'])

# import torch
# from torch_geometric.data import InMemoryDataset
# class PubChemDataset(InMemoryDataset):
#     def __init__(self, path):
#         super(PubChemDataset, self).__init__()
#         self.data, self.slices = torch.load(path)

#     def __getitem__(self, idx):
#         return self.get(idx)
    
# smi_list = []
# text_list = []
# dataset = PubChemDataset('./pretrain_data/PubChem324kV2/pretrain.pt')
# for i in range(len(dataset)):
#     smi = dataset[i]['smiles']
#     smi_list.append(smi)
#     tex = dataset[i]['text']
#     text_list.append(tex)


# from CLIP import clip
# class MoleculeTextDataset(Dataset):
#     def __init__(self):
#         fp_path = "/home/jovyan/prompts_learning/pretrain_data/rdkfp1-7_512.npz"
#         md_path = "/home/jovyan/prompts_learning/pretrain_data/molecular_descriptors.npz"
#         # with open(smiles_path, 'r') as f:
#         #     lines = f.readlines()
#         #     self.smiles_list = [line.strip('\n') for line in lines]
#         self.fps = torch.from_numpy(sps.load_npz(fp_path).todense().astype(np.float32))
#         mds = np.load(md_path)['md'].astype(np.float32)
#         mds = np.where(np.isnan(mds), 0, mds)
#         self.mds = torch.from_numpy(mds)
#         self.d_fps = self.fps.shape[1]
#         self.d_mds = self.mds.shape[1]        
        
#         self.smiles_list = smi_list
#         self.text_list = text_list

#     def __len__(self):
#         return len(self.smiles_list)
    
#     def __getitem__(self, idx):
#         # text = clip.tokenize(self.text_list[idx], truncate=True)
#         text = self.text_list[idx]
#         return self.smiles_list[idx], self.fps[idx], self.mds[idx], text



def clean_dataset(smiless, text_list):
    error_list = [6412, 6860, 9294, 9295, 9436, 9493, 14242, 14307, 15403, 15404, 18775, 19481, 19490, 19606, 19607, 19608, 19626, 19675, 19677, 19679, 19681]
    for index in sorted(error_list, reverse=True):
        del text_list[index]
        del smiless[index]
    return smiless, text_list
    
    
def read_smiles(path):
    data = pd.read_csv(path)
    smilist = data['canonical_smiles'].tolist()
    text = data['llm_answer'].tolist()
    return smilist, text

path_list = ["/home/jovyan/prompts_learning/12_2_data/chembl_try_mol_417_update_data_1251_llmv2_.csv",
             "/home/jovyan/prompts_learning/12_2_data/equal_1_v1.csv",
             "/home/jovyan/prompts_learning/12_2_data/equal_2_v1.csv"
            ]

smi_list = []
text_list = []
for path in path_list:
    smiles, text = read_smiles(path)
    smi_list += smiles
    text_list += text

smi_list, text_list = clean_dataset(smi_list, text_list)
    
print(f"Length of SMILES: {len(smi_list)}")
print(f"Length of text: {len(text_list)}")


class MoleculeTextDataset(Dataset):
    def __init__(self):
        fp_path = "/home/jovyan/prompts_learning/pretrain_data/12_2_data_rdkfp1-7_512.npz"
        md_path = "/home/jovyan/prompts_learning/pretrain_data/12_2_data_molecular_descriptors.npz"
        # with open(smiles_path, 'r') as f:
        #     lines = f.readlines()
        #     self.smiles_list = [line.strip('\n') for line in lines]
        self.fps = torch.from_numpy(sps.load_npz(fp_path).todense().astype(np.float32))
        mds = np.load(md_path)['md'].astype(np.float32)
        mds = np.where(np.isnan(mds), 0, mds)
        self.mds = torch.from_numpy(mds)
        self.d_fps = self.fps.shape[1]
        self.d_mds = self.mds.shape[1]        
        
        self.smiles_list = smi_list
        self.text_list = text_list

    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        # text = clip.tokenize(self.text_list[idx], truncate=True)
        text = self.text_list[idx]
        return self.smiles_list[idx], self.fps[idx], self.mds[idx], text

    
if __name__ == "__main__":
    config = config_dict['base']
    print(config)
    torch.backends.cudnn.benchmark = True
    # torch.cuda.set_device(local_rank)
    # torch.distributed.init_process_group(backend='nccl')
    # device = torch.device('cuda', local_rank)
    # set_random_seed(args.seed)
    # print(local_rank)
    
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    collator = Collator_pretrain(vocab, max_length=config['path_length'], n_virtual_nodes=2, candi_rate=config['candi_rate'], fp_disturb_rate=config['fp_disturb_rate'], md_disturb_rate=config['md_disturb_rate'])
    train_dataset = MoleculeTextDataset(smi_list, text_list)
    # train_loader = DataLoader(train_dataset, batch_size=config['batch_size']// 1, num_workers=16, drop_last=True, collate_fn=collator)
    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=16, drop_last=True, collate_fn=collator)
    model = LiGhT(
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
        input_drop=config['input_drop'],
        attn_drop=config['attn_drop'],
        feat_drop=config['feat_drop'],
        n_node_types=vocab.vocab_size
    ).to("cuda")
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load("/home/jovyan/prompts_learning/KPGT/src/models/base.pth").items()})
    print("Pre-trained weights of KPGT were loaded successfully!")
    
    device = "cuda"
    for b_id, batched_data in enumerate(train_loader):
        (smiles, batched_graph, fps, mds, sl_labels, disturbed_fps, disturbed_mds, text) = batched_data
        batched_graph = batched_graph.to(device)
        fps = fps.to(device)
        mds = mds.to(device)

        mol_fps_feat = model.generate_fps(batched_graph, fps, mds)
        print(mol_fps_feat.shape)
        print(text.shape)
        break
        
    
    print("okk")