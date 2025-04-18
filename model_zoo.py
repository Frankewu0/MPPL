import sys
from CLIP import clip
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

                
class MLP(nn.Module):
    def __init__(self, embed_dim=512, expand=4):
        super().__init__()
        
        self.fc1 = nn.Linear(embed_dim, embed_dim*expand)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(embed_dim*expand, embed_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    
class SEBlock(nn.Module):
    def __init__(self, num_feature, r):
        super().__init__()
        assert num_feature % r == 0, "the num_feature can not v dive by ratio."
        n = num_feature // r
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(num_feature, n, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n, num_feature, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(1, 2, 0)
        se = self.avgpool(x).flatten(1)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sig(se)
        se = x * se.unsqueeze(-1)
        se = se.permute(2, 0, 1)
        return se

    
class Image_Encoder(nn.Module):
    def __init__(self, clip_model, conf, pad_len):
        super().__init__()
        self.visual = clip_model.visual
        self.dtype = clip_model.dtype
        
        self.coor_linear = nn.Sequential(
            nn.Linear(conf*pad_len, 768),
            # nn.GELU(),
            # nn.Linear(768, 768)
        )
        # self.positional_embedding = nn.Parameter(torch.randn(200, 768))
    
    def pre_forward(self, x: torch.Tensor):
        # x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        # x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = self.coor_linear(x)
        # x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # x = torch.cat([self.visual.class_embedding.to(x.dtype) +\
        #     torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # x = x + self.visual.positional_embedding.to(x.dtype)
        # x = x + self.positional_embedding.to(x.dtype)
        # n = x.shape[0]
        
        x = self.visual.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.visual.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD

        # img_tokens = x[:, 1:, :]
        # x = self.visual.ln_post(x[:, 0, :])
        # x = x[:, 0, :]
        
        # if self.visual.proj is not None:
        #     x = x @ self.visual.proj
            
        return x
    
    def post_forward(self, x):
        x = x.permute(1, 0, 2)  # LND -> NLD

        # img_tokens = x[:, 1:, :]
        # x = self.visual.ln_post(x[:, 0, :])
        x = x[:, 0, :]
        
        if self.visual.proj is not None:
            x = x @ self.visual.proj
        return x
        

class Text_Encoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # print(tokenized_prompts.shape)
        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x.type(self.dtype)
        # x = self.ln_final(x) # w/o best
        
        # x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        x = x[torch.arange(x.shape[0]), tokenized_prompts] @ self.text_projection
        
        return x
    
    
class Prompt_processor(nn.Module):
    def __init__(self, model, initials=None):
        super(Prompt_processor,self).__init__()
        print("The initial prompts are:",initials)
        self.text_encoder = Text_Encoder(model)
        
        # if isinstance(initials,list):
        #     text = clip.tokenize(initials)
        #     self.embedding_prompt = nn.Parameter(model.token_embedding(text).requires_grad_())
        #     self.num_prompts = self.embedding_prompt.shape[0]
        if isinstance(initials,list):
            text = clip.tokenize(initials)
            self.tokenized_text = text
            # self.teacher_embedding_prompt = nn.Parameter(model.token_embedding(text[:16]), False)
            self.embedding_prompt = nn.Parameter(model.token_embedding(text).requires_grad_())
            self.num_prompts = self.embedding_prompt
        elif isinstance(initials,str):
            prompt_path=initials

            state_dict = torch.load(prompt_path)
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            self.embedding_prompt=nn.Parameter(new_state_dict['embedding_prompt']).cuda()
            self.embedding_prompt.requires_grad = True
        else:
            print([" ".join(["X"]*\
                16)," ".join(["X"]*16)])
            self.embedding_prompt=torch.nn.init.xavier_normal_(nn.Parameter(model.token_embedding([" ".join(["X"]*\
                16)," ".join(["X"]*16)]).requires_grad_())).cuda()
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        for i in range(len(self.text_encoder.transformer.resblocks)):
            self.text_encoder.transformer.resblocks[i].attn_mask = self.build_attention_mask()
            # self.text_encoder.transformer.resblocks[i].attn_mask = None
    
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(77, 77)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
        
    def forward(self):
        tokenized_prompts = torch.cat([p.argmax(dim=-1).unsqueeze(0) for p in self.tokenized_text], dim=0)
        text_features = self.text_encoder(self.embedding_prompt,tokenized_prompts)
        return text_features
        

class Atom_Encoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        # for pretrain single atom encoder
        # self.positional_embedding = nn.Parameter(torch.randn(150, 512))
        # self.positional_embedding = clip_model.positional_embedding
        # print(self.positional_embedding.shape)
        
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts):
        # print(tokenized_prompts.shape)
        x = prompts # + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x.type(self.dtype)
        x = self.ln_final(x)
        
        # x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        # x = x[torch.arange(x.shape[0]), tokenized_prompts] @ self.text_projection
        
        x = x[:, 0, :] @ self.text_projection
        
        return x
    
    
class AtomsProcessor(nn.Module):
    def __init__(self, model, pad_len=200, width=512):
        super(AtomsProcessor,self).__init__()
        self.pad_len = pad_len
        self.text_encoder = Atom_Encoder(model)
        self.atom_emb = nn.Embedding(300, 512)
        # nn.init.normal_(self.atom_emb.weight, std=0.02)
        # if self.atom_emb.padding_idx is not None:
        #     self.atom_emb.weight.data[self.atom_emb.padding_idx].zero_()
        # self.positional_embedding = nn.Parameter(torch.randn(201, 512))
        # self.class_embedding = nn.Parameter(torch.randn(width))
        
        for i in range(len(self.text_encoder.transformer.resblocks)):
            self.text_encoder.transformer.resblocks[i].attn_mask = self.build_attention_mask(pad_len)
            # self.text_encoder.transformer.resblocks[i].attn_mask = None
    
    def build_attention_mask(self, pad_len):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(pad_len, pad_len)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
        
    def pre_forward(self, emb_id):
        atom_features = self.atom_emb(emb_id)
        atom_features = atom_features.permute(1, 0, 2)  # NLD -> LND
        return atom_features
    
    def post_forward(self, x):
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x.type(self.text_encoder.dtype)
        x = self.text_encoder.ln_final(x)
        
        # x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        # x = x[torch.arange(x.shape[0]), tokenized_prompts] @ self.text_projection
        
        x = x[:, 0, :] @ self.text_encoder.text_projection
        return x

    
class QFormer_Layer(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        # self.sa = MHSA(embed_dim, 8)
        self.sa = nn.MultiheadAttention(embed_dim, 1, batch_first=False)
        
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_fi = nn.LayerNorm(embed_dim)
        # self.cross_a = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
        self.cross_a = nn.MultiheadAttention(embed_dim, 1, batch_first=False)
        
        self.ln3 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)
    
    def forward(self, x, queries):
        
        residual = queries
        queries = self.ln1(queries)
        queries, _ = self.sa(queries, queries, queries)
        residual = queries + residual
        
        queries = self.ln_q(residual)
        x = self.ln_fi(x)
        cross_queries_residual, _ = self.cross_a(queries, x, x)
        cross_queries_residual = cross_queries_residual + residual
        
        queries = self.ln3(cross_queries_residual)
        queries = self.mlp(queries) + cross_queries_residual
        return queries
        

q_len = 30
# Modified by Text Encoder
class Atom_Attn_Encoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        
        self.q_len = q_len+2
        zeros = torch.zeros(size=[77-self.q_len], dtype=torch.long)
        self.zeros = nn.Parameter(clip_model.token_embedding(zeros), False)
        

    def forward(self, prompts):
        # print(tokenized_prompts.shape)
        zeros = self.zeros.expand(prompts.shape[0], -1, -1).to(prompts.device)
        prompts = torch.cat([prompts, zeros], dim=1)

        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x.type(self.dtype)
        # x = self.ln_final(x)
        
        
        tokenized_prompts = torch.full(size=[prompts.shape[0]], fill_value=self.q_len-1, dtype=torch.long) # n_q 18
        x = x[torch.arange(x.shape[0]), tokenized_prompts] @ self.text_projection
        
        return x
        

class QFormer(nn.Module):
    def __init__(self, n_q=q_len+2, embed_dim=512, n_layer=1): # n_q 18=16+2 best
        super().__init__()
        self.proj = nn.Linear(768, 512)
        
        self.query = nn.Parameter(torch.randn([n_q, embed_dim]), True)
        nn.init.normal_(self.query) # best
        
        self.layer = nn.ModuleList()
        for i in range(n_layer):
            self.layer.append(QFormer_Layer(embed_dim))
    
    def forward(self, atom_feat):
        atom_feat = self.proj(atom_feat)
        atom_feat = atom_feat.permute(1, 0, 2)
        n = atom_feat.shape[0]
        queries = self.query.expand(n, -1, -1)
        for i in range(len(self.layer)):
            queries = self.layer[i](atom_feat, queries)
        return queries 

    
class MHSA(nn.Module):
    def __init__(self, input_dim=512, heads=8):
        super().__init__()
        # self.ln = nn.LayerNorm(input_dim)
        self.mhsa = nn.MultiheadAttention(embed_dim=input_dim, num_heads=heads, batch_first=True)
        # self.mhsa = nn.MultiheadAttention(embed_dim=input_dim, num_heads=heads, batch_first=False)
    
    def forward(self, x):
        # x = self.ln(x)
        x, _ = self.mhsa(x, x, x, need_weights=False)
        return x


class Normalized_Linear(nn.Module):
    def __init__(self, in_feat=768, out_feat=1, bias=True) -> None:
        super().__init__()
        # self.w = nn.Parameter(torch.randn(out_feat, in_feat))
        # self.b = nn.Parameter(torch.zeros(out_feat))
        
        linear = nn.Linear(in_features=in_feat, out_features=out_feat)
        self.w = nn.Parameter(linear.weight, True)
        self.b = nn.Parameter(linear.bias, True)
        
        # nn.init.kaiming_normal_(self.w, a=0, mode='fan_in', nonlinearity='leaky_relu')
        # nn.init.xavier_normal_(self.w, gain=)
        # nn.init.normal_(self.w, std=0.02)
    
    def forward(self, x):
        w = self.w
        b = self.b
        x = F.linear(x, w, b)
        return x


class AttentionPool1d(nn.Module):
    def __init__(self, seq_len: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(seq_len + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        # x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        # x = x.permute(1, 0, 2)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=True
        )
        return x[0, ...]  

    
# class EnergyHead(nn.Module):
#     def __init__(
#         self,
#         input_dim,
#         output_dim,
#     ):
#         super().__init__()
#         self.layer_norm = nn.LayerNorm(input_dim)
#         self.linear_in = nn.Linear(input_dim, input_dim)
        
#         self.linear_out = nn.Linear(input_dim, output_dim)
#         # self.drop = nn.Dropout(0.05)

#     def forward(self, x):
#         x = x.type(self.linear_in.weight.dtype)
#         # x = self.drop(x)
        
#         x = F.gelu(self.linear_in(x))
#         # recurrent head, better performance for 2024-10-18 model
#         x = self.linear_in(x)
#         x = self.layer_norm(x)
#         x = F.gelu(x)
        
#         # x = self.drop(x)
#         x = self.linear_out(x)
#         return x
    
    
class EnergyHead(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
    ):
        super().__init__()
        self.linear_in = nn.Linear(input_dim, input_dim)
        self.linear_out = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.type(self.linear_in.weight.dtype)
        x = F.gelu(self.linear_in(x))
        x = self.linear_out(x)
        return x


class Graph_Conv_Block(nn.Module):
    def __init__(self, feats=768, hidden=768, norm=False):
        super().__init__()
        self.norm = norm
        if norm:
            self.ln1 = nn.LayerNorm(feats)
        self.in_proj = nn.Linear(feats, hidden)
        self.out_proj = nn.Linear(hidden, feats)
    
    def forward(self, x, edge):
        if self.norm:
            x = self.ln1(x)
        
        # x: LBD; edge: LBL -> res: LBD
        x = x.permute(1, 2, 0) # BDL
        edge = edge.permute(1, 2, 0) # for matrix muliplication
        x = torch.bmm(x, edge) # BDL
        x = x.permute(2, 0, 1) # BLD
        x = self.in_proj(x)
        x = F.gelu(x)
        x = self.out_proj(x)
        
        return x


# class Molecular_prompts_Interaction_block(nn.Module):
#      def __init__(self, featm, featp, layers=2):
    
        
class Equivariant_FFN(nn.Module):
    def __init__(self, feat1, feat2, norm=True):
        super().__init__()
        self.norm = norm
        if norm:
            self.ln1 = nn.LayerNorm(feat1)
            self.ln2 = nn.LayerNorm(feat2)
        self.in_proj1 = nn.Linear(feat1, feat1)
        self.in_proj2 = nn.Linear(feat2, feat1)
        self.out_proj = nn.Linear(feat1, feat1)
        
        
    def forward(self, x1, x2):
        if self.norm:
            x1 = self.ln1(x1)
            x2 = self.ln2(x2)
        x1 = self.in_proj1(x1)
        
        x2 = self.in_proj2(x2)
        x2 = F.gelu(x2)
        
        x = x1 + x1 * torch.mean(x2, dim=0, keepdim=True)
        x = self.out_proj(x)
        return x

    
class Interaction_Module(nn.Module):
    def __init__(self, feat1, feat2, norm=False):
        super().__init__()    
        # self.in_proj_1_2 = nn.Linear(feat2, feat1)
        # self.in_proj_2_1 = nn.Linear(feat1, feat2)
        
        self.norm = norm
        if norm:
            self.ln1 = nn.LayerNorm(feat1)
            self.ln2 = nn.LayerNorm(feat2)
        self.cross_attn_1_2 = nn.MultiheadAttention(embed_dim=feat1, num_heads=1, batch_first=False)
        self.cross_attn_2_1 = nn.MultiheadAttention(embed_dim=feat2, num_heads=1, batch_first=False)
        self.e_ffn_1_2 = Equivariant_FFN(feat1, feat2, norm)
        self.e_ffn_2_1 = Equivariant_FFN(feat2, feat1, norm)
        
    def forward(self, x1, x2):
        # x1_2_1 = self.in_proj_2_1(x1)
        # x2_1_2 = self.in_proj_1_2(x2)
        if self.norm:
            x1 = self.ln1(x1)
            x2 = self.ln2(x2)
        # x1_attn, _ = self.cross_attn_1_2(x1, x2_1_2, x2_1_2)
        # x2_attn, _ = self.cross_attn_2_1(x2, x1_2_1, x1_2_1)
        x1_attn, _ = self.cross_attn_1_2(x1, x2, x2)
        x2_attn, _ = self.cross_attn_2_1(x2, x1, x1)
        x1 = x1 + x1_attn
        x2 = x2 + x2_attn
        # x1 = x1_attn
        # x2 = x2_attn
        
        # print(x1.shape, x2.shape)
        x1_ffn = self.e_ffn_1_2(x1, x2)
        x2_ffn = self.e_ffn_2_1(x2, x1)
        x1 = x1 + x1_ffn
        x2 = x2 + x2_ffn
        # x1 = x1_ffn
        # x2 = x2_ffn
        return x1, x2
        
    
class Atom_Coordination_Fusion_Block(nn.Module):
    def __init__(self, pad_len):
        super().__init__()
        
        # current best fusion
        self.coor_proj = nn.Linear(512+768, 768)
        self.atom_proj = nn.Linear(512+768, 512)
        self.coor_se = SEBlock(768, 4)
        self.atom_se = SEBlock(512, 4)
        
        # self.interaction_module = Interaction_Module(768, 512, norm=True)
        
        self.spd_proj = nn.Sequential(
                    nn.Linear(pad_len, 768),
                    # nn.GELU(),
                    # nn.Linear(768, 768),
                )
        self.edge_proj = nn.Sequential(
                    nn.Linear(pad_len, 768),
                    # nn.GELU(),
                    # nn.Linear(768, 768),
                )
        
        self.gra_proj = nn.Linear(512+768, 768) # best
        self.gragh_conv = Graph_Conv_Block(feats=768, hidden=768, norm=True) # best
        

    
    def single_attn_forward(self, block, x: torch.Tensor):
        x = x + block.attention(block.ln_1(x))
        return x
        
    def single_mlp_forward(self, block, x: torch.Tensor):
        x = x + block.mlp(block.ln_2(x))
        return x
    
    def forward(self, atom_block, coor_block, atom_feat, coor_feat, spd, edge):
        spd = spd.permute(1, 0, 2)
        edge = edge.permute(1, 0, 2)
        
        spd_emb = self.spd_proj(spd)
        edge_emb = self.edge_proj(edge)
        
        # atom_feat = self.atom_gragh_conv(atom_feat, edge)
        # coor_feat = self.gragh_conv(coor_feat, edge) 
        fuse_feat = torch.cat([atom_feat, coor_feat], dim=2) 
        coor_feat = self.gragh_conv(self.gra_proj(fuse_feat), edge) # best
        
        atom_attn = self.single_attn_forward(atom_block, atom_feat)
        coor_attn = self.single_attn_forward(coor_block, coor_feat)
        
        # Fusion/Interaction part
        fuse_feat = torch.cat([atom_feat, coor_feat], dim=2) # best
        coor_bias = self.coor_se(self.coor_proj(fuse_feat) + spd_emb + edge_emb) # best
        atom_bias = self.atom_se(self.atom_proj(fuse_feat))
        
        # x: LBD
        # coor_bias, atom_bias = self.interaction_module(coor_feat, atom_feat) 
        
        atom_feat = self.single_mlp_forward(atom_block, atom_attn+atom_bias)
        coor_feat = self.single_mlp_forward(coor_block, coor_attn+coor_bias) 
        
        return atom_feat, coor_feat
    

class Multiview_QFormer_Layer(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_q2 = nn.LayerNorm(embed_dim)
        # self.sa = MHSA(embed_dim, 8)
        self.sa_q = nn.MultiheadAttention(embed_dim, 1, batch_first=False)
        
        self.ln_mol = nn.LayerNorm(embed_dim)
        # self.cross_a = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
        self.cross_a = nn.MultiheadAttention(embed_dim, 1, batch_first=False)
        
        self.sa_a = nn.MultiheadAttention(embed_dim, 1, batch_first=False)
        
        self.ln_q3 = nn.LayerNorm(embed_dim)
        self.ln_p = nn.LayerNorm(embed_dim)
        self.ffn_mol = MLP(embed_dim)
        self.ffn_pro = MLP(embed_dim)
    
    def forward(self, queries, mol, prompts):
        residual = queries
        queries = self.ln_q(queries)
        queries, _ = self.sa_q(queries, queries, queries)
        residual = queries + residual
        
        # mol = self.ln_mol(mol)
        queries = self.ln_q2(queries)
        cross_queries_residual, _ = self.cross_a(queries, mol, mol)
        cross_queries_residual = cross_queries_residual + residual
        
        all_feats = torch.cat([prompts, mol], dim=0)
        all_feats, _ = self.sa_a(all_feats, all_feats, all_feats)
        queries = all_feats[cross_queries_residual.shape[0]:]
        prompts = all_feats[:cross_queries_residual.shape[0]]
        
        # queries = self.ffn_mol(queries) + queries
        # prompts = self.ffn_pro(prompts) + prompts
        queries = self.ffn_mol(self.ffn_mol(queries)) + queries
        prompts = self.ffn_pro(self.ffn_pro(prompts)) + prompts
        return prompts

    
class Protein_Prompts_Interatctoin_Module(nn.Module):
    def __init__(self, num_layer=1):
        super().__init__()
        self.num_layer = num_layer
        
        # self.query = nn.Parameter(torch.randn([50, 1, 512]), True)
        # nn.init.normal_(self.query) # best

        self.model = nn.ModuleList()
        for i in range(self.num_layer):
            self.model.append(
                nn.TransformerEncoderLayer(d_model=512, 
                                                # nhead=8, 
                                                   nhead=1, # best
                                                activation=F.gelu,
                                               batch_first=False,
                                               norm_first=False #best
                                          # norm_first=True
                                            )
                # Multiview_QFormer_Layer(512)
            )
            
        self.label_emb = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.GELU(),
                    nn.Linear(256, 512)
                )
        
        self.atom_emb = nn.Sequential(
                    nn.Linear(768, 512),
                    nn.GELU(),
                    nn.Linear(512, 512)
  
                    
                )
        

        self.attr_heads = nn.ModuleList()
        self.attr_num = 6
        for i in range(self.attr_num):
            self.attr_heads.append(EnergyHead(512, 1))

        
    
    def forward(self, mol_feats, prompts):
        prompts = prompts.unsqueeze(1).repeat(1, mol_feats.shape[1], 1)
        prompts = self.label_emb(prompts)
        
        mol_feats = self.atom_emb(mol_feats)
        # mol_feats = self.atom_emb(torch.cat([at, mol_feats], dim=2))
        
#         n = mol_feats.shape[1]
#         queries = self.query.repeat(1, n, 1)
#         for i in range(self.num_layer):
#             feats = self.model[i](queries, mol_feats, prompts)
            
#         cls_token = feats[0, :, :]
#         attr_pre_list = []
#         attr_prompts = feats[1:self.attr_num+1, :, :]
#         for i in range(self.attr_num):
#             attr_pred = self.attr_heads[i](attr_prompts[i, :, :])
#             attr_pre_list.append(attr_pred)
#         return cls_token, attr_pre_list
        
        feats = torch.cat([prompts, mol_feats], dim=0)
        for i in range(self.num_layer):
            feats = self.model[i](feats) # TransformerEncoderLayer
        cls_token = feats[0, :, :]
        attr_pre_list = []
        attr_prompts = feats[1:self.attr_num+1, :, :]
        for i in range(self.attr_num):
            attr_pred = self.attr_heads[i](attr_prompts[i, :, :])
            attr_pre_list.append(attr_pred)
        return cls_token, attr_pre_list


        
            
class CLIP_Protein(nn.Module):
    def __init__(self, model, conf, pad_len=200):
        super(CLIP_Protein, self).__init__()
        
        sd = torch.load("/home/jovyan/prompts_learning/trained_weight/A_10_17_cliPM_pre_stage1_Epoch100.pth")
        
        self.atom_encoder = AtomsProcessor(model, pad_len=pad_len).cuda()
        # set_requires_grad(self.atom_encoder.text_encoder, False)
        self.atom_encoder.load_state_dict(sd["atom_encoder"])
        
        self.coor_encoder = Image_Encoder(model, conf, pad_len)
        # set_requires_grad(self.coor_encoder.visual, False)
        self.coor_encoder.load_state_dict(sd["coor_encoder"])
        
        self.fusion_blocks = nn.ModuleList()
        self.num_blocks = 12  # 12 best
        print(f"Layer number of backbone is {self.num_blocks}.")
        for i in range(self.num_blocks-1):
            self.fusion_blocks.append(Atom_Coordination_Fusion_Block(pad_len))
        self.fusion_blocks.load_state_dict(sd["fusion_blocks"])
        
        
        from copy import deepcopy
        model = deepcopy(model)
        self.prompts_processor = Prompt_processor(model,
            [
            # Best
            # "The protein has the blood-brain barrier.",
                
            # "The protein has the blood-brain barrier.",
            "The molecule has the X X X X X",
            "The LogD of the molecule is A A A A",
            "The LogP of the molecule is B B B B",
            "The pKa of the molecule is C C C C",
            "The pKb of the molecule is D D D D",
            "The LogSol of the molecule is E E E E",
            "The wLogSol of the molecule is F F F F",
            ]).cuda()
        set_requires_grad(self.prompts_processor.text_encoder, False)
        self.ppim = Protein_Prompts_Interatctoin_Module()
        
        self.head = EnergyHead(512, 1)

    def forward(self, atom_rep, pair_rep, spd, edge):
        atom_feat = self.atom_encoder.pre_forward(atom_rep)  # [NBD] 512
        pair_feat = self.coor_encoder.pre_forward(pair_rep)  # [NBD] 768
        
        
        for i in range(self.num_blocks-1):
            atom_block = self.atom_encoder.text_encoder.transformer.resblocks[i]
            coor_block = self.coor_encoder.visual.transformer.resblocks[i]
            atom_feat, pair_feat = self.fusion_blocks[i](atom_block, coor_block, atom_feat, pair_feat, spd, edge)
        
        # atom_feat = self.atom_encoder.text_encoder.transformer.resblocks[self.num_blocks-1](atom_feat)
        pair_feat = self.coor_encoder.visual.transformer.resblocks[self.num_blocks-1](pair_feat)
        prompts = self.prompts_processor()
        feat, attr_pre_list = self.ppim(pair_feat, prompts)
        
        p = self.head(feat)
        
        if self.training:
            # atom_feat = self.atom_encoder.text_encoder.transformer.resblocks[self.num_blocks-1](atom_feat)
            # loss_con = self.contrastive(atom_feat, pair_feat)
            return p, attr_pre_list # , loss_con
        else:
            return p
        

class CLIPM_Stage1(nn.Module):
    def __init__(self, model, conf, pad_len=200):
        super().__init__()
        self.atom_encoder = AtomsProcessor(model, pad_len=pad_len).cuda()
        self.coor_encoder = Image_Encoder(model, conf, pad_len)
        
        self.fusion_blocks = nn.ModuleList()
        self.num_blocks = 12  # 12 best
        print(f"Layer number of backbone is {self.num_blocks}.")
        for i in range(self.num_blocks-1):
            self.fusion_blocks.append(Atom_Coordination_Fusion_Block(pad_len))
        
        self.pretrain_head1 = EnergyHead(512, 21)
        self.pretrain_head2 = nn.Sequential(
            # nn.Linear(768, 1500),
            # nn.LayerNorm(768),
            EnergyHead(768, 30)
        )

    def forward(self, atom_rep, pair_rep, spd, edge, mk_id):
        atom_feat = self.atom_encoder.pre_forward(atom_rep)  # [NBD] 512
        pair_feat = self.coor_encoder.pre_forward(pair_rep)  # [NBD] 768
        
        for i in range(self.num_blocks-1):
            atom_block = self.atom_encoder.text_encoder.transformer.resblocks[i]
            coor_block = self.coor_encoder.visual.transformer.resblocks[i]
            atom_feat, pair_feat = self.fusion_blocks[i](atom_block, coor_block, atom_feat, pair_feat, spd, edge)
        
        atom_feat = self.atom_encoder.text_encoder.transformer.resblocks[self.num_blocks-1](atom_feat)
        pair_feat = self.coor_encoder.visual.transformer.resblocks[self.num_blocks-1](pair_feat)
        
        # print(atom_feat.shape, mk_id.shape, atom_feat[mk_id.permute(1, 0), :].shape)
        atom_feat = atom_feat.permute(1, 0, 2)
        atom_feat = torch.gather(atom_feat, dim=1, index=mk_id.unsqueeze(-1).expand(-1,-1,atom_feat.shape[2]))
        loss_1 = self.pretrain_head1(atom_feat)
        loss_2 = self.pretrain_head2(pair_feat)
        
        return loss_1, loss_2

    
###########################################################################  
###########################################################################
###########################################################################

    
class Moltext_Encoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        # self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        # self.text_projection = clip_model.text_projection
        self.text_projection = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.GELU(),
                    nn.Linear(512, 512)
                )
        self.dtype = clip_model.dtype
        
        scale = 512 ** -0.5
        self.mol_class_embedding = nn.Parameter(scale*torch.randn(512))
        self.positional_embedding = nn.Parameter(torch.randn(78, 512))
        nn.init.normal_(self.positional_embedding, std=0.01)

    def forward(self, x):
        x = torch.cat([self.mol_class_embedding.to(x.dtype) +\
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x.type(self.dtype)
        x = self.ln_final(x[:, 0, :]) 
        # x = x[:, 0, :]
        
        # x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        # x = x[torch.arange(x.shape[0]), tokenized_prompts] @ self.text_projection
        # x = x @ self.text_projection
        x = self.text_projection(x)
        return x

class Pretrain_text_encoder_processor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.text_encoder = Moltext_Encoder(model)
        self.token_embedding = model.token_embedding        
        
        for i in range(len(self.text_encoder.transformer.resblocks)):
            self.text_encoder.transformer.resblocks[i].attn_mask = self.build_attention_mask()
            # self.text_encoder.transformer.resblocks[i].attn_mask = None
    
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(78, 78)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
        
    def forward(self, tokenized_text):
        text_emb = self.token_embedding(tokenized_text)
        text_features = self.text_encoder(text_emb)
        return text_features
    

class Pretrain_SCIBert_processor(nn.Module):
    def __init__(self):
        super().__init__()   
        

        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.text_encoder = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        # from transformers import BertTokenizer, BertModel
        # self.tokenizer = BertTokenizer.from_pretrained('/home/liu/.cache/huggingface/transformers/scibert_scivocab_uncased')
        # self.text_encoder = BertModel.from_pretrained('/home/liu/.cache/huggingface/transformers/scibert_scivocab_uncased')
        # set_requires_grad(self.text_encoder, False)
        
        self.tex_pad_token = nn.Parameter(torch.randn(768))
        self.model = nn.ModuleList()
        for i in range(2):
            self.model.append(
                nn.TransformerEncoderLayer(d_model=768, 
                                                nhead=8, 
                                                activation=F.gelu,
                                               batch_first=True,
                                          norm_first=True,
                                            )
            )
        self.positional_embedding = nn.Parameter(torch.randn(77, 768)) 
        self.text_proj = nn.Sequential(
                    nn.Linear(768, 768),
                    nn.GELU(),
                    nn.Linear(768, 768)
                )
        
        
    def forward(self, batch_sentences):
        tokenized_text = self.tokenizer(batch_sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")
        for k in tokenized_text.keys():
             tokenized_text[k] = tokenized_text[k].cuda()
        text_features = self.text_encoder(**tokenized_text) # , output_hidden_states=True
        # all_tokens, _ = text_features 
        all_tokens = text_features['last_hidden_state'] # AutoModel

        if all_tokens.shape[1] > 77:
            all_tokens = all_tokens[:, :77, :] + self.positional_embedding
        else:
            pad_len = 77 - all_tokens.shape[1]
            pad_token = self.tex_pad_token.unsqueeze(0).repeat(all_tokens.shape[0], pad_len, 1).cuda()
            all_tokens = torch.cat([all_tokens, pad_token], dim=1) + self.positional_embedding
            
        # cls_tokens = text_features['pooler_output']
        
        for blk in self.model:
            all_tokens = blk(all_tokens) # TransformerEncoderLayer
        text_feat = self.text_proj(all_tokens[:, 0, :])
        return text_feat
        # print(a.shape, b.shape)
        # return text_features
    
    
class Mol_Encoder(nn.Module):
    def __init__(self, kpgt):
        super().__init__()
        self.mol_encoder = kpgt
        # set_requires_grad(self.mol_encoder, False)
        
        self.model = nn.ModuleList()
        for i in range(2):
            self.model.append(
                nn.TransformerEncoderLayer(d_model=768, 
                                                nhead=12, 
                                                # nhead=1, 
                                                activation=F.gelu,
                                               batch_first=True,
                                               # norm_first=False #best
                                          norm_first=True,
                                            )
            )
        scale = 768 ** -0.5
        # self.mol_class_embedding = nn.Parameter(scale*torch.randn(768))
        self.positional_embedding = nn.Parameter(torch.randn(32, 768))     
            
        
        # self.final_ln = nn.LayerNorm(768)
        self.mol_proj = nn.Sequential(
                    nn.Linear(768*2, 768),
                    nn.GELU(),
                    nn.Linear(768, 768)
                )
        
        
    def forward(self, mol):
        batched_graph, fps, mds = mol
        # mol_features = self.mol_encoder.generate_fps(batched_graph, fps, mds)
        # mol_features = self.mol_encoder.get_feats(batched_graph, fps, mds)
        readout, fp, md = self.mol_encoder.get_feats(batched_graph, fps, mds)
        mol_features = torch.cat([fp, md, readout], dim=1) + self.positional_embedding
        # mol_features = torch.cat([self.mol_class_embedding.to(mol_features.dtype) +\
        #     torch.zeros(mol_features.shape[0], 1, mol_features.shape[-1], dtype=mol_features.dtype, device=mol_features.device), mol_features], dim=1)
        for blk in self.model:
            mol_features = blk(mol_features) # TransformerEncoderLayer
        
        fp = mol_features[:, 0, :]
        md = mol_features[:, 1, :]
        mol_features = torch.cat([fp, md], dim=-1)
        
        # mol_features = self.final_ln(mol_features)
        mol_features = self.mol_proj(mol_features)
        return mol_features
        

class MMPL_Pretrain_Model(nn.Module):
    def __init__(self, kpgt):
        super().__init__()
        # self.text_proc = Pretrain_text_encoder_processor(model).cuda()
        self.text_proc = Pretrain_SCIBert_processor().cuda()
        self.mol_encoder = Mol_Encoder(kpgt).cuda()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, mol, text):
        mol_features = self.mol_encoder(mol)
        text_features = self.text_proc(text)

        # normalized features
        mol_features = mol_features / mol_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_mol = logit_scale * mol_features @ text_features.t()
        # logits_per_mol = 1. * mol_features @ text_features.t()
        logits_per_text = logits_per_mol.t()
        
        n = mol_features.shape[0]
        labels = torch.arange(n).cuda() # .unsqueeze(0).repeat(feata.shape[1], 1).cuda()
        loss_m = F.cross_entropy(logits_per_mol, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_m + loss_t) / 2.
        return loss

        # # shape = [global_batch_size, global_batch_size]
        # return logits_per_mol, logits_per_text
        

############################################################        
# # #                Finetune Model Part               # # #
############################################################ 


class MolPrompts_SCIBert_processor(nn.Module):
    def __init__(self, initials):
        super().__init__()   
        
        from transformers import BertTokenizer, BertModel
        self.tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.text_encoder = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
        set_requires_grad(self.text_encoder.encoder, False)
        set_requires_grad(self.text_encoder.pooler, False)
        
        
        self.tex_pad_token = nn.Parameter(torch.randn(768))
        self.tex_pad_token.requires_grad = False
        
        self.model = nn.ModuleList()
        for i in range(2):
            self.model.append(
                nn.TransformerEncoderLayer(d_model=768, 
                                                nhead=8, 
                                                activation=F.gelu,
                                               batch_first=True,
                                          norm_first=True,
                                          dropout=0.0
                                           # dropout=0.1
                                            )
            )
        set_requires_grad(self.model, False)
        
        self.positional_embedding = nn.Parameter(torch.randn(77, 768)) 
        self.positional_embedding.requires_grad = False
        
        self.text_proj = nn.Sequential(
                    nn.Linear(768, 768),
                    nn.GELU(),
                    nn.Linear(768, 768)
                )
        set_requires_grad(self.text_proj, False)
        
        
        self.initials = initials
        
            
    def init_learnable_p(self):
        if isinstance(self.initials,list):
            self.p = self.tokenizer(self.initials, padding=True, truncation=True, max_length=512, return_tensors="pt")
            self.p['input_ids'] = self.p['input_ids'].cuda()
            self.p['token_type_ids'] = self.p['token_type_ids'].cuda()
            self.p['attention_mask'] = self.p['attention_mask'].cuda()
            
        # token_embedding = self.text_encoder.get_input_embeddings()
        # self.embedding_prompt = nn.Parameter(token_embedding(self.tokenized_text).requires_grad_()).cuda()
        
    # def forward(self):
    #     text_features = self.text_encoder(**self.p)
    #     # text_features, cls_tokens = text_features
    #     all_tokens = text_features['last_hidden_state']
    #     cls_tokens = text_features['pooler_output']
    #     return cls_tokens
        

    def forward(self):
        text_features = self.text_encoder(**self.p) # , output_hidden_states=True
        # text_features, cls_tokens = text_features # AutoModel
        # all_tokens, _ = text_features 
        all_tokens = text_features['last_hidden_state']

        if all_tokens.shape[1] > 77:
            all_tokens = all_tokens[:, :77, :] + self.positional_embedding
        else:
            pad_len = 77 - all_tokens.shape[1]
            pad_token = self.tex_pad_token.unsqueeze(0).repeat(all_tokens.shape[0], pad_len, 1).cuda()
            all_tokens = torch.cat([all_tokens, pad_token], dim=1) + self.positional_embedding
            
        # cls_tokens = text_features['pooler_output']
        
        for blk in self.model:
            all_tokens = blk(all_tokens) # TransformerEncoderLayer
        text_feat = self.text_proj(all_tokens[:, 0, :])
        return text_feat
    
    def edit_forward(self, text):
        text_emb = self.tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors="pt")
        text_emb['input_ids'] = text_emb['input_ids'].cuda()
        text_emb['token_type_ids'] = text_emb['token_type_ids'].cuda()
        text_emb['attention_mask'] = text_emb['attention_mask'].cuda()
        
        text_features = self.text_encoder(**text_emb)
        all_tokens = text_features['last_hidden_state']

        if all_tokens.shape[1] > 77:
            all_tokens = all_tokens[:, :77, :] + self.positional_embedding
        else:
            pad_len = 77 - all_tokens.shape[1]
            pad_token = self.tex_pad_token.unsqueeze(0).repeat(all_tokens.shape[0], pad_len, 1).cuda()
            all_tokens = torch.cat([all_tokens, pad_token], dim=1) + self.positional_embedding
        
        for blk in self.model:
            all_tokens = blk(all_tokens) # TransformerEncoderLayer
        text_feat = self.text_proj(all_tokens[:, 0, :])
        return text_feat

        
class MolPrompts_processor(nn.Module):
    def __init__(self, model, initials):
        super().__init__()
        self.text_encoder = Moltext_Encoder(model)
        self.token_embedding = model.token_embedding
        
        print("The initial prompts are:",initials)
        
        if isinstance(initials,list):
            text = clip.tokenize(initials)
            self.tokenized_text = text
        
        for i in range(len(self.text_encoder.transformer.resblocks)):
            self.text_encoder.transformer.resblocks[i].attn_mask = self.build_attention_mask()
            # self.text_encoder.transformer.resblocks[i].attn_mask = None
    def init_learnable_p(self):
        self.embedding_prompt = nn.Parameter(self.token_embedding(self.tokenized_text.cuda()).requires_grad_())
    
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(78, 78)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
        
    def forward(self):
        text_features = self.text_encoder(self.embedding_prompt)
        return text_features
    
    
class Mol_Encoder_tune(nn.Module):
    def __init__(self, kpgt):
        super().__init__()
        self.mol_encoder = kpgt
        self.model = nn.ModuleList()
        for i in range(2):
            self.model.append(
                nn.TransformerEncoderLayer(d_model=768, 
                                                nhead=12, 
                                                # nhead=1, 
                                                activation=F.gelu,
                                               batch_first=False,
                                               # norm_first=False #best
                                          norm_first=True,
                                          dropout=0.0,
                                           # dropout=0.1
                                            )
            )
        scale = 768 ** -0.5
        # self.mol_class_embedding = nn.Parameter(scale*torch.randn(768))
        self.positional_embedding = nn.Parameter(torch.randn(32, 768))     
            
        
        # self.final_ln = nn.LayerNorm(768)
        self.mol_proj = nn.Sequential(
                    nn.Linear(768*2, 768),
                    nn.GELU(),
                    nn.Linear(768, 768)
                )
        
    def forward(self, mol):
        batched_graph, fps, mds = mol
        # mol_features = self.mol_encoder.get_feats(batched_graph, fps, mds)
        # mol_features = self.mol_encoder.generate_fps(batched_graph, fps, mds)
        # mol_features = self.final_ln(mol_features)
        # mol_features = self.mol_proj(mol_features)
        # print(mol_features.shape)
        readout, fp, md = self.mol_encoder.get_feats(batched_graph, fps, mds)
        
        mol_features = torch.cat([fp, md, readout], dim=1) + self.positional_embedding
        for blk in self.model:
            mol_features = blk(mol_features) # TransformerEncoderLayer
            
        return mol_features, fp, md
    
    def edit_forward(self, mol):
        batched_graph, fps, mds = mol
        # mol_features = self.mol_encoder.generate_fps(batched_graph, fps, mds)
        # mol_features = self.mol_encoder.get_feats(batched_graph, fps, mds)
        readout, fp, md = self.mol_encoder.get_feats(batched_graph, fps, mds)
        mol_features = torch.cat([fp, md, readout], dim=1) + self.positional_embedding
        # mol_features = torch.cat([self.mol_class_embedding.to(mol_features.dtype) +\
        #     torch.zeros(mol_features.shape[0], 1, mol_features.shape[-1], dtype=mol_features.dtype, device=mol_features.device), mol_features], dim=1)
        for blk in self.model:
            mol_features = blk(mol_features) # TransformerEncoderLayer
        
        fp = mol_features[:, 0, :]
        md = mol_features[:, 1, :]
        mol_features = torch.cat([fp, md], dim=-1)
        
        # mol_features = self.final_ln(mol_features)
        mol_features = self.mol_proj(mol_features)
        return mol_features


class Cross_Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.ln1 = nn.LayerNorm(dim)
        # self.lnq = nn.LayerNorm(dim)
        # self.query = nn.Parameter(torch.randn([8, 1, 768]), True)
        # nn.init.normal_(self.query, std=0.02)
      
        self.cross_a = nn.MultiheadAttention(dim, 1, batch_first=False)
        
        # self.ln2 = nn.LayerNorm(dim)
        self.ffn = MLP(dim)
        
    def forward(self, q, kv):
        # q = self.lnq(q)
        # kv = self.ln1(kv)
        
        # bz = q.shape[1]
        # q_b = self.query.repeat(1, bz, 1)
        # q = q + q_b
        # q = self.lnq(q)
        
        k = kv
        v = kv
        res, _ = self.cross_a(q, k, v)
        res_ = res + q
        
        # res = self.ln2(res_)
        res = self.ffn(res) + res_
        # return res
        return torch.tanh(res) # best
        # return torch.sigmoid(res)

        
class Moltex_Attention(nn.Module):
    def __init__(self, dim, n_cls):
        super().__init__()
        
        # self.query = nn.Parameter(torch.randn([1, 1, 1536]), True)
        self.query = nn.Parameter(torch.randn([n_cls, 1, 1536]), True) # for mix
        nn.init.normal_(self.query, std=1.) # best
        
        self.in_proj = nn.Sequential(
            nn.Linear(1536, 768),
            nn.GELU(),
            nn.Linear(768, 768)
        )
        
        # self.ln1 = nn.LayerNorm(dim)
        self.lnq = nn.LayerNorm(dim)
        self.cross_a = nn.MultiheadAttention(dim, 1, batch_first=False)
        
        # self.ln2 = nn.LayerNorm(dim)
        self.ffn = MLP(dim)
        
        self.out_proj = nn.Sequential(
            nn.Linear(768, 1536),
            nn.GELU(),
            nn.Linear(1536, 1536)
        )
        
    def forward(self, kv):
        k = kv
        v = kv
        bz = v.shape[1]
        q = self.query.repeat(1, bz, 1).to(k.device)
        q = self.in_proj(q)
        q = self.lnq(q)
        
        res, _ = self.cross_a(q, k, v)
        res_ = res # + q
        
        # res = self.ln2(res_)
        res = self.ffn(res_) # + res_ 
        # return res
        res = self.out_proj(res)
        
        # return torch.tanh(res)[0, :, :]
        return res[0, :, :] # best
        # return res
        
    
class Molecule_Prompts_Interatctoin_Module(nn.Module):
    def __init__(self, num_layer=1, n_cls=1):
        super().__init__()
        self.num_layer = num_layer
        
        # self.query = nn.Parameter(torch.randn([50, 1, 512]), True)
        # nn.init.normal_(self.query) # best

        self.model = nn.ModuleList()
        for i in range(self.num_layer):
            self.model.append(
                nn.TransformerEncoderLayer(d_model=768, 
                                                nhead=12, 
                                                # nhead=1, 
                                                activation=F.gelu,
                                               batch_first=False,
                                               norm_first=False, #best
                                          # norm_first=True
                                          dropout=0.0
                                           # dropout=0.1
                                            )
            )
            
#         self.prom_emb = nn.Sequential(
#                     nn.Linear(768, 768),
#                     nn.GELU(),
#                     nn.Linear(768, 768)
#                 )
        
#         self.mol_emb = nn.Sequential(
#                     nn.Linear(768, 768),
#                     nn.GELU(),
#                     nn.Linear(768, 768)
#                 )
        
        self.prom_emb = nn.Sequential(
                    nn.Linear(768, 768*4),
                    nn.GELU(),
                    nn.Linear(768*4, 768)
                )
        
        self.mol_emb = nn.Sequential(
                    nn.Linear(768, 768*4),
                    nn.GELU(),
                    nn.Linear(768*4, 768)
                )
        
        self.attn_model = nn.ModuleList()
        for i in range(self.num_layer):
            self.attn_model.append(
                Cross_Attention(768)
            )
        

        self.attr_heads = nn.ModuleList()
        self.attr_num = 6
        for i in range(self.attr_num):
            self.attr_heads.append(EnergyHead(2304, 1))

        self.positional_embedding = nn.Parameter(torch.randn(self.attr_num+n_cls+32, 768))  #known property + "unknown" + [cls] from text enc. + mol tokens
        # self.positional_embedding = nn.Parameter(torch.randn(n_cls+32, 768)) 
        # nn.init.normal_(self.positional_embedding, std=0.01)
        
        self.moltex_attn = Moltex_Attention(768, n_cls)
    
    def forward(self, mol_feats, prompts, fd, ind=0):
        prompts = prompts.unsqueeze(1).repeat(1, mol_feats.shape[1], 1)
        # prompts = self.prom_emb(prompts)
        
        # mol_feats = self.mol_emb(mol_feats)

        feats = torch.cat([prompts, mol_feats], dim=0) + self.positional_embedding.unsqueeze(1)
        for i in range(self.num_layer):
            feats = self.model[i](feats) # TransformerEncoderLayer
        
        attn = prompts
        for i in range(self.num_layer):
            attn = self.attn_model[i](attn, mol_feats) 
            
        # fp = mol_feats[self.attr_num+1, ...] 
        # md = mol_feats[self.attr_num+2, ...]
        # fd = torch.cat([fp, md], dim=-1)
        # moltex_attn = self.moltex_attn(feats)
        fd = fd # * moltex_attn
            
        cls_token = feats[0, :, :] # * attn[0, :, :]
        cls_token = torch.cat([fd, cls_token], dim=-1)
        # cls_token = feats[ind, :, :] * attn[ind, :, :]
        # cls_token = torch.cat([fd, cls_token], dim=-1)
        
        attr_pre_list = []
        attr_prompts = feats[1:self.attr_num+1, :, :] * attn[1:self.attr_num+1, :, :]
            
        for i in range(self.attr_num):
            attr_prompt = attr_prompts[i, :, :]
            attr_prompt = torch.cat([fd, attr_prompt], dim=-1)
            attr_pred = self.attr_heads[i](attr_prompt)
            attr_pre_list.append(attr_pred)
        return cls_token, attr_pre_list    



class MMPL_Finetune_Model(nn.Module):
    def __init__(self, kpgt):
        super().__init__()
        
        # sd = torch.load("/home/jovyan/prompts_learning/trained_weight/MMPL_KPGT_12_2_data_Epoch30.pth")
        sd = torch.load("/home/jovyan/prompts_learning/trained_weight/MMPL_KPGT_Pre_Epoch10.pth")
        
        self.mol_encoder = Mol_Encoder_tune(kpgt).cuda()
        # self.mol_encoder.load_state_dict(sd["mol_encoder"])
        # set_requires_grad(self.mol_encoder, False)
        
        # self.text_proc = MolPrompts_processor(model,
        self.text_proc = MolPrompts_SCIBert_processor(
            [
            "The molecule has the X X X X X",
            "The LogD of the molecule is A A A A",
            "The LogP of the molecule is B B B B",
            "The pKa of the molecule is C C C C",
            "The pKb of the molecule is D D D D",
            "The LogSol of the molecule is E E E E",
            "The wLogSol of the molecule is F F F F",
            # "The unknown property of the molecule is G G G G"
            ]).cuda()
        # set_requires_grad(self.text_proc.text_encoder, False)
        # self.text_proc.load_state_dict(sd["text_proc"], strict=False)
        self.text_proc.init_learnable_p()
        
        self.mpim = Molecule_Prompts_Interatctoin_Module()
        self.head = EnergyHead(2304, 1)
        # self.head = nn.Sequential(
        # nn.Linear(2304, 2304),
        # # nn.Tanh(),
        # nn.GELU(),
        # # nn.Dropout(0.3),
        # nn.Dropout(0.1),
        # nn.Linear(2304,1)
        # )
        # self.head = nn.Linear(2304, 1)
        print("Mol-text pre-trained weights loaded....")

    def forward(self, mol):
        mol_features, fp_vn, md_vn = self.mol_encoder(mol)
        mol_features = mol_features.permute(1, 0, 2) # LND containing fd & md
        # mol_features = torch.cat([fp_vn.unsqueeze(0), md_vn.unsqueeze(0), mol_features],dim=0)
        
        prompts = self.text_proc()
        
        if self.training and np.random.rand() < 0.15:
            scale_list = [0.1, 0.2, 0.3]
            s = np.random.choice(scale_list)
            mol_features = torch.randn_like(mol_features) * s + mol_features
            prompts = torch.randn_like(prompts) * s + prompts
            
        fd = torch.cat([fp_vn.squeeze(1), md_vn.squeeze(1)], dim=-1)   
        feat, attr_pre_list = self.mpim(mol_features, prompts, fd)
        
        # feat = F.dropout(feat, 0.2, training=self.training)
        feat = F.dropout(feat, 0.3, training=self.training)
        # feat = F.dropout(feat, 0.4, training=self.training)
        p = self.head(feat)
        
        if self.training:
            return p, attr_pre_list 
        else:
            return p

        
class KPGT_Backbone(nn.Module):
    def __init__(self, kpgt):
        super().__init__()
        self.mol_encoder = kpgt
        
    def forward(self, mol):
        batched_graph, fps, mds = mol
        mol_features = self.mol_encoder.generate_fps(batched_graph, fps, mds)
        return mol_features
        
class KPGT(nn.Module):
    def __init__(self, kpgt):
        super().__init__()
        self.mol_encoder = KPGT_Backbone(kpgt).cuda()
        self.head = EnergyHead(2304, 1)

    def forward(self, mol):
        mol_features = self.mol_encoder(mol)
        p = self.head(mol_features)
        
        return p
    
    def get_feats(self, mol):
        mol_features = self.mol_encoder(mol)
        return mol_features

    
    
class MMPL(nn.Module):
    def __init__(self, kpgt, n_cls=2):
        super().__init__()
        
        sd = torch.load("/home/jovyan/prompts_learning/trained_weight/MMPL_KPGT_Pre_Epoch10.pth")
        
        self.mol_encoder = Mol_Encoder_tune(kpgt).cuda()
        self.mol_encoder.load_state_dict(sd["mol_encoder"])
        # set_requires_grad(self.mol_encoder, False)
        
        # self.text_proc = MolPrompts_processor(model,
        self.text_proc = MolPrompts_SCIBert_processor(
            [
            "The molecule has the X X X X X",
            # "The molecule has the Y Y Y Y Y",
            # "The LogD of the molecule is A A A A",
            # "The LogP of the molecule is B B B B",
            # "The pKa of the molecule is C C C C",
            # "The pKb of the molecule is D D D D",
            # "The LogSol of the molecule is E E E E",
            # "The wLogSol of the molecule is F F F F",
            # "The unknown property of the molecule is G G G G"
            ]).cuda()
        # set_requires_grad(self.text_proc.text_encoder, False)
        self.text_proc.load_state_dict(sd["text_proc"], strict=False)
        self.text_proc.init_learnable_p()
        
        self.mpim = Molecule_Prompts_Interatctoin_Module(num_layer=1, n_cls=n_cls)
        
        self.heads = nn.ModuleList()
        for i in range(n_cls):
            head = EnergyHead(768, 1)
            self.heads.append(head)
        # self.head = nn.Linear(512, 1)
        print("Mol-text pre-trained weights loaded....")
        

    def forward(self, mol, ind=0): # ind: 0 for herg; 1 for bbb
        mol_features, fp_vn, md_vn = self.mol_encoder(mol)
        mol_features = mol_features.permute(1, 0, 2) # LND containing fd & md
        # mol_features = torch.cat([fp_vn.unsqueeze(0), md_vn.unsqueeze(0), mol_features],dim=0)
        
        prompts = self.text_proc()
        
        # if self.training and np.random.rand() < 0.2:
        #     scale_list = [0.1, 0.2, 0.3]
        #     s = np.random.choice(scale_list)
        #     mol_features = torch.randn_like(mol_features) * s + mol_features
        #     prompts = torch.randn_like(prompts) * s + prompts
            
        fd = torch.cat([fp_vn.squeeze(1), md_vn.squeeze(1)], dim=-1)
        feat, attr_pre_list = self.mpim(mol_features, prompts, fd, ind)
        
        
        p = self.heads[ind](feat)
        
        if self.training:
            return p, attr_pre_list, feat
        else:
            return p
        
    def backbone_forward(self, mol):
        mol_features, fp_vn, md_vn = self.mol_encoder(mol)
        mol_features = mol_features.permute(1, 0, 2) # LND containing fd & md
        
        mol_features = mol_features[0, ...]
        p = self.heads[0](mol_features)
        # fd = torch.cat([fp_vn.squeeze(1), md_vn.squeeze(1)], dim=-1)
        # cls_token = torch.cat([fd, mol_features], dim=-1)
        # p = self.heads[0](cls_token)
        return p

    
# if __name__ == '__main__':
#     from CLIP import clip
#     clip_model, preprocess = clip.load(name="ViT-B/16", device="cpu", download_root="/home/jovyan/clip_download_root")
#     model = CLIP_Protein(clip_model, 2, pad_len=150).cuda()
#     atoms = torch.randint(0, 1, [2, 150]).cuda()
#     coord = torch.randn([2, 150, 2*150]).cuda()
#     p = model(atoms, coord)
#     print(p)
#     print("ok")
