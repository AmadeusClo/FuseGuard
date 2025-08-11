import torch
import torch.nn as nn
from einops import rearrange
from peft import LoraConfig, TaskType, get_peft_model
# from models.GPT2_arch import AccustumGPT2Model
from models.GPT2_arch_multi import AccustumGPT2Model
from torch_geometric.nn import GCNConv
from utils.masked_attention import Mahalanobis_mask, AttentionLayer, FullAttention, EncoderLayer, Encoder
from torch_geometric.data import Data, Batch
from transformers import GPT2Tokenizer
import torch.nn.functional as F

class Encoder_metric(nn.Module):
    def __init__(self, input_dim, word_embedding, hidden_dim=768, num_heads=12, num_encoder_layers=1, mask_generator=None):
        super(Encoder_metric, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

        self.mask_generator = mask_generator

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        # Channel_transformer
        self.Channel_transformer = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag = True,
                            attention_dropout = 0.1,
                            output_attention = False,
                        ),
                        hidden_dim,
                        num_heads,
                    ),
                    d_model=hidden_dim,
                    d_ff=hidden_dim,
                    dropout=0.1,
                )
                for _ in range(num_encoder_layers)
            ],
            norm_layer=torch.nn.LayerNorm(hidden_dim)
        )

        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        
        self.word_embedding = word_embedding.T

    def forward(self, x):
        B = x.shape[0]
        F = x.shape[1]
        if self.word_embedding.ndim == 2:
            self.word_embedding = self.word_embedding.repeat(B, 1, 1)
        elif self.word_embedding.shape[0] != B:
            self.word_embedding = self.word_embedding[0].repeat(B, 1, 1)

        channel_mask = self.mask_generator(x)  # (B, F, F)

        x = self.linear(x) # (B, F, hidden_dim)

        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)

        changed_input = x  # (B,F,hidden_dim)
        x, attention = self.Channel_transformer(x=changed_input, attn_mask=channel_mask)  # (B, F, hidden_dim)

        x_time = x

        q = x.transpose(0, 1)
        k = v = self.word_embedding.transpose(0, 1)
        x, _ = self.cross_attention(q, k, v)

        x = x.transpose(0, 1)

        return x_time, x


class Encoder_trace(nn.Module):
    def __init__(self, input_dim, word_embedding, hidden_dim=768, num_heads=12, num_encoder_layers=1,
                 mask_generator=None):
        super(Encoder_trace, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

        self.mask_generator = mask_generator

        self.gcn = GCNConv(hidden_dim, hidden_dim)

        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

        self.word_embedding = word_embedding.T

    def forward(self, x):
        B = x.shape[0]
        F = x.shape[1]
        if self.word_embedding.ndim == 2:
            self.word_embedding = self.word_embedding.repeat(B, 1, 1)
        elif self.word_embedding.shape[0] != B:
            self.word_embedding = self.word_embedding[0].repeat(B, 1, 1)

        x = self.linear(x)  # (B, F, hidden_dim)

        x = x.reshape(-1, x.shape[-1])  # (B * F, hidden_dim)

        # Build a call relationship diagram based on the actual situation
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long).to(x.device)

        data_list = []
        for b in range(B):
            data = Data(x=x[b], edge_index=edge_index)
            data_list.append(data)

        batch = Batch.from_data_list(data_list)
        x = self.gcn(batch.x, batch.edge_index)  # (B * F, hidden_dim)

        x = x.reshape(B, F, -1)  # (B, F, hidden_dim)

        x_time = x

        q = x.transpose(0, 1)
        k = v = self.word_embedding.transpose(0, 1)
        x, _ = self.cross_attention(q, k, v)

        x = x.transpose(0, 1)

        return x_time, x

class Model(nn.Module):
    def __init__(self, configs, device):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=configs.r,
            lora_alpha=configs.lora_alpha,
            lora_dropout=configs.lora_dropout,
            target_modules=["c_attn"]
        )
    
        self.task_name = configs.task_name
    
        self.gpt2 = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
        self.gpt2_text = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model

        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        self.gpt2_text.h = self.gpt2_text.h[:configs.gpt_layers]
        self.gpt2 = get_peft_model(self.gpt2, peft_config)
        self.tokenizer = GPT2Tokenizer.from_pretrained('C:\\mym\\CALF-main\\CALF-main\\LLM\\gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        word_embedding = torch.tensor(torch.load(configs.word_embedding_path)).to(device=device)
        
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name or 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        for i, (name, param) in enumerate(self.gpt2_text.named_parameters()):
            if 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.time_proj = nn.ModuleList([nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(configs.gpt_layers+1)])
        
        self.text_proj = nn.ModuleList([nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(configs.gpt_layers+1)])

        self.mask_generator = Mahalanobis_mask(configs.seq_len)
        self.in_layer_trace = Encoder_trace(configs.seq_len, word_embedding, hidden_dim=configs.d_model, mask_generator=self.mask_generator)
        self.in_layer_metric = Encoder_metric(configs.seq_len, word_embedding, hidden_dim=configs.d_model,
                                       mask_generator=self.mask_generator)
        
        self.out_layer = nn.Linear(configs.d_model, configs.pred_len)

        self.gate_time = nn.Linear(configs.d_model, 1)
        self.gate_text = nn.Linear(configs.d_model, 1)
        # 文本通道注意力（可选）
        self.text_channel_attention = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model // 2),
            nn.ReLU(),
            nn.Linear(configs.d_model // 2, 1)
        )

        for layer in (self.gpt2_text, self.gpt2, self.in_layer, self.out_layer, self.time_proj, self.text_proj):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0

        self.cross_attention_modal = nn.MultiheadAttention(embed_dim=configs.d_model, num_heads=6)

    def forecast(self, x_trace, x_log, x_metric):
        B, L, M = x_trace.shape

        ###### Data preprocessing for each modality #####
        # Trace modality
        x_trace = rearrange(x_trace, 'b l m -> b m l')
        outputs_time1, outputs_text1 = self.in_layer_trace(x_trace)

        # Metric modality
        x_metric = rearrange(x_metric, 'b l m -> b m l')
        outputs_time2, outputs_text2 = self.in_layer_metric(x_metric)

        # Log modality processing
        B, F_dim, Hidden_dim = outputs_time1.shape
        service_lists = len(x_log)
        device = next(self.gpt2_text.parameters()).device
        Log_embeddings = torch.zeros((B, service_lists, Hidden_dim)).to(device)
        for i in range(service_lists):
            inputs = self.tokenizer(x_log[i], padding=True, truncation=True, return_tensors="pt")
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            outputs = self.gpt2_text(input_ids, attention_mask=attention_mask)
            hidden_states = outputs[0]
            Log_embeddings[:, i, :] = hidden_states.mean(dim=1)

        ###### Multimodal alignment #####
        # Trace modality processing
        outputs_time_trace, intermidiate_feat_time_trace = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text_trace, intermidiate_feat_text_trace = self.gpt2_text(inputs_embeds=outputs_text1)

        # Metric modality processing
        outputs_time_metric, intermidiate_feat_time_metric = self.gpt2(inputs_embeds=outputs_time2)
        outputs_text_metric, intermidiate_feat_text_metric = self.gpt2_text(inputs_embeds=outputs_text2)

        # Residual connection
        outputs_time_trace += outputs_time1
        outputs_text_trace += outputs_text1
        outputs_time_metric += outputs_time2
        outputs_text_metric += outputs_text2

        # Feature projection
        intermidiate_feat_time_trace = tuple(
            [self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time_trace))])
        intermidiate_feat_text_trace = tuple(
            [self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text_trace))])
        intermidiate_feat_time_metric = tuple(
            [self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time_metric))])
        intermidiate_feat_text_metric = tuple(
            [self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text_metric))])

        ###### Trimodal dynamic fusion #####
        # 1. Calculate energy scores for each modality
        # Trace time modality
        time_energy_trace = -torch.logsumexp(self.gate_time(outputs_time_trace), dim=1)  # (B, 1)
        # Metric time modality
        time_energy_metric = -torch.logsumexp(self.gate_time(outputs_time_metric), dim=1)  # (B, 1)
        # Text modality
        text_energy = -torch.logsumexp(self.gate_text(Log_embeddings), dim=1)  # (B, 1)

        # Convert energy to confidence weights
        time_conf_trace = -0.1 * time_energy_trace
        time_conf_metric = -0.1 * time_energy_metric
        text_conf = -0.1 * text_energy

        # Weight normalization
        conf_total = torch.cat([time_conf_trace, time_conf_metric, text_conf], dim=1)  # (B, 3)
        weights = F.softmax(conf_total, dim=1)  # (B, 3)
        gate_time_trace, gate_time_metric, gate_text = weights[:, 0], weights[:, 1], weights[:, 2]

        # 2. Weight each modality
        # Trace time modality
        weighted_time_trace = outputs_time_trace * gate_time_trace.view(B, 1, 1)
        # Metric time modality
        weighted_time_metric = outputs_time_metric * gate_time_metric.view(B, 1, 1)

        # 3. Weighted pooling for text modality
        text_weights = F.softmax(self.text_channel_attention(Log_embeddings), dim=1)  # (B, C, 1)
        weighted_text = (Log_embeddings * text_weights).sum(dim=1, keepdim=True)  # (B, 1, H)
        weighted_text = weighted_text * gate_text.view(B, 1, 1)

        # 4. Trimodal fusion
        # Time modality fusion (trace + metric)
        fused_time = weighted_time_trace + weighted_time_metric
        # Final fusion (time + text)
        fused_output = fused_time + weighted_text.expand(-1, F_dim, -1)
        outputs_time_fused = self.out_layer(fused_output[:, -M:, :])

        # 5. Auxiliary outputs for each modality (containing both modality-specific and fused features)
        outputs_time_trace = self.out_layer(
            torch.cat([weighted_time_trace[:, -M:, :], fused_output[:, -M:, :]], dim=-1)
        )
        outputs_time_metric = self.out_layer(
            torch.cat([weighted_time_metric[:, -M:, :], fused_output[:, -M:, :]], dim=-1)
        )

        # Output transformation
        outputs_text_trace = self.out_layer(outputs_text_trace[:, -M:, :])
        outputs_text_metric = self.out_layer(outputs_text_metric[:, -M:, :])

        return {
            'outputs_time_trace': rearrange(outputs_time_trace, 'b m l -> b l m'),
            'outputs_text_trace': rearrange(outputs_text_trace, 'b m l -> b l m'),
            'outputs_time_metric': rearrange(outputs_time_metric, 'b m l -> b l m'),
            'outputs_text_metric': rearrange(outputs_text_metric, 'b m l -> b l m'),
            'intermidiate_time_trace': intermidiate_feat_time_trace,
            'intermidiate_text_trace': intermidiate_feat_text_trace,
            'intermidiate_time_metric': intermidiate_feat_time_metric,
            'intermidiate_text_metric': intermidiate_feat_text_metric,
        }


    def forward(self, x_trace, x_log, x_metric, mask=None):
        output = self.forecast(x_trace, x_log, x_metric)
        return output
