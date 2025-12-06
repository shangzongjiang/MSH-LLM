from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize


import math


###new_added
# from .Layers import get_mask, get_subsequent_mask, refer_points, get_k_q, get_q_k
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import MessagePassing
import torch.nn.init as init
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree, softmax

from .Layers import EncoderLayer, Decoder, Predictor
from .Layers import Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct
from .embed import DataEmbedding, CustomEmbedding,DataEmbedding_new
from torch_geometric.utils import scatter

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.enc_in = configs.enc_in


        ###new_added
        self.all_size = get_mask(configs.seq_len, configs.window_size)
        self.Ms_length = sum(self.all_size)
        self.conv_layers = eval(configs.CSCM)(configs.enc_in, configs.window_size, configs.enc_in)
        self.batch = configs.batch_size
        self.dim=configs.d_model
        self.hyper_num = 50
        self.embedhy = nn.Embedding(self.hyper_num, self.dim)
        self.embednod = nn.Embedding(self.Ms_length, self.dim)
        self.Lprompt=nn.ModuleList()
        self.embedtra = nn.Linear(self.enc_in, self.dim)
        self.soft_prompt = nn.ParameterList()
        self.idx = torch.arange(self.hyper_num)
        self.nodidx=torch.arange(self.Ms_length)
        self.alpha=4
        self.k=10
        self.window_size = configs.window_size
        self.multiadphyper = multi_adaptive_hypergraoh(configs)
        self.hyper_num1 = configs.hyper_num
        self.len_prompt=configs.learn_prompt
        self.hyconv = nn.ModuleList()
        self.hyperedge_atten = SelfAttentionLayer(configs)
        self.proto_num=configs.prototypes_num

        self.weight = nn.Parameter(torch.randn(self.pred_len, 76))
        self.CMA = nn.ModuleList()
        self.prototype=nn.ModuleList()
        self.lastmap = None
        self.dimtrans=nn.Linear(self.d_llm,self.enc_in)
        self.lentrans=nn.Linear(self.hyper_num1[0],self.pred_len)



        ###LLM_load
        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            model_dir="/mnt/external/szj/szj/Llama-2-7b-hf/Llama-2-7b-hf/"
            self.llama_config = LlamaConfig.from_pretrained(model_dir)
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    # 'huggyllama/llama-7b',
                    model_dir,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    # 'huggyllama/llama-7b',
                    model_dir,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    # 'huggyllama/llama-7b',
                    model_dir,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    # 'huggyllama/llama-7b',
                    model_dir,
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            model_dir = "/mnt/external/szj/szj/GPT/GPT2_s/"
            self.gpt2_config = GPT2Config.from_pretrained(model_dir)
            # self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    model_dir,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )

                # self.llm_model = GPT2Model.from_pretrained(
                #     'openai-community/gpt2',
                #     trust_remote_code=True,
                #     local_files_only=True,
                #     config=self.gpt2_config,
                # )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    model_dir,
                    trust_remote_code=True,
                    local_files_only=True
                )

                # self.tokenizer = GPT2Tokenizer.from_pretrained(
                #     'openai-community/gpt2',
                #     trust_remote_code=True,
                #     local_files_only=True
                # )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = configs.num_token
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)


        for i in range (len(self.hyper_num1)):
            self.hyconv.append(HypergraphConv(configs.enc_in, configs.enc_in))
            param=nn.Parameter(torch.randn(self.batch, 4, self.d_llm))
            init.xavier_uniform_(param)
            self.soft_prompt.append(param)
            self.CMA.append(ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm))
            self.Lprompt.append(nn.Embedding(self.len_prompt[i],self.d_llm))
            if i==0:
                self.prototype.append(nn.Linear(self.num_tokens,self.hyper_num1[i]))
            else:
                self.prototype.append(nn.Linear(self.hyper_num1[i-1],self.hyper_num1[i]))



    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        # x_enc = self.normalize_layers(x_enc, 'norm')
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc - mean_enc
        x_enc = x_enc / std_enc

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        capability_prompt = []
        scale_num=len(self.hyper_num1)
        # prompt=prompt.to(x_enc.device)
        for b in range(B):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>,"
                "think about it step by step."
                "That's really important for me."
                "Considering ARIMA (AutoRegressive Intergrated Moving Average)."
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        data_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).input_ids
        Dprompt_embeddings = self.llm_model.get_input_embeddings()(data_prompt.to(x_enc.device))[:,:data_prompt.shape[1] // 2, :]
        # source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        adj_matrix = self.multiadphyper(x_enc)
        seq_enc = self.conv_layers(x_enc)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0))
        sum_hyper_list = []
        text_prototypes=[]
        for i in range(len(self.hyper_num1)):
            sum_hyper_list1 = []
            mask = torch.tensor(adj_matrix[i]).to(x_enc.device)
            node_value = seq_enc[i].permute(0, 2, 1)
            node_value = torch.tensor(node_value).to(x_enc.device)
            edge_sums = {}
            counts_num={}
            for edge_id, node_id in zip(mask[1], mask[0]):
                if edge_id not in edge_sums:
                    edge_id = edge_id.item()
                    node_id = node_id.item()
                    edge_sums[edge_id] = node_value[:, :, node_id]
                    counts_num[edge_id]=1
                else:
                    edge_sums[edge_id] += node_value[:, :, node_id]
                    counts_num[edge_id] += 1
            for edge_id in edge_sums:
                edge_sums[edge_id] = edge_sums[edge_id] / counts_num[edge_id]
            for edge_id, sum_value in edge_sums.items():
                sum_value = sum_value.unsqueeze(1)
                sum_hyper_list1.append(sum_value)
            sum_hyper_list1 = torch.cat(sum_hyper_list1, dim=1)
            padding = self.hyper_num1[i] - sum_hyper_list1.size(1)
            pad = torch.nn.functional.pad(sum_hyper_list1, (0, 0, 0, padding, 0, 0))
            pad = self.embedtra(pad)
            sum_hyper_list.append(pad)
            if i==0:
                scale_prototype=self.prototype[i](source_embeddings)
                text_prototypes.append(scale_prototype)
            else:
                scale_prototype = self.prototype[i](text_prototypes[i-1])
                text_prototypes.append(scale_prototype)



        result_all=[]
        for i in range(len(self.hyper_num1)):
            # lpromptidx = torch.arange(self.len_prompt[i]).to(x_enc.device)
            scale_out=self.CMA[i](sum_hyper_list[i],text_prototypes[i].permute(1, 0),text_prototypes[i].permute(1, 0))
            scale_result=torch.cat([scale_out,self.soft_prompt[i]],dim=1)
            if i==0:
                result_all=scale_result
            else:
                result_all=torch.cat([result_all,scale_result],dim=1)

        llama_enc_out = torch.cat([Dprompt_embeddings, result_all], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :self.hyper_num1[0], :]
        dec_out=self.dimtrans(dec_out).transpose(1,2)
        dec_out=self.lentrans(dec_out).transpose(1,2)

        dec_out=dec_out*std_enc + mean_enc
        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding



class multi_adaptive_hypergraoh(nn.Module):
    def __init__(self,configs):
        super(multi_adaptive_hypergraoh, self).__init__()
        self.seq_len = configs.seq_len
        self.window_size=configs.window_size
        self.inner_size=configs.inner_size
        self.dim=configs.d_model
        self.hyper_num=configs.hyper_num
        # self.alpha=3
        self.k=4
        self.embedhy=nn.ModuleList()
        self.embednod=nn.ModuleList()
        self.linhy=nn.ModuleList()
        self.linnod=nn.ModuleList()
        self.linear = nn.ModuleList()
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.phi = nn.Parameter(torch.tensor(1.0))
        for i in range(len(self.hyper_num)):
            self.embedhy.append(nn.Embedding(self.hyper_num[i],self.dim))
            self.linhy.append(nn.Linear(self.dim,self.dim))
            self.linnod.append(nn.Linear(self.dim,self.dim))
            self.linear.append(nn.Linear(self.hyper_num[i],self.hyper_num[i]))
            if i==0:
                self.embednod.append(nn.Embedding(self.seq_len,self.dim))
            else:
                product=math.prod(self.window_size[:i])
                layer_size=math.floor(self.seq_len/product)
                self.embednod.append(nn.Embedding(int(layer_size),self.dim))

        self.dropout = nn.Dropout(p=0.1)


    def forward(self,x):
        node_num = []
        node_num.append(self.seq_len)
        for i in range(len(self.window_size)):
            layer_size = math.floor(node_num[i] / self.window_size[i])
            node_num.append(layer_size)
        hyperedge_all=[]



        for i in range(len(self.hyper_num)):
            hypidxc=torch.arange(self.hyper_num[i]).to(x.device)
            nodeidx=torch.arange(node_num[i]).to(x.device)
            hyperen=self.embedhy[i](hypidxc)
            nodeec=self.embednod[i](nodeidx)
            hyperen=torch.tanh(hyperen*self.beta)
            nodeec=torch.tanh(nodeec*self.phi)
            a = torch.mm(nodeec, hyperen.transpose(1, 0))
            # a=self.linear[i](a)
            adj=self.linear[i](F.relu(a))
            mask = torch.zeros(nodeec.size(0), hyperen.size(0)).to(x.device)
            mask.fill_(float('0'))
            s1, t1 = adj.topk(min(adj.size(1),self.k), 1)
            mask.scatter_(1, t1, s1.fill_(1))
            # adj = adj * mask
            # adj = torch.where(adj > 0.5, torch.tensor(1).to(x.device), torch.tensor(0).to(x.device))
            # adj = adj[:, (adj != 0).any(dim=0)]
            # mask = mask[:, (mask != 0).any(dim=0)]
            matrix_array = torch.tensor(mask, dtype=torch.int)
            result_list = [list(torch.nonzero(matrix_array[:,col]).flatten().tolist()) for col in
                           range(matrix_array.shape[1])]

            # node_list = torch.cat([torch.tensor(sublist) for sublist in result_list if len(sublist) > 0]).tolist()
            node_list = torch.cat([torch.tensor(sublist) for sublist in result_list if len(sublist) > 0]).tolist()
            count_list = torch.sum(mask, dim=0).to(dtype=torch.int).tolist()
            hperedge_list = torch.cat([torch.full((count,), idx) for idx, count in enumerate(count_list, start=0)]).tolist()
            hypergraph=np.vstack((node_list,hperedge_list))
            hyperedge_all.append(hypergraph)





        a=hyperedge_all
        return a

class SelfAttentionLayer(nn.Module):
    def __init__(self, configs):
        super(SelfAttentionLayer, self).__init__()
        self.query_weight = nn.Linear(configs.enc_in, configs.enc_in)
        self.key_weight = nn.Linear(configs.enc_in, configs.enc_in)
        self.value_weight = nn.Linear(configs.enc_in, configs.enc_in)

    def forward(self, x):
        q = self.query_weight(x)
        k = self.key_weight(x)
        v = self.value_weight(x)

        # calculate attention scores
        attention_scores = F.softmax(torch.matmul(q, k.transpose(1, 2)) / (k.shape[-1] ** 0.5), dim=-1)

        # using attention scores for weighted averaging
        attended_values = torch.matmul(attention_scores, v)

        return attended_values


class HypergraphConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_attention=True,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0.1,
                 bias=False):
        super(HypergraphConv, self).__init__(aggr='add')
        self.soft=nn.Softmax(dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(
                torch.Tensor(in_channels, out_channels))
            self.att = Parameter(torch.Tensor(1, heads, 2 * int(out_channels / heads)))
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)
    def __forward__(self,
                    x,
                    hyperedge_index,
                    alpha=None):

        D = degree(hyperedge_index[0], x.size(0), x.dtype)
        num_edges = 2 * (hyperedge_index[1].max().item() + 1)
        B = 1.0 / degree(hyperedge_index[1], int(num_edges/2), x.dtype)
        # --------------------------------------------------------
        B[B == float("inf")] = 0


        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)
        return out



    def message(self, x_j, edge_index_i, norm, alpha):
        out = norm[edge_index_i].view(-1, 1, 1) * x_j####
        if alpha is not None:
            out=alpha.unsqueeze(-1)*out
        return out

    def forward(self, x, hyperedge_index, hyperedge_weight=None):
        x = torch.matmul(x, self.weight)
        x1=x.transpose(0,1)
        x_i = torch.index_select(x1, dim=0, index=hyperedge_index[0])
        edge_sums = {}
        for edge_id, node_id in zip(hyperedge_index[1], hyperedge_index[0]):
            if edge_id not in edge_sums:
                edge_id = edge_id.item()
                node_id = node_id.item()
                edge_sums[edge_id] = x1[node_id, :, :]
            else:
                edge_sums[edge_id] += x1[node_id, :, :]
        result_list = torch.stack([value for value in edge_sums.values()], dim=0)
        x_j = torch.index_select(result_list, dim=0, index=hyperedge_index[1])
        loss_hyper = 0


        for k in range(len(edge_sums)):
            for m in range(len(edge_sums)):
                inner_product = torch.sum(edge_sums[k] * edge_sums[m], dim=1, keepdim=True)
                norm_q_i = torch.norm(edge_sums[k], dim=1, keepdim=True)
                norm_q_j = torch.norm(edge_sums[m], dim=1, keepdim=True)
                alpha = inner_product / (norm_q_i * norm_q_j)
                distan = torch.norm(edge_sums[k] - edge_sums[m],dim=1, keepdim=True)
                loss_item = alpha * distan + (1 - alpha) * (torch.clamp(torch.tensor(4.2) - distan, min=0.0))
                loss_hyper += torch.abs(torch.mean(loss_item))


        loss_hyper = loss_hyper / ((len(edge_sums) + 1)**2)
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, hyperedge_index[0], num_nodes=x1.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        D = degree(hyperedge_index[0], x1.size(0), x.dtype)
        num_edges = 2 * (hyperedge_index[1].max().item() + 1)
        B = 1.0 / degree(hyperedge_index[1], int(num_edges/2), x.dtype)
        B[B == float("inf")] = 0

        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x1, norm=B, alpha=alpha)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)
        out=out.transpose(0, 1)
        constrain_loss = x_i - x_j
        constrain_lossfin1=torch.mean(constrain_loss)
        constrain_losstotal = abs(constrain_lossfin1) + loss_hyper



        return out, constrain_losstotal
    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


def get_mask(input_size, window_size):
    """Get the attention mask of HyperGraphConv"""
    # Get the size of all layers
    # window_size=[4,4,4]
    all_size = []
    all_size.append(input_size)
    for i in range(len(window_size)):
        layer_size = math.floor(all_size[i] / window_size[i])
        all_size.append(layer_size)
    return all_size
