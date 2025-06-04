import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import MessagePassing
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.data import data as D
from torch.nn import Linear
import torch_scatter
from math import sqrt
import torch.nn.init as init
from .Layers import EncoderLayer, Decoder, Predictor
from .Layers import Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct
# from .Layers import get_mask, get_subsequent_mask, refer_points, get_k_q, get_q_k
from .embed import DataEmbedding, CustomEmbedding,DataEmbedding_new
from torch_geometric.utils import scatter

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()

import math




class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        configs.device = torch.device("cuda")
        self.channels = configs.enc_in
        self.enc_in=configs.enc_in
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

            # self.Linear_Tran = nn.Linear(self.pred_len, self.pred_len)


        self.all_size=get_mask(configs.seq_len, configs.window_size)
        self.Ms_length = sum(self.all_size)
        self.conv_layers = eval(configs.CSCM)(configs.enc_in, configs.window_size, configs.enc_in)
        # self.out_tran = nn.Linear(self.Ms_length, self.pred_len)
        # self.inter_tran = nn.Linear(80, self.pred_len)
        # self.out_tran.weight=nn.Parameter((1/self.Ms_length)*torch.ones([self.pred_len,self.Ms_length]))
        # self.chan_tran=nn.Linear(configs.d_model,configs.enc_in)
        # self.concat_tra=nn.Linear(248,self.pred_len)

        self.d_ff = configs.d_ff
        self.d_llm = configs.llm_dim
        self.batch=configs.batch_size

        self.dim=configs.d_model
        self.hyper_num=50

        self.embedhy=nn.Embedding(self.hyper_num,self.dim)
        self.embednod=nn.Embedding(self.Ms_length,self.dim)
        self.embedtra = nn.Linear(self.enc_in, self.dim)
        # self.soft_prompt=nn.ModuleList()
        self.soft_prompt = nn.ParameterList()
        # self.soft_prompt=nn.Parameter(torch.randn(self.batch,4,self.d_llm))
        # init.xavier_uniform_(self.soft_prompt)


        self.idx = torch.arange(self.hyper_num)
        self.nodidx=torch.arange(self.Ms_length)

        self.alpha=4
        self.k=10

        self.window_size=configs.window_size
        self.multiadphyper=multi_adaptive_hypergraoh(configs)
        self.hyper_num1 = configs.hyper_num
        self.hyconv=nn.ModuleList()
        self.hyperedge_atten=SelfAttentionLayer(configs)
        for i in range (len(self.hyper_num1)):
            self.hyconv.append(HypergraphConv(configs.enc_in, configs.enc_in))
            param=nn.Parameter(torch.randn(self.batch, 4, self.d_llm))
            init.xavier_uniform_(param)
            self.soft_prompt.append(param)

        # self.slicetran=nn.Linear(100,configs.pred_len)
        self.weight = nn.Parameter(torch.randn(self.pred_len, 76))

        # self.argg = nn.ModuleList()
        # for i in range(len(self.hyper_num1)):
        #     self.argg.append(nn.Linear(self.all_size[i],self.pred_len))
        # self.chan_tran=nn.Linear(configs.enc_in,configs.enc_in)

        # self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
        self.CMA=nn.ModuleList()
        for i in range(len(self.hyper_num1)):
            self.CMA.append(ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm))

        # self.dimenmap=nn.Linear(self.d_llm,self.enc_in)
        self.lastmap=None

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            model_dir="/mnt/szj/Llama-2-7b-hf/Llama-2-7b-hf/"
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
            model_dir = "/mnt/szj/GPT/GPT2_s/"
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
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)


    def forward(self, x,x_mark_enc):
        # normalization
        mean_enc=x.mean(1,keepdim=True).detach()
        x=x - mean_enc
        std_enc=torch.sqrt(torch.var(x,dim=1,keepdim=True,unbiased=False)+1e-5).detach()
        x=x / std_enc

        # adj_matrix = self.multiadphyper(x)

        # B, T, N = x.size()
        # x=x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x, dim=1)[0]
        max_values = torch.max(x, dim=1)[0]
        medians = torch.median(x, dim=1).values
        prompt = []
        # prompt = []
        capability_prompt=[]
        # prompt=prompt.to(x_enc.device)
        scale_num=len(self.hyper_num1)
        for b in range(x.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task Instruction: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Data statistics: "
                f"The number of scale is  {scale_num}, "
                f"the max value is {max_values_str}, "
                f"the min value is {min_values_str}, "
                f"median value is {median_values_str}. "
            )

            prompt.append(prompt_)
        Cprompt_=("Think about it step by step."
                "That's really important for me."
                "Considering ARIMA (AutoRegressive Intergrated Moving Average).")
        capability_prompt.append(Cprompt_)
        # x_enc = x.reshape(B, N, T).permute(0, 2, 1).contiguous()
        x_enc=x
        data_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).input_ids
        capability_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).input_ids
        Dprompt_embeddings = self.llm_model.get_input_embeddings()(data_prompt.to(x_enc.device))[:, :data_prompt.shape[1] // 2, :]
        Cprompt_embeddings = self.llm_model.get_input_embeddings()(capability_prompt.to(x_enc.device))[:, :capability_prompt.shape[1] // 2, :]
        prompt_embeddings=torch.cat([Dprompt_embeddings, Cprompt_embeddings], dim=1)
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        adj_matrix = self.multiadphyper(x_enc)



        seq_enc = self.conv_layers(x_enc)

        sum_hyper_list = []
        multi_all=[]
        for i in range(len(self.hyper_num1)):
            sum_hyper_list1 = []
            mask = torch.tensor(adj_matrix[i]).to(x.device)
            ###inter-scale
            node_value = seq_enc[i].permute(0,2,1)
            node_value = torch.tensor(node_value).to(x.device)
            edge_sums={}
            for edge_id, node_id in zip(mask[1], mask[0]):
                if edge_id not in edge_sums:
                    edge_id=edge_id.item()
                    node_id=node_id.item()
                    edge_sums[edge_id] = node_value[:, :, node_id]
                else:
                    edge_sums[edge_id] += node_value[:, :, node_id]


            for edge_id, sum_value in edge_sums.items():
                sum_value = sum_value.unsqueeze(1)
                sum_hyper_list.append(sum_value)
                sum_hyper_list1.append(sum_value)
            sum_hyper_list1 = torch.cat(sum_hyper_list1, dim=1)
            padding = self.hyper_num1[i] - sum_hyper_list1.size(1)
            pad = torch.nn.functional.pad(sum_hyper_list1, (0, 0, 0, padding, 0, 0))##pad[16,50,7]
            pad=self.embedtra(pad)
            enc_out = self.CMA[i](pad, source_embeddings, source_embeddings)
            assert self.soft_prompt[i].size(2) == enc_out.size(2), "The last dimension must match"
            if self.soft_prompt[i].size(0) != self.batch:
                self.soft_prompt[i] = self.soft_prompt[i].unsqueeze(0).expand(self.batch, -1, -1)
            multi_concat=torch.cat([self.soft_prompt[i], enc_out], dim=1)
            multi_all.append(multi_concat)

                                                               ###intra-scale
            # output,constrainloss = self.hyconv[i](seq_enc[i], mask)

            # if i==0:
            #     result_tensor=output
            #     result_conloss=constrainloss
            # else:
            #     result_tensor = torch.cat((result_tensor, output), dim=1)
            #     result_conloss+=constrainloss

        multi_all=torch.cat(multi_all,dim=1)
        enc_out=torch.cat([multi_all],dim=1)
        all_out=torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=all_out).last_hidden_state
        dec_out=dec_out.permute(0,2,1)
        self.linearmap=nn.Linear(dec_out.size(2),self.pred_len).to(x.device)

        dec_out=self.linearmap(dec_out).permute(0,2,1)
        # dec_out=self.dimenmap(dec_out)
        dec_out = dec_out[:, :, :self.enc_in]

        # sum_hyper_list=torch.cat(sum_hyper_list,dim=1)
        # sum_hyper_list=sum_hyper_list.to(x.device)
        # padding_need=80-sum_hyper_list.size(1)
        # hyperedge_attention=self.hyperedge_atten(sum_hyper_list)
        # pad = torch.nn.functional.pad(hyperedge_attention, (0, 0, 0, padding_need, 0, 0))
        # pad = torch.nn.functional.pad(sum_hyper_list, (0, 0, 0, padding_need, 0, 0))



        # concat_result=torch.cat((result_tensor,pad),dim=1)###ori
        """
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x_enc.permute(0,2,1))
            # x_out1=self.concat_tra(concat_result.permute(0,2,1))
        # x=x_out1+x
        x=self.Linear_Tran(x).permute(0,2,1)
        x=self.chan_tran(x)
        """
        x = self.Linear(x_enc.permute(0, 2, 1)).permute(0,2,1)
        x = dec_out + x
        x = x*std_enc + mean_enc
        return x


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
