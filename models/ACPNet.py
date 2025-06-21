import torch.nn.functional as F
from torch_geometric.nn import GATConv, LayerNorm, global_max_pool, global_mean_pool, GCNConv,GINConv,DenseGCNConv, dense_diff_pool
from torch_geometric.nn import TopKPooling
from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.nn.pool import topk_pool
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import torch_geometric as Pyg
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.utils import dense_to_sparse
import torch

# 导入库
import torch
import torch.nn as nn
# Applies weight normalization to a parameter in the given module.
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1
                                )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.LeakyReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out+res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, dilation_size=1):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            # dilation_size = 2 ** i
            dilation_size = dilation_size
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class DiffPoolLayer(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_embed):
        super().__init__()

        self.gcn = DenseGCNConv(dim_input, dim_hidden)

        self.pool = DenseGCNConv(dim_input, dim_embed)

    def forward(self, x, adj, mask=None):

        x_out = self.gcn(x, adj, mask).relu()

        s = self.pool(x, adj, mask)
        s = F.softmax(s, dim=-1)


        x_pooled, adj_pooled, link_loss, entropy_loss = dense_diff_pool(
            x_out, adj, s, mask
        )
        return x_pooled, adj_pooled, link_loss + entropy_loss


class MultiScaleGCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, scales=[1, 2, 3]):
        super().__init__()
        self.convs = nn.ModuleList([
            GCNConv(in_dim, out_dim) for _ in range(len(scales))
        ])
        self.scales = scales
        self.attention = nn.Sequential(
            nn.Linear(out_dim * len(scales), len(scales)),
            nn.Softmax(dim=-1)
        )
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None


    def forward(self, x, edge_index):
        outs = []
        for i, scale in enumerate(self.scales):
            if scale > 1:
                edge_index, _ = dense_to_sparse(
                    torch.matrix_power(
                        to_dense_adj(edge_index).squeeze(0),
                        scale))
            x_out = self.convs[i](x, edge_index)
            outs.append(F.leaky_relu(x_out))


        h = torch.cat(outs, dim=-1)
        attn_weights = self.attention(h)
        c=attn_weights[:, i].unsqueeze(-1)
        h = sum(outs[i] * attn_weights[:, i].unsqueeze(-1) for i in range(len(self.scales)))


        if self.residual:
            h += self.residual(x)
        return h



class ACPNet(nn.Module):
    def __init__(self, hidden_dim, output_dim, dropout_rate=0.5):

        super(ACPNet, self).__init__()
        layers = []
        self.temporalConvNet1 = TemporalConvNet(512*2,[512], 3,dropout_rate ,1)
        self.temporalConvNet2 = TemporalConvNet(512, [512], 3, dropout_rate, 1)
        self.downsample = nn.Conv1d(2560, 512, 1)


        self.dropout_rate = dropout_rate
        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=512,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate

        )
        self.rnn = torch.nn.GRU(input_size=2560,
                                hidden_size=30,
                                num_layers=1,
                                bidirectional=True,
                                batch_first=True,
                                dropout=dropout_rate
                                )
        self.lstm_norm = nn.LayerNorm(512 * 2)  # For bidirectional output



        self.conv1 = GCNConv(869, hidden_dim)


        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.gcn = nn.ModuleList([
            MultiScaleGCN(593 + 512, hidden_dim),
            MultiScaleGCN(hidden_dim, hidden_dim)
        ])

        self.pool = TopKPooling(in_channels=hidden_dim, ratio=0.5)






        self.fc = nn.Linear(hidden_dim, output_dim)

        fc_dim = 256
        self.head = nn.Sequential(
            nn.Linear(hidden_dim*2, fc_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        )
        self.head2 = nn.Linear(fc_dim, output_dim)
        self._initialize_weights()

    def _initialize_weights(self):


        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x, edge_index, batch, esm_batch, n_num):

        lstm_out,_ = self.lstm(esm_batch)

        lstm_out = self.lstm_norm(lstm_out)
        esm_batch = esm_batch.permute(0, 2, 1)

        lstm_out = lstm_out



        lstm_out = lstm_out.permute(0,2,1)

        TCN_out= self.temporalConvNet1(lstm_out)

        z0 = lstm_out

        f_list = []
        e_list = []

        for i, n in enumerate(n_num):
            # print(i, n)
            f = TCN_out[i][:, :n]
            f_list.append(f)

            e_list.append(TCN_out[i])
        z1 = e_list

        emb = torch.cat(f_list, dim=1).permute(1, 0)
        emb_out = global_mean_pool(emb, batch)




        cat = torch.cat((x, emb), dim=1)
        cat_out = global_mean_pool(cat, batch)
        edge_index = edge_index.clone().detach()

        x = self.gcn[0](cat, edge_index)
        x, edge_index, edge_attr, batch, perm, scor = self.pool(x, edge_index, batch=batch)
        x = self.gcn[1](x, edge_index)


        graph_emb = torch.cat([
            global_max_pool(x, batch),
            global_mean_pool(x, batch)
        ], dim=-1)



        x1 = self.head(graph_emb)
        x = self.head2(x1)


        return torch.sigmoid(x), lstm_out,emb_out,cat_out,x1





