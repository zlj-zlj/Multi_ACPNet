import re
import numpy as np
import torch
from utils.feature_encoding import onehot_encoding, position_encoding, hhm_encoding,  cat, esm_encoding
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
import pandas as pd


def load_seqs(fn, label=1):
    """
    :param fn: source file name in fasta format
    :param tag: label = 1(positive, AMPs) or 0(negative, non-AMPs)
    :return:
        ids: name list
        seqs: peptide sequence list
        labels: label list
    """
    ids = []
    seqs = []


    with open(fn, 'r') as f:
        lines = f.readlines()
        max_len = 0
        file_name = 0
        for line in lines:
            line = line.strip()
            if line[0] == '>' and len(line) != 2:
                t = line.replace('|', '_')
            elif line[0] == '>' and len(line) == 2:
                t = '>' + str(file_name)
                file_name += 1
            else:
                if len(line) >max_len:
                    max_len = len(line)
                seqs.append(line)
                ids.append(t)

    if label == 1:
        labels = np.ones(len(ids))
    else:
        labels = np.zeros(len(ids))
    return ids, seqs, labels, max_len


def tensor2d_norm(tensor_2d):
    mean = tensor_2d.mean(dim=0)
    std = tensor_2d.std(dim=0)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    std_tensor = (tensor_2d - mean) / std

    return std_tensor
def get_lstm_minibatch(data, max_len=50):

    esm_list = []
    n_num_list = []
    lenn = data.batch.max().item() + 1
    for i in range(lenn):
        # 提取当前样本的节点
        mask = (data.batch == i)
        node_num = mask.sum().item()
        esm = data.s[mask]  # [node_num, esm_dim]

        t_esm = esm.permute(1, 0)
        norm_esm = tensor2d_norm(t_esm)




        pad_esm = F.pad(norm_esm, (0, max_len - t_esm.shape[1]))

        esm_list.append(pad_esm.permute(1,0))
        n_num_list.append(node_num)

    esm_batch = torch.stack(esm_list, dim=0)
    node_num = torch.tensor(n_num_list)
    return esm_batch, node_num
# def get_lstm_minibatch(data, max_len=50):
#
#     esm_list = []
#     n_num_list = []
#     lenn = data.batch.max().item() + 1
#     for i in range(lenn):
#         # 提取当前样本的节点
#         mask = (data.batch == i)
#         node_num = mask.sum().item()
#         esm = data.s[mask]  # [node_num, esm_dim]
#         c = data[i]
#         node_num1 = data[i].num_nodes
#
#         esm1 = data[i].s
#         t_esm = esm.permute(1, 0)
#
#         norm_esm = tensor2d_norm(t_esm)
#
#
#         pad_esm = F.pad(norm_esm, (0, max_len - norm_esm.shape[1]))
#
#         esm_list.append(pad_esm.permute(1,0))
#         n_num_list.append(node_num)
#
#     esm_batch = torch.stack(esm_list, dim=0)
#     node_num = torch.tensor(n_num_list)
#     return esm_batch

def aaindex_ecoding(seqs):
    aaindex_df = pd.read_csv("aaindex1.csv", index_col='Description')
    aaindex_dict = {aa: aaindex_df[aa].values for aa in aaindex_df.columns}
    features = []
    for seq in seqs:
        feature = []
        for aa in seq:
            feature.append(aaindex_dict[aa])
        features.append(np.array(feature))
    return features

def tensor2d_norm(tensor_2d):
    mean = tensor_2d.mean(dim=0)
    std = tensor_2d.std(dim=0)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    std_tensor = (tensor_2d - mean) / std

    return std_tensor

def pad(data, max_len=50):

    esm_list = []
    node_list = []
    lenn = len(data)
    for i in range(lenn):
        # 提取当前样本的节点





        esm = data[i]  # [node_num, esm_dim]
        n_node, esm_dim = esm.shape
        esm = torch.from_numpy(esm)
        t_esm = esm.permute(1, 0)

        norm_esm = tensor2d_norm(t_esm)

        pad_esm = F.pad(norm_esm, (0, max_len - norm_esm.shape[1]))

        esm = pad_esm.permute(1, 0)

        esm_list.append(esm)
        node_list.append(n_node)

    esm_batch = torch.stack(esm_list, dim=0)
    node_num = torch.tensor(node_list)


    from sklearn.decomposition import PCA
    n_components = 256
    model_pca = PCA(n_components=256)
    esm_batch = esm_batch.reshape((esm_batch.shape[0], esm_batch.shape[1], esm_batch.shape[2]))
    esm_batch = esm_batch.cpu().numpy()
    esm_batch = model_pca.fit_transform(esm_batch.reshape(-1, esm_batch.shape[-1]))
    esm_batch = esm_batch.reshape(-1, 50, n_components)
    esm_list = []

    for i, n in enumerate(node_num):
        # print(i, n)
        f = esm_batch[i][:n, :]
        esm_list.append(f)









    return esm_list

def load_data(fasta_path, npz_dir, threshold=37, label=1, add_self_loop=True):
    """
    :param fasta_path: file path of fasta
    :param npz_dir: dir that saves npz files
    :param threshold: threshold for build adjacency matrix
    :param label: labels
    :return:
        data_list: list of Data
        labels: list of labels
    """
    ids, seqs, labels, max_len = load_seqs(fasta_path, label)
    As, Es = get_cmap(npz_dir, ids, threshold, add_self_loop)

    one_hot_encodings = onehot_encoding(seqs)
    position_encodings = position_encoding(seqs)
    pssm_dir = '/'.join(fasta_path.split('/')[:-1]) + '/blos/'

    esm_dir = '/'.join(fasta_path.split('/')[:-1]) + '/esm_t33/'
    esm_encondings = esm_encoding(ids, esm_dir)
    AAindexx_econdings = aaindex_ecoding(seqs)



    # corr_matrix = np.corrcoef(pssm_encodings[0], rowvar=False)  # 计算特征间相关系数

    # 可视化

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    #
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(corr_matrix, cmap='coolwarm')
    # plt.title("ESM Feature Correlation Matrix")
    # plt.show()

    # hhm_dir = '/'.join(fasta_path.split('/')[:-1]) + '/npz_no_hhm/'
    # hhm_encodings = hhm_encoding(ids, hhm_dir)
    Xs_cnn = esm_encondings
    # Xs_cnn = pad(Xs_cnn)


    Xs_gcn = cat(one_hot_encodings, position_encodings,AAindexx_econdings)
    # Xs_gcn = cat(one_hot_encodings, AAindexx_econdings)

    n_samples = len(As)
    data_list = []
    for i in range(n_samples):

        assert As[i].shape[0] == Xs_gcn[i].shape[0] == Xs_cnn[i].shape[0], \
            f"节点数不一致：{As[i].shape[0]} (A) != {Xs_gcn[i].shape[0]} (X_gcn) != {Xs_cnn[i].shape[0]} (X_cnn)"
        data_list.append(to_parse_matrix(As[i], Xs_gcn[i], Es[i], labels[i], S=Xs_cnn[i]))
    return data_list, labels, max_len


def get_cmap(npz_folder, ids, threshold, add_self_loop=True):
    if npz_folder[-1] != '/':
        npz_folder += '/'

    list_A = []
    list_E = []

    for id in ids:
        npz = id[1:] + '.npz'
        f = np.load(npz_folder + npz)

        mat_dist = f['dist']
        mat_omega = f['omega']
        mat_theta = f['theta']
        mat_phi = f['phi']

        """ 
        The distance range (2 to 20 Å) is binned into 36 equally spaced segments, 0.5 Å each, 
        plus one bin indicating that residues are not in contact.
            - Improved protein structure prediction using predicted interresidue orientations: 
        """
        dist = np.argmax(mat_dist, axis=2)  # 37 equally spaced segments
        omega = np.argmax(mat_omega, axis=2)
        theta = np.argmax(mat_theta, axis=2)
        phi = np.argmax(mat_phi, axis=2)

        A = np.zeros(dist.shape, dtype=np.int)

        A[dist < threshold] = 1
        A[dist == 0] = 0
        # A[omega < threshold] = 1
        if add_self_loop:
            A[np.eye(A.shape[0]) == 1] = 1
        else:
            A[np.eye(A.shape[0]) == 1] = 0

        dist[A == 0] = 0
        omega[A == 0] = 0
        theta[A == 0] = 0
        phi[A == 0] = 0

        dist = np.expand_dims(dist, -1)
        omega = np.expand_dims(omega, -1)
        theta = np.expand_dims(theta, -1)
        phi = np.expand_dims(phi, -1)

        edges = dist
        edges = np.concatenate((edges, omega), axis=-1)
        edges = np.concatenate((edges, theta), axis=-1)
        edges = np.concatenate((edges, phi), axis=-1)

        list_A.append(A)
        list_E.append(edges)

    return list_A, list_E


def to_parse_matrix(A, X, E, Y, S=None, eps=1e-6):
    """
    :param A: Adjacency matrix with shape (n_nodes, n_nodes)
    :param E: Edge matrix with shape (n_nodes, n_nodes, n_edge_features)
    :param X: node embedding with shape (n_nodes, n_node_features)
    :return:
    """
    num_row, num_col = A.shape
    rows = []
    cols = []
    e_vec = []

    for i in range(num_row):
        for j in range(num_col):
            if A[i][j] >= eps:
                rows.append(i)
                cols.append(j)
                e_vec.append(E[i][j])
    edge_index = torch.tensor([rows, cols], dtype=torch.int64)
    x_cnn = torch.tensor(S, dtype=torch.float32)
    x = torch.tensor(X, dtype=torch.float32)
    edge_attr = torch.tensor(e_vec, dtype=torch.float32)
    y = torch.tensor([Y], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, s=x_cnn)



