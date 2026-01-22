# coding: utf-8
import os
import yaml
import argparse
from torch.ao.nn.quantized.functional import threshold
from torch_geometric.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import numpy as np
from models.ACPNet import ACPNet
from torch_geometric.data import Data
from utils.data import load_data,get_cmap,onehot_encoding,position_encoding,aaindex_ecoding,to_parse_matrix,get_lstm_minibatch
from utils.feature_encoding import cat
import subprocess

with open("config.yaml", 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

rosetta = cfg['rosetta']
rosetta_model = cfg['rosetta_model']


def run(args):
    input_fasta = args.fasta
    sa3m_folder = args.oa3m


    ids = []
    seqs = []
    with open(input_fasta, 'r') as f:
        lines = f.readlines()
        print(f'Num of fasta seqs is {len(lines) / 2}')
        for line in lines:
            line = line.strip()
            if line[0] == '>':
                ids.append(line)
            else:
                seqs.append(line)

        if not os.path.exists(sa3m_folder):
            os.makedirs(sa3m_folder)

        for f in os.listdir(sa3m_folder):
            os.remove(sa3m_folder + f)

        for i in range(len(ids)):
            name = ids[i]
            if len(name) == 2:
                fname = str(i)
            else:
                fname = name.replace('|', '_')[1:]
            seq = seqs[i]
            with open(sa3m_folder + fname + '.a3m', 'w') as f:
                f.write(name + '\n')
                f.write(seq)


def seq_encode(input_dir,model):
    save_dir = args.esm_opssm
    os.makedirs(save_dir, exist_ok=True)
    names = []
    seqs = []

    with open(input_dir) as f:
        line_acp = f.readlines()
        file_name = 0
        for line in line_acp:
            line = line.strip()
            if line[0] == '>' and len(line) != 2:
                name = line[1:].replace('|', '_')
                names.append(name)
            elif line[0] == '>' and len(line) == 2:
                name = file_name
                file_name += 1
                names.append(name)
            else:
                seqs.append(line)
    for index, acp in enumerate(seqs):
        protein = ESMProtein(sequence=acp)
        encoded_seqs = model.encode(protein).sequence

        embed = model(encoded_seqs.unsqueeze(0)).embeddings.to(torch.float32)
        embeddings = embed.cpu().squeeze().numpy()
        savepath = os.path.join(save_dir, f'{names[index]}.npy')
        np.save(savepath, encod)
    return names, seqs

def generate_features(args):
    rosetta_cmd = 'python ' + rosetta + ' ' + args.oa3m + ' ' + args.tr_onpz + ' -m ' + rosetta_model
    subprocess.run(rosetta_cmd)


    model = ESMC.from_pretrained("esmc_600m").to("cuda:0")
    names, seqs = seq_encode(args.fasta, model)


    return names, seqs






def test(args,pre_model, n_class):
    threshold = 0.5
    data_list, _, max_len = load_data(args.fasta, args.tr_onpz, threshold, n_class)
    # labels = np.concatenate((labels, neg_data[1]), axis=0)
    test_batch_size = max_len
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_pretrained_model(pre_model, device, output=n_class)
    test_dataloader = DataLoader(data_list, batch_size=test_batch_size, shuffle=False)
    y_pred = []
    prob = []

    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            data = data.to(device)
            true_label = data.y.float().to(device)
            esm_batch, n_num = get_lstm_minibatch(data, max_len)

            output = model(data.x, data.edge_index, data.batch, esm_batch, n_num)

            out = output[0].squeeze(1)

            y_pred.extend(out.cpu().detach().data.numpy())
        y_pred_array = np.array(y_pred)
        y_pred_binary = (y_pred_array > 0.5).astype(np.int32)
    return y_pred_binary


def load_pretrained_model(model_path, device,output):
    """加载预训练模型"""
    hd = 256
    model = ACPNet(
        hd,
        output,
        0.0
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    input_fasta = "data/predict_data/"
    parser.add_argument('-result_save', type=str,
                        default=f'{input_fasta}/result.txt',
                        help='Path to save results')
    parser.add_argument('-task1_model', type=str, default='weights/binary_classification.pth', help='Path to trained model')

    parser.add_argument('-task2_model', type=str, default='weights/multi_label_classification.pth',
                        help='Path to trained model')
    parser.add_argument('-task', type=str,
                        default='1',
                        help='1 is ACPs and non-ACPs classification, 2 is ACP Functional Activity Prediction')

    parser.add_argument('-fasta', type=str, default=f'{input_fasta}ACP.fasta',
                        help='Input files in fasta format')
    parser.add_argument('-oa3m', type=str, default=f'{input_fasta}a3m_no_hhm/',
                        help='Output folder saving o3m files')

    # trRosetta parameters

    parser.add_argument('-tr_onpz', type=str, default=f'{input_fasta}npz_no_hhm/',
                        help='Output folder saving .npz files')
    # esm
    parser.add_argument('-esm_opssm', type=str, default=f'{input_fasta}esm_t33/',
                        help='Output folder saving .pssm files')
    args = parser.parse_args()




    run(args)

    names, seqs = generate_features(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.task == '1':
        pre_model = args.task1_model
        probs = test(args, pre_model, n_class=1)
        for index, pro in enumerate(probs):
            seq = seqs[index]

            print(f"\nPrediction for sequence: {seq}")
            print(f"Predicted class: {'ACP' if pro == 1 else 'non-ACP'}")
            with open(args.result_save,"a+") as f:
                f.write(f'Sequence: {seq}\n')
                f.write(f"Prediction: {'ACP' if pro == 1 else 'non-ACP'}\n")
    elif args.task == '2':
        pre_model = args.task2_model
        probs = test(args, pre_model, n_class=7)
        label_mapping = ["Colon", "Breast", "Cervix", "Skin", "Lung", "Prostate", "Blood"]
        for i, pro in enumerate(probs):
            seq = seqs[i]
            result = ""
            index = 0
            for index, value in enumerate(pro):
                if value == 1:
                    result += label_mapping[index]
                    if index != 6:
                        result += ","
            if result == "":
                print(f"\nPrediction for sequence: {seq}")
                print(
                    "No anticancer activity detected (tested for: Colon, Breast, Cervix, Skin, Lung, Prostate, Blood)")
                with open(args.result_save, "a+") as f:
                    f.write(f'Sequence: {seq}\n')
                    f.write(f"No anticancer activity detected\n")
            else:
                print(f"\nPrediction for sequence: {seq}")
                print(
                    "anticancer activity: " + result + " (tested for: Colon, Breast, Cervix, Skin, Lung, Prostate, Blood)")
                with open(args.result_save, "a+") as f:
                    f.write(f'Sequence: {seq}\n')
                    f.write("anticancer activity: " + result+ "\n")


