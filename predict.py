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
    # with open(input_fasta, 'w') as f:
    #     f.write(">0|ACP|test" + '\n')
    #     f.write(args.seq)


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


def seq_encode(input_dir,encoder, tokenizer):
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
            elif line[0] == '>' and len(line) ==2:
                name = file_name
                file_name += 1
                names.append(name)
            else:
                seqs.append(line)
    for index, acp in enumerate(seqs):
        spaced_seq = " ".join(list(acp))
        inputs = tokenizer.encode_plus(
            spaced_seq,
            return_tensors=None,
            add_special_tokens=True,
            max_length=300,
            padding=True,
            truncation=True
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long).unsqueeze(0).cuda()
        with torch.no_grad():
            outputs = encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        last_hidden_states = outputs[0]
        encoded_seq = last_hidden_states[inputs['attention_mask'].bool()][1:-1]
        encod = encoded_seq.cpu().numpy()
        print(encod.shape)

        savepath = os.path.join(save_dir,f'{names[index]}.npy')
        np.save(savepath, encod)
    return names, seqs

def generate_features(args):
    rosetta_cmd = 'python ' + rosetta + ' ' + args.oa3m + ' ' + args.tr_onpz + ' -m ' + rosetta_model
    subprocess.run(rosetta_cmd)



    model = "esm2_t33_650M_UR50D"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    tokenizer = AutoTokenizer.from_pretrained(
        model)
    config = AutoConfig.from_pretrained(model,output_hidden_states=True)
    config.hidden_dropout = 0.
    config.hidden_dropout_prob = 0.
    config.attention_dropout = 0.
    config.attention_probs_dropout_prob = 0.
    encoder = AutoModel.from_pretrained(model, config=config).to(device).eval()
    print("model loaded")
    names, seqs = seq_encode(args.fasta,encoder, tokenizer)

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

    parser.add_argument('-task2_model', type=str, default='weights/multi-label-classification.pth',
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
    # edit_inputname(args.i, "data/Cancerppd/cancerppd.fasta")



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


