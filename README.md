# Multi_ACPNet
Multi_ACPNet is a dual-function predictor capable of both anticancer peptide identification (ACP/non-ACP) and anticancer peptide function prediction (multi-label classification), The corresponding research paper is ***Multi-ACPNet: A Multi-scale Sequence-Structure Feature Fusion Framework for Anticancer Peptide Identification and Functional Prediction.***
## Download 
You need to download the ESM-2 model from: https://huggingface.co/facebook/esm2_t33_650M_UR50D.
Additionally, download trRosetta from: https://github.com/gjoni/trRosetta and place the model in the "utils/trRosetta" directory.
## Requirements
The main packages and their versions used in this project are as follows:
```
python  3.7
torch  1.13.0
torch-cluster  1.6.1
torch-scatter  2.1.1
torch-sparse  0.6.16
torch-geometric  1.7.2
transformers 4.30.2
numpy 1.21.6
pandas 1.3.5
tensorboardX  2.5
tensorboard  1.14.0
```
 ## Predict
Place the prediction data in "predict_data"

To identify whether peptides are ACPs, run:
```
python predict.py --task 1
```
To predict whether ACPs show activity against the seven cancer types (Colon, Breast, Cervix, Skin, Lung, Prostate, and Blood), run:
```
python predict.py --task 2
```
