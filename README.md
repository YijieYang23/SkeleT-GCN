# Expressive Keypoints for Skeleton-based Action Recognition via Skeleton Transformation
Official implementation of the paper (Expressive Keypoints for Skeleton-based Action Recognition via Skeleton Transformation).

## Abstract
In the realm of skeleton-based action recognition, the traditional methods which rely on coarse body keypoints fall short of capturing subtle human actions. In this work, we propose Expressive Keypoints that incorporates hand and foot details to form a fine-grained skeletal representation, improving the discriminative ability for existing models in discerning intricate actions. To efficiently model Expressive Keypoints, the Skeleton Transformation strategy is presented to gradually downsample the keypoints and prioritize prominent joints by allocating the importance weights. Additionally, a plug-and-play Instance Pooling module is exploited to extend our approach to multi-person scenarios without surging computation costs. Extensive experimental results over seven datasets present the superiority of our method compared to the state-of-the-art for skeleton-based human action recognition.

## Installation
Our core code is based on the PYSKL toolbox, please refer to https://github.com/kennymckormick/pyskl for installation.

Or just simply run:

```shell
conda env create -f pyskl.yaml
conda activate pyskl
pip install -e .
```

## Data Preparation
Since the license of the NTU RGB+D 60 and 120 datasets do not allow further distribution, derivation or generation, we cannot release the processed datasets publicly. You can download the official data [here](https://rose1.ntu.edu.sg/dataset/actionRecognition/). If someone is interested at the processed data, please email me (yijieyang23@gmail.com).

## Training & Testing

You can use following commands for training and testing. Basically, we support distributed training on a single server with multiple GPUs.

```shell
# Training
bash tools/dist_train.sh {config_name} {num_gpus} {other_options}
# Testing
bash tools/dist_test.sh {config_name} {checkpoint} {num_gpus} --out {output_file} --eval top_k_accuracy 
```

e.g.

```shell
# Training
bash tools/dist_train.sh configs/skelet_dgstgcn/ntu120_xsub/j.py 4 --validate --test-best --test-last
# Testing
bash tools/dist_test.sh configs/skelet_dgstgcn/ntu120_xsub/j.py ./work_dirs/skelet_dgstgcn/ntu120_xsub/j/best_top1_acc_epoch_24.pth 4 --out ./work_dirs/skelet_dgstgcn/ntu120_xsub/j/final_pred.pkl --eval top_k_accuracy 
```

