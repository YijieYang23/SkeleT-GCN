import pickle
import numpy as np
import torch
from tqdm import tqdm
import os

fuse_stream = ['j', 'b', 'jm', 'bm']
dataset = 'ntu120'
benchmark = 'xsub'

assert dataset in ['ntu120', 'ntu60']
assert benchmark in ['xset', 'xsub'] if dataset == 'ntu120' else benchmark in ['xview', 'xsub']


template = '/mnt/data/yyj/code/SkeleT-GCN/work_dirs/SkeleT-DGSTGCN/ntu120_xsub/{}/final_pred.pkl'



lookup_table = dict(
    j=dict(
        dir=template.replace('{}', 'j'),
        weight=(3, 4)
    ),
    b=dict(
        dir=template.replace('{}', 'b'),
        weight=(3, 4)
    ),
    jm=dict(
        dir=template.replace('{}', 'jm'),
        weight=(2, 2)
    ),
    bm=dict(
        dir=template.replace('{}', 'bm'),
        weight=(2, 2)
    )
)

ref = os.path.join(f'/home/yyj/dataset/ntu_rgb/2d_skeleton/{dataset}_expressive-keypoints.pkl')
if not os.path.exists(ref):
    ref = os.path.join(f'/mnt/data3/yyj/dataset/ntu_rgb/2d_skeleton/{dataset}_expressive-keypoints.pkl')
with open(ref, 'rb') as f:
    annotations = pickle.load(f)['split'][f'{benchmark}_val']

data = dict()
for modality in fuse_stream:
    with open(lookup_table[modality]['dir'], 'rb') as f:
        data[modality] = (pickle.load(f))
right_num_4s_32 = 0
right_num_4s_42 = 0
right_num_2s_11 = 0
right_num_single_modality = {
    'j': 0,
    'b': 0,
    'jm': 0,
    'bm': 0,
}
total_num = 0
for i in tqdm(range(len(annotations))):
    label = int(annotations[i][-3:]) - 1
    result32 = .0
    result42 = .0
    result11 = .0
    for modality in fuse_stream:
        weight32 = lookup_table[modality]['weight'][0]
        weight42 = lookup_table[modality]['weight'][1]
        result32 = result32 + data[modality][i] * weight32
        result42 = result42 + data[modality][i] * weight42

        right_num_single_modality[modality] += int(np.argmax(data[modality][i]) == label)

        if modality in ['j', 'b']:
            result11 = result11 + data[modality][i]

    result11 = np.argmax(result11)
    right_num_2s_11 += int(result11 == label)
    result32 = np.argmax(result32)
    right_num_4s_32 += int(result32 == label)
    result42 = np.argmax(result42)
    right_num_4s_42 += int(result42 == label)
    total_num += 1

for modality in fuse_stream:
    print(f"{modality} acc:{(right_num_single_modality[modality] / total_num)*100:.1f}")

print(f"2s acc:{(right_num_2s_11 / total_num)*100:.2f}")
print(f"4s_32 acc:{(right_num_4s_32 / total_num)*100:.2f}")
print(f"4s_42 acc:{(right_num_4s_42 / total_num)*100:.2f}")
