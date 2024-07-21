import os
import tqdm
import json
from visual_nuscenes import NuScenes
use_gt = False
out_dir = './result_vis/'
result_json = "/data/workspace/lhjj/PETR-main/test/petr_r50dcn_gridmask_c5/Wed_Jan__3_15_24_23_2024/pts_bbox/results_nusc"
dataroot='./data/nuscenes'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if use_gt:
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True, pred = False, annotations = "sample_annotation")
else:
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True, pred = True, annotations = result_json, score_thr=0.25)

with open('{}.json'.format(result_json)) as f:
    table = json.load(f)
tokens = list(table['results'].keys())
token = '4fbd8ca56cc942cfb092846192fb3bd3'
nusc.render_sample(token, out_path = "./result_vis/"+token+"_pred.png", verbose=False)
# for token in tqdm.tqdm(tokens[:100]):
#     if use_gt:
#         nusc.render_sample(token, out_path = "./result_vis/"+token+"_gt.png", verbose=False)
#     else:
#         nusc.render_sample(token, out_path = "./result_vis/"+token+"_pred.png", verbose=False)

