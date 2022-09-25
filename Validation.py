import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import gc
import sys
import cv2
import glob
import numpy as np
import pandas as pd
from math import ceil
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf

from src.models import MMSegModel, Segformer, SMPModel, Net, SMPUNet


def rle_decode(mask_rle, shape):
    s = np.array(mask_rle.split(), dtype=int)
    starts, lengths = s[0::2] - 1, s[1::2]
    ends = starts + lengths
    h, w = shape
    img = np.zeros((h * w,), dtype = np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = 1
    return img.reshape(shape)

def rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
    
def get_model(cfg):
    cfg = cfg.copy()
    model = eval(cfg.pop("type"))(**cfg)
    return model

def load(row):
    file_name = row.file_name
    img = cv2.imread(file_name)
    mask = rle_decode(row.rle, img.shape[:2]).T
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, mask

def resize(row, img, base_ratio, force_resize):
    resize_ratio = base_ratio / row.pixel_size
    old_size = img.shape[:2]
    new_size = (int(round(img.shape[0] / resize_ratio)), int(round(img.shape[1] / resize_ratio)))
    img = cv2.resize(img, new_size[::-1])
    if force_resize:
        img = cv2.resize(img, force_resize[::-1])
        new_size = force_resize
    return img, old_size, new_size

def cut_pad_norm(img, crop_size, stride):
    pad_size = []
    cut_point = []
    for i in range(2):
        h, H, sh = crop_size[i], img.shape[i], stride[i]
        nh = ceil((H - sh) / (h - sh))
        pH = nh * (h - sh) + sh - H
        H += pH
        cut_point.append(np.linspace(0, H - sh, nh, endpoint = False).astype(int))
        pad_size.append([pH // 2, pH - pH // 2])
    pad_img = np.pad(img, [*pad_size, [0,0]])
    pad_img = ((pad_img / 255. - mean) / std).astype(np.float32)
    return pad_img, pad_size, cut_point

def predict(row, models, cut):
    cut = torch.tensor(cut.transpose(2, 0, 1)).unsqueeze(0).cuda()
    with torch.no_grad():
        preds = []; sum_weight = 0
        for model, weight, tta, exclude_func in models:
            if exclude_func is not None and exclude_func(row): continue
            pred = model(cut).sigmoid()
            for dim in tta:
                pred += torch.flip(model(torch.flip(cut, dim)), dim).sigmoid()
            pred = pred.squeeze().detach().cpu().numpy() / (len(tta) + 1)
            preds.append(pred * weight); sum_weight += weight
        pred = sum(preds) / sum_weight
    return pred

def sliding_window_inference(row, models, pad_img, pad_size, cut_point, old_size, crop_size):
    base = np.zeros(pad_img.shape[:2], dtype = np.float32)
    sliding_cnt = np.zeros(pad_img.shape[:2], dtype = int)
    for i, x in enumerate(cut_point[0]):
        for j, y in enumerate(cut_point[1]):
            cut = pad_img[x:x + crop_size[0], y:y + crop_size[1]]
            
            pred = predict(row, models, cut)
            
            base[x:x + crop_size[0], y:y + crop_size[1]] += pred
            sliding_cnt[x:x + crop_size[0], y:y + crop_size[1]] += 1 # kernel
    base /= sliding_cnt
    base = base[pad_size[0][0]:base.shape[0]-pad_size[0][1], pad_size[1][0]:base.shape[1]-pad_size[1][1]]
    base = cv2.resize(base, old_size[::-1])
    return base



def iter_folds(names, folds = range(5)):
    D = []
    for fold in folds:
        data_dir = "./data"
        sub = pd.read_csv(os.path.join(data_dir, "train.csv"))
        test_images = glob.glob(os.path.join(data_dir, "train/images", "*.*"), recursive = True)
        sub["uid"] = [f"{o}_{i}" for o, i in zip(sub.organ, sub.id)]

        sub = sub[sub.uid.isin(pd.read_csv(os.path.join(data_dir, f"./train/splits/holdout_{fold}.txt"), header = None).iloc[:,0])].reset_index(drop = True)

            
        id2img = {int(os.path.basename(_).split(".")[0]): _ for _ in test_images}
        sub["file_name"] = sub.id.map(id2img)


        model_infos = [
            dict(
                ckpt = f"./logs/{name}/f{fold}/{last_or_best}.ckpt",
                weight = weight,
                tta = tta,
                exclude_func = exclude_func
            ) for name, weight, exclude_func in names
        ]

        models = []
        for model_info in model_infos:
            if not os.path.exists(model_info["ckpt"]):
                model_info['ckpt'] = sorted(glob.glob(model_info['ckpt']))[-1]
            stt = torch.load(model_info["ckpt"], map_location = "cpu")
            if "hyper_parameters" not in stt:
                cfg = OmegaConf.load(os.path.join(os.path.dirname(model_info["ckpt"]), "hparams.yaml"))
            else:
                cfg = OmegaConf.create(eval(str(stt["hyper_parameters"])))
            cfg = cfg.model
            if cfg.type == "Segformer":
                cfg.pretrained = True
            elif cfg.type == "MMSegModel":
                if cfg.backbone.type == "mmseg.PVT_b2":
                    cfg.backbone.model_name = "b2"
                    cfg.backbone.type = "mmseg.PyramidVisionTransformerV2"
                cfg.backbone.pretrained = None
            elif cfg.type == "SMPModel":
                cfg.pretrained_weight = None
            if all([k.startswith("model") for k in stt["state_dict"]]):
                stt = {k[6:]: v for k, v in stt["state_dict"].items()}
            elif all([k.startswith("net") for k in stt["state_dict"]]):
                stt = {k[4:]: v for k, v in stt["state_dict"].items()}
            else:
                stt = stt["state_dict"]
            
            model = get_model(cfg)
            model.load_state_dict(stt, strict = True)
            model.eval()
            model.cuda()
            models.append([model, 
                           model_info["weight"], 
                           model_info["tta"],
                           model_info["exclude_func"]])


        dices = []
        for idx in tqdm(range(len(sub))):
            row = sub.loc[idx]
            img, mask = load(row)
            pred = 0
            for force_resize in force_resizes:
                resized_img, old_size, new_size = resize(row, img, base_ratio, force_resize)
                
                pad_img, pad_size, cut_point = cut_pad_norm(resized_img, crop_size, stride)
                
                pred += sliding_window_inference(row, models, pad_img, pad_size, cut_point, old_size, crop_size)
            pred /= len(force_resizes)
            pred = (pred > thres.get(row.organ, base_thres)).astype(np.uint8)
            
            I = ((pred == 1) & (mask == 1)).sum()
            C = (pred == 1).sum() + (mask == 1).sum()
            dices.append([2 * I / C, row.organ])
        dices = pd.DataFrame(dices, columns = ["dice", "organ"])
        D.append(dices.groupby("organ").agg("mean").T)
    return D

fold_sizes = np.array([[20, 12, 9, 19, 11],
                    [20, 11, 10, 19, 10],
                    [20, 11, 10, 19, 10],
                    [19, 12, 10, 18, 11],
                    [20, 12, 9, 18, 11]])
crop_size = 1536
base_ratio = 3000 / crop_size * 0.4
stride = (crop_size // 2, crop_size // 2)
force_resizes = [(crop_size, crop_size)]
crop_size = (crop_size, crop_size)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
tta = [[2], [3], [2,3]]
thres = {"lung": 0.2, "spleen": 0.5}
base_thres = 0.5
organs = ["prostate", "spleen", "largeintestine", "kidney", "lung"]
organ_id = {o: -1 for i, o in enumerate(organs)}
last_or_best = ["epoch*", "last"][0]

names_list = [
    [
        # ("raw_coatsm", 1.0, None),
        # ("raw_mmsgf2", 1.0, None)
        ("roc_logs/sgf2_768_v1", 1.0, None),
        # ("raw_pv2d_stain", 1.0, None),
        # ("raw_b7_stain_v2", 1.0, None),
        # ("raw_pvt2daf_noLU", 1.0, lambda row: row.organ == "lung"),
        # ("fups_b4", 1.0, lambda row: row.organ == "lung"),
        # ("raw_b4", 1.0, lambda row: row.organ != "lung"),
    ],
]
for names in names_list:
    print([_[0] for _ in names])
    D = iter_folds(names)
    d = pd.concat(D).reset_index(drop = True)
    d["avg"] = (d * fold_sizes).sum(1) / fold_sizes.sum(1)
    d["avgLU"] = (d.iloc[:,:5] * fold_sizes).iloc[:,[0,1,3,4]].sum(1) / fold_sizes[:,[0,1,3,4]].sum(1)

    # print(d)
    print(",".join(d.mean(0).round(3).apply(str)))
