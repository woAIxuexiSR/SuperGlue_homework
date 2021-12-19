import argparse
import torch
import cv2
import numpy as np
from SyntheticDataset import *
from superpoint import SuperPoint
from superglue import SuperGlue
from utils import match_images

def process(img_0):
    mat = np.identity(3)
    img_1 = img_0.copy()
    if np.random.rand() < 0.5:
        img_1 = random_brightness(img_1)
    if np.random.rand() < 0.5:
        img_1 = random_contrast(img_1)
    if np.random.rand() < 0.5:
        img_1, m = random_tailor(img_1)
        mat = np.matmul(m, mat)
    if np.random.rand() < 0.5:
        img_1, m = random_affine(img_1)
        mat = np.matmul(m, mat)
    return img_1, mat


if __name__ == "__main__":

    parse = argparse.ArgumentParser()
    parse.add_argument("img", type=str, help="image")
    parse.add_argument("-o", type=str, default="test.jpg", help="output image")
    parse.add_argument("-m", type=str, default="indoor", help="superglue model name")
    args = parse.parse_args()

    img0 = cv2.imread(args.img, cv2.IMREAD_GRAYSCALE)
    img0 = cv2.resize(img0, (512, 512), interpolation=cv2.INTER_AREA)
    img1, mat = process(img0)

    device = torch.device("cpu")

    superpoint_config = {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
    }
    superpoint_net = SuperPoint(superpoint_config).to(device)

    superglue_config = {
            'weights': args.m,
            'sinkhorn_iterations': 100,
            'match_threshold': 0.2,
    }
    superglue_net = SuperGlue(superglue_config).to(device)


    data = {
        "image0" : torch.from_numpy(img0/255.).float()[None, None].to(device),
        "image1" : torch.from_numpy(img1/255.).float()[None, None].to(device)
    }
    with torch.no_grad():
        pred0 = superpoint_net({"image" : data["image0"]})
        pred1 = superpoint_net({"image" : data["image1"]})

    pred = {}
    pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
    pred = {**pred, **{k+'1': v for k, v in pred1.items()}}
    data = {**data, **pred}
    for k in data:
        if isinstance(data[k], (list, tuple)):
            data[k] = torch.stack(data[k])
    
    with torch.no_grad():
        matches = superglue_net(data)
    
    m = matches["scores"].numpy()[0, :, :]

    n0, n1 = matches["matches0"].shape[1], matches["matches1"].shape[1]
    point0, point1, score = [], [], []
    threshold = 10.0
    for i in range(n0):
        if matches["matches0"][0, i] == -1:
            continue
        a, b = i, matches["matches0"][0, i]
        p0, p1 = data["keypoints0"][0, a, :].numpy(), data["keypoints1"][0, b, :].numpy()
        point0.append(p0)
        point1.append(p1)
        
        project_0 = np.matmul(mat, np.hstack((p0, 1)))
        diff = project_0 - np.hstack((p1, 1))
        if np.sum(diff * diff) < threshold:
            score.append(1)
        else: score.append(0)


    match_pairs = {
        "point0" : np.array(point0, dtype=np.int32),
        "point1" : np.array(point1, dtype=np.int32),
        "score" : np.array(score)
    }
    out = match_images(img0, img1, match_pairs)
    cv2.imwrite(args.o, out)