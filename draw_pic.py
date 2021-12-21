import argparse
import torch
import cv2
import numpy as np
import tqdm
from torch.utils.data.dataloader import DataLoader
from SyntheticDataset import *
from superpoint import SuperPoint
from superglue import SuperGlue
from utils import match_images

def get_match_pairs(matches):
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

    return {
        "point0" : np.array(point0, dtype=np.int32),
        "point1" : np.array(point1, dtype=np.int32),
        "score" : np.array(score)
    }

if __name__ == "__main__":

    parse = argparse.ArgumentParser()
    parse.add_argument("--device", type=int, default=6, help="device number")
    args = parse.parse_args()

    step = 0.1
    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

    superpoint_config = {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
    }
    superpoint_net = SuperPoint(superpoint_config).to(device)

    dataset = SyntheticDataset("./pictures")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    superglue_config_0 = {
            'weights': "synthetic",
            'sinkhorn_iterations': 100,
            'match_threshold': 0.2,
    }
    superglue_net0 = SuperGlue(superglue_config_0).to(device)
    
    superglue_config_1 = {
            'weights': "indoor",
            'sinkhorn_iterations': 100,
            'match_threshold': 0.2,
    }
    superglue_net1 = SuperGlue(superglue_config_1).to(device)

    idx = 0
    for img0, img1, mat in tqdm.tqdm(dataloader):

        if np.random.rand() > 0.02: continue

        im0 = (img0 / 255.0).float()[None].to(device)
        im1 = (img1 / 255.0).float()[None].to(device)

        data = { "image0" : im0, "image1" : im1 }
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
            pred0 = superglue_net0(data)
            pred1 = superglue_net1(data)

        data["keypoints0"] = data["keypoints0"].cpu()
        data["keypoints1"] = data["keypoints1"].cpu()
        n0, n1 = data["keypoints0"].shape[1], data["keypoints1"].shape[1]
        mat = mat.numpy()[0, :, :]

        match_pairs0 = get_match_pairs(pred0)
        match_pairs1 = get_match_pairs(pred1)

        img0 = img0[0, :, :].cpu().numpy()
        img1 = img1[0, :, :].cpu().numpy()
        out0 = match_images(img0, img1, match_pairs0)
        out1 = match_images(img0, img1, match_pairs1)

        cv2.imwrite("img/0_" + str(idx) + ".jpg", out0)
        cv2.imwrite("img/1_" + str(idx) + ".jpg", out1)
        idx += 1