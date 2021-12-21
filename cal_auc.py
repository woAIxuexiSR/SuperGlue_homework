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

if __name__ == "__main__":

    parse = argparse.ArgumentParser()
    parse.add_argument("m", type=str, help="model")
    parse.add_argument("--device", type=int, default=0, help="device number")
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


    area = 0.0
    for threshold in np.arange(0.0, 1.0, step):
        
        superglue_config = {
                'weights': args.m,
                'sinkhorn_iterations': 100,
                'match_threshold': threshold,
        }
        superglue_net = SuperGlue(superglue_config).to(device)

        correct_rate_list = []
        for img0, img1, mat in tqdm.tqdm(dataloader):

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
                pred = superglue_net(data)

            data["keypoints0"] = data["keypoints0"].cpu()
            data["keypoints1"] = data["keypoints1"].cpu()
            n0, n1 = data["keypoints0"].shape[1], data["keypoints1"].shape[1]
            mat = mat.numpy()[0, :, :]

            total, correct = 0, 0
            for i in range(n0):
                if pred["matches0"][0, i] == -1:
                    continue
                a, b = i, pred["matches0"][0, i]
                p0, p1 = data["keypoints0"][0, a, :].numpy(), data["keypoints1"][0, b, :].numpy()
                project_0 = np.matmul(mat, np.hstack((p0, 1)))
                diff = project_0 - np.hstack((p1, 1))
                if np.sum(diff * diff) < 10.0:
                    correct += 1
                total += 1
            correct_rate_list.append(correct / total)
        
        print("threshold {} correct rate: {}".format(threshold, np.mean(correct_rate_list)))
        area += correct / total * step
    
    print("auc is {}".format(area))