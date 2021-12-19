import cv2
import tqdm
import argparse
import numpy as np

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from superpoint import SuperPoint
from superglue import SuperGlue
from SyntheticDataset import *
from utils import match_images


def find_matches(kp0, kp1, transform_mat, threshold = 10.0):

    n_0, n_1 = kp0.shape[0], kp1.shape[0]
    match = []
    used0 = np.zeros(n_0, np.int32)
    used1 = np.zeros(n_1, np.int32)
    for i in range(n_0):
        for j in range(n_1):
            if used1[j] != 0: continue
            p0, p1 = kp0[i, :], kp1[j, :]
            project_0 = np.matmul(transform_mat, np.hstack((p0, 1)))
            diff = project_0 - np.hstack((p1, 1))
            if np.sum(diff * diff) < threshold:
                match.append([i, j])
                used0[i] = used1[j] = 1
                break
    unmatch0 = np.where(used0 == 0)[0]
    unmatch1 = np.where(used1 == 0)[0]
    return np.array(match), unmatch0, unmatch1



if __name__ == "__main__":

    epoches = 1000
    test_interval = 1

    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

    superpoint_config = {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
    }
    superpoint_net = SuperPoint(superpoint_config).to(device)

    superglue_config = {
            'weights': "synthetic",
            'sinkhorn_iterations': 100,
            'match_threshold': 0.2,
    }
    superglue_net = SuperGlue(superglue_config).to(device)
    # torch.save(superglue_net.state_dict(), "./weights/superglue_synthetic.pth")

    writer = SummaryWriter(os.path.join("runs", "test_0"))

    opt = optim.Adam(superglue_net.parameters(), lr=0.0001)

    dataset = SyntheticDataset("./pictures")
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    train_batches = len(train_dataloader)
    test_batches = len(test_dataloader)

    for epoch in range(epoches):

        superglue_net.train()
        train_loss = 0.0
        for img0, img1, mat in tqdm.tqdm(train_dataloader):
            # print(img0.shape, img1.shape, mat.shape)

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

            n0, n1 = data["keypoints0"].shape[1], data["keypoints1"].shape[1]
            match, unmatch0, unmatch1 = find_matches(data["keypoints0"].cpu().numpy()[0, :, :], data["keypoints1"].cpu().numpy()[0, :, :], mat.numpy()[0, :, :])
            # print(n0, n1, match.shape, unmatch0.shape, unmatch1.shape)

            pred = superglue_net(data)
            scores = pred["scores"][0, :, :]
            # print(scores.shape, matches_mat.shape)
            
            loss = 0.0
            for i in range(match.shape[0]):
                loss += -scores[ match[i][0], match[i][1] ]
            for i in range(unmatch0.shape[0]):
                loss += -scores[ unmatch0[i], -1 ]
            for i in range(unmatch1.shape[0]):
                loss += -scores[ -1, unmatch1[i] ]

            train_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()
            # print(loss)

        train_loss /= train_batches
        print("epoch {} : train loss = {}".format(epoch, train_loss))
        writer.add_scalar("train loss", train_loss, epoch)

        if epoch % test_interval != test_interval - 1:
            continue

        superglue_net.eval()
        test_loss = 0.0
        for img0, img1, mat in test_dataloader:

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

            n0, n1 = data["keypoints0"].shape[1], data["keypoints1"].shape[1]
            match, unmatch0, unmatch1 = find_matches(data["keypoints0"].cpu().numpy()[0, :, :], data["keypoints1"].cpu().numpy()[0, :, :], mat.numpy()[0, :, :])
            # print(n0, n1, match.shape, unmatch0.shape, unmatch1.shape)

            with torch.no_grad():
                pred = superglue_net(data)
            scores = pred["scores"][0, :, :]
            
            loss = 0.0
            for i in range(match.shape[0]):
                loss += -scores[ match[i][0], match[i][1] ]
            for i in range(unmatch0.shape[0]):
                loss += -scores[ unmatch0[i], -1 ]
            for i in range(unmatch1.shape[0]):
                loss += -scores[ -1, unmatch1[i] ]

            test_loss += loss.item()

        test_loss /= test_batches
        print("test loss = {}".format(test_loss))
        writer.add_scalar("test loss", test_loss, epoch)

        torch.save(superglue_net.state_dict(), "./weights/superglue_synthetic.pth")