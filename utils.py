import cv2
import numpy as np

def match_images(image0, image1, match_pairs):
    height, width = image0.shape
    
    out = np.zeros((height, width * 2, 3), np.uint8)
    for channel in range(3):
        out[:, 0 : width, channel] = image0
        out[:, width : 2 * width, channel] = image1

    mn = match_pairs["point0"].shape[0]
    for i in range(mn):
        p0, p1 = match_pairs["point0"][i, :], match_pairs["point1"][i, :]
        score = match_pairs["score"][i]
        color = (np.array([0, 255, 0]) * score + np.array([0, 0, 255]) * (1 - score)).astype(np.int32)
        cv2.line(out, p0, p1 + np.array([width, 0]), color.tolist(), thickness=1)
    
    return out