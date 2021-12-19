import os
import cv2
import numpy as np
from torch.utils.data import Dataset


def random_brightness(img, delta = 32):
    noise = np.random.randint(-delta, delta + 1, img.shape)
    p = np.random.randint(0, 2, img.shape)
    ans = img.astype(np.int32) + noise * p
    ans[ans < 0] = 0
    ans[ans > 255] = 255
    return ans.astype(np.uint8)

def random_contrast(img):
    scale = np.random.uniform(0.5, 1.5)
    ans = img.astype(np.float32) * scale
    ans[ans > 255] = 255
    return ans.astype(np.uint8)

def random_mirror(img):
    return cv2.flip(img, 1)

def random_affine(img):
    height, width = img.shape
    center = (height / 2, width / 2)
    angle = np.random.uniform(-45, 45)
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, angle, scale)
    ans = cv2.warpAffine(img, M, img.shape)
    return ans, np.vstack([M, np.array([0, 0, 1])])

def random_tailor(img):
    height, width = img.shape
    c_h, c_w = height // 2, width // 2
    tailor = np.random.uniform(0.6, 0.9)
    M = np.identity(3)
    if np.random.rand() < 0.5:
        n_h, n_w = int(tailor * height), width
        M[1, 1], M[1, 2] = 1.0 / tailor, (1.0 - 1.0 / tailor) * c_w
    else:
        n_h, n_w = height, int(tailor * width)
        M[0, 0], M[0, 2] = 1.0 / tailor, (1.0 - 1.0 / tailor) * c_h
    ans = img[c_h - n_h // 2 : c_h + n_h // 2, c_w - n_w // 2 : c_w + n_w // 2]
    ans = cv2.resize(ans, (width, height), interpolation=cv2.INTER_AREA)
    return ans, M


class SyntheticDataset(Dataset):
    def __init__(self, folder):
        super().__init__()

        self.img = []
        self.resize = (512, 512)
        for item in os.listdir(folder):
            subpath = os.path.join(folder, item)
            for pic in os.listdir(subpath):
                path = os.path.join(subpath, pic)
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, self.resize, interpolation=cv2.INTER_AREA)
                self.img.append(image)

        self.size = len(self.img)
        self.mirror_m = np.array([[-1, 0, 512], [0, 1, 0], [0, 0, 1]])
        self.rng = np.random.default_rng()

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        img_0 = self.img[index]

        mat = np.identity(3)
        img_1 = img_0.copy()
        if self.rng.random() < 0.5:
            img_1 = random_brightness(img_1)
        if self.rng.random() < 0.5:
            img_1 = random_contrast(img_1)
        # if self.rng.random() < 0.5:
        #     img_1 = random_mirror(img_1)
        #     mat = np.matmul(self.mirror_m, mat)
        if self.rng.random() < 0.5:
            img_1, m = random_tailor(img_1)
            mat = np.matmul(m, mat)
        if self.rng.random() < 0.5:
            img_1, m = random_affine(img_1)
            mat = np.matmul(m, mat)
            
        return img_0, img_1, mat


def highlight_point(img, x, y):
    x, y = int(x), int(y)
    for i in range(-4, 5):
        for j in range(-4, 5):
            img[x + i][y + j] = 255
    return img

if __name__ == "__main__":

    dataset = SyntheticDataset("./pictures")

    print(len(dataset))
    
    im0, im1, mat = dataset[100]

    pos0 = np.array([300, 80, 1])
    pos1 = np.matmul(mat, pos0)
    print(pos0, pos1)

    im0 = highlight_point(im0, pos0[1], pos0[0])
    im1 = highlight_point(im1, pos1[1], pos1[0])

    cv2.imwrite("0.jpg", im0)
    cv2.imwrite("1.jpg", im1)