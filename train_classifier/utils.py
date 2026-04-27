import random
import glob
import os
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import *
from sklearn.model_selection import train_test_split


class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None, data_cnt=-1, y=0):
        self.data_dir = data_dir
        self.filenames = glob.glob(os.path.join(data_dir, "*/*.png"))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.png")))
        random.seed(42)
        random.shuffle(self.filenames)
        if data_cnt != -1:
            self.filenames = self.filenames[:data_cnt]
        if data_dir[-1] != '/':
            data_dir += '/'
        self.img_ids = [x.replace(data_dir, '') for x in self.filenames]
        self.transform = transform
        self.y = y

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = Image.open(filename).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.y

    def __len__(self):
        return len(self.filenames)


def load_dataset(args):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            # transforms.Resize((512, 512)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset_wm = CustomImageFolder(args.wm_dir, transform=transform, y=1, data_cnt=args.data_cnt)
    dataset_org = CustomImageFolder(args.org_dir, transform=transform, y=0, data_cnt=args.data_cnt)

    print(f'number of watermarked images --> {len(dataset_wm)}')
    print(f'number of original images --> {len(dataset_org)}')

    dataset = torch.utils.data.ConcatDataset([dataset_org, dataset_wm])
    # train, test = train_test_split(np.arange(len(dataset_org) + len(dataset_wm)), test_size=0.1)

    # train_set = torch.utils.data.Subset(dataset, train)
    # test_set = torch.utils.data.Subset(dataset, test)

    # print(f'train/test split: {len(train_set), len(test_set)}')

    train_indices, test_indices = train_test_split(
        np.arange(len(dataset)), 
        test_size=0.1, 
        random_state=42,  # 固定随机种子，确保结果可复现
        stratify=[y for _, y in dataset]  # 保持类别平衡
    )
    train_set = torch.utils.data.Subset(dataset, train_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)

    # # 打印训练集标签分布
    # train_labels = [y for _, y in train_set]
    # train_positive = sum(train_labels)
    # train_negative = len(train_labels) - train_positive
    # print(f'训练集标签分布: 正样本 {train_positive}, 负样本 {train_negative}, 比例 {train_positive/train_negative:.2f}')
    
    # # 打印测试集标签分布
    # test_labels = [y for _, y in test_set]
    # test_positive = sum(test_labels)
    # test_negative = len(test_labels) - test_positive
    # print(f'测试集标签分布: 正样本 {test_positive}, 负样本 {test_negative}, 比例 {test_positive/test_negative:.2f}')
    
    # 打印前5个训练样本信息
    # print("\n前5个训练样本信息:")
    # for i in range(min(5, len(train_set))):
    #     img, label = train_set[i]
    #     print(f"  样本 {i}: 形状={img.shape}, 标签={label}, 数据类型={img.dtype}")
    
    # # 打印前5个测试样本信息
    # print("\n前5个测试样本信息:")
    # for i in range(min(5, len(test_set))):
    #     img, label = test_set[i]
    #     print(f"  样本 {i}: 形状={img.shape}, 标签={label}, 数据类型={img.dtype}")

    return train_set, test_set


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.register_buffer('means', torch.tensor(means))
        self.register_buffer('sds', torch.tensor(sds))

    def forward(self, input: torch.tensor, y=None):
        # print("norm layer input", input.max(), input.min())
        # print(self.means)
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).to(input.device)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).to(input.device)
        return (input - means)/sds


class ResNet50_BinaryClassifier(nn.Module):
    def __init__(self):
        super(ResNet50_BinaryClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 2)
        
        normalize_layer = NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
        
        self.resnet = torch.nn.Sequential(normalize_layer, self.resnet)

    def forward(self, x):
        x = self.resnet(x)
        return x

