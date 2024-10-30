import os
import numpy as np
from PIL import Image
import scipy, scipy.io
from easydict import EasyDict
from collections import OrderedDict
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def get_metadata(name):
    if name == "mnist":
        metadata = EasyDict(
            {
                "image_size": 28,
                "num_classes": 10,
                "train_images": 60000,
                "val_images": 10000,
                "num_channels": 1,
            }
        )
    elif name == "mnist_m":
        metadata = EasyDict(
            {
                "image_size": 28,
                "num_classes": 10,
                "train_images": 60000,
                "val_images": 10000,
                "num_channels": 3,
            }
        )
    elif name == "cifar10":
        metadata = EasyDict(
            {
                "image_size": 32,
                "num_classes": 10,
                "train_images": 50000,
                "val_images": 10000,
                "num_channels": 3,
            }
        )
    elif name == "melanoma":
        metadata = EasyDict(
            {
                "image_size": 64,
                "num_classes": 2,
                "train_images": 33126,
                "val_images": 0,
                "num_channels": 3,
            }
        )
    elif name == "afhq":
        metadata = EasyDict(
            {
                "image_size": 64,
                "num_classes": 3,
                "train_images": 14630,
                "val_images": 1500,
                "num_channels": 3,
            }
        )
    elif name == "celeba":
        metadata = EasyDict(
            {
                "image_size": 64,
                "num_classes": 4,
                "train_images": 109036,
                "val_images": 12376,
                "num_channels": 3,
            }
        )
    elif name == "cars":
        metadata = EasyDict(
            {
                "image_size": 64,
                "num_classes": 196,
                "train_images": 8144,
                "val_images": 8041,
                "num_channels": 3,
            }
        )
    elif name == "flowers":
        metadata = EasyDict(
            {
                "image_size": 64,
                "num_classes": 102,
                "train_images": 2040,
                "val_images": 6149,
                "num_channels": 3,
            }
        )
    elif name == "gtsrb":
        metadata = EasyDict(
            {
                "image_size": 32,
                "num_classes": 43,
                "train_images": 39252,
                "val_images": 12631,
                "num_channels": 3,
            }
        )
    elif name == "shadow":
        metadata = EasyDict(
            {
                "image_size": 128,
                "num_classes": 6,
                "train_images": 12913,
                "val_images": 152,
                "num_channels": 1,
            }
        )
    elif name == "shadow_memory":
        metadata = EasyDict(
            {
                "image_size": (64, 128),
                "num_classes": 7,
                "train_images": 37132,
                "val_images": 297,
                "num_channels": 1,
            }
        )
    else:
        raise ValueError(f"{name} dataset nor supported!")
    return metadata


class oxford_flowers_dataset(Dataset):
    def __init__(self, indexes, labels, root_dir, transform=None):
        self.images = []
        self.targets = []
        self.transform = transform

        for i in indexes:
            self.images.append(
                os.path.join(
                    root_dir,
                    "jpg",
                    "image_" + "".join(["0"] * (5 - len(str(i)))) + str(i) + ".jpg",
                )
            )
            self.targets.append(labels[i - 1] - 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        target = self.targets[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, target

class shadow_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = []
        self.targets = []
        self.transform = transform
        self.targets_npy = np.load(os.path.join(root_dir, "finger_dots_w_frame_label.npy"))

        # Hard coded from dataset
        # leaving out two evaluation sequencesone, one sequence from jiaying's demo, one sequence from marion's demo
        for i in range(152, 37289):
            self.images.append(
                os.path.join(
                    root_dir, "mask_img", "".join(["0"] * (5 - len(str(i)))) + str(i) + ".png",
                )
            )
            self.targets.append(self.targets_npy[i])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("L")
        target = self.targets[idx]
        # Transformation
        if self.transform is not None:
            image = self.transform(image)
        return image, target

class shadow_memory_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = []
        self.targets = []
        self.transform = transform
        self.targets_npy = np.load(os.path.join(root_dir, "finger_dots_w_frame_label.npy"))

        # Hard coded from dataset
        # leaving out two evaluation sequencesone, one sequence from jiaying's demo, one sequence from marion's demo
        for i in range(157, 37289):
            if self.targets_npy[i][-1] == 0:
                self.images.append(
                    [os.path.join(
                        root_dir, "mask_img", "".join(["0"] * (5 - len(str(i)))) + str(i) + ".png",
                    ),
                    os.path.join(
                        root_dir, "mask_img", "".join(["0"] * (5 - len(str(i+1)))) + str(i+1) + ".png",
                    )
                    ]
                )
            else:
                self.images.append(
                    [os.path.join(
                        root_dir, "mask_img", "".join(["0"] * (5 - len(str(i)))) + str(i) + ".png",
                    ),
                    os.path.join(
                        root_dir, "mask_img", "".join(["0"] * (5 - len(str(i)))) + str(i) + ".png",
                    )
                    ]
                )
            self.targets.append(self.targets_npy[i][:7])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Concatenate images
        image0 = Image.open(self.images[idx][0]).convert("L")
        image1 = Image.open(self.images[idx][1]).convert("L")
        image = np.concatenate((np.array(image0), np.array(image1)), axis=1)
        image = Image.fromarray(image.astype('uint8'), 'L')
        target = self.targets[idx]
        # Transformation
        if self.transform is not None:
            image = self.transform(image)
        return image, target


# TODO: Add datasets imagenette/birds/svhn etc etc.
def get_dataset(name, data_dir, metadata):
    """
    Return a dataset with the current name. We only support two datasets with
    their fixed image resolutions. One can easily add additional datasets here.

    Note: To avoid learning the distribution of transformed data, don't use heavy
        data augmentation with diffusion models.
    """
    if name == "mnist":
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    metadata.image_size, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                ),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform_train,
        )
    elif name == "mnist_m":
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    metadata.image_size, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                ),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.ImageFolder(
            data_dir,
            transform=transform_train,
        )
    elif name == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform_train,
        )
    elif name in ["imagenette", "melanoma", "afhq"]:
        transform_train = transforms.Compose(
            [
                transforms.Resize(74),
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.ImageFolder(
            data_dir,
            transform=transform_train,
        )
    elif name == "celeba":
        # celebA has a large number of images, avoiding randomcropping.
        transform_train = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.ImageFolder(
            data_dir,
            transform=transform_train,
        )
    elif name == "cars":
        transform_train = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.ImageFolder(
            data_dir,
            transform=transform_train,
        )
    elif name == "flowers":
        transform_train = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        splits = scipy.io.loadmat(os.path.join(data_dir, "setid.mat"))
        labels = scipy.io.loadmat(os.path.join(data_dir, "imagelabels.mat"))
        labels = labels["labels"][0]
        train_set = oxford_flowers_dataset(
            np.concatenate((splits["trnid"][0], splits["valid"][0]), axis=0),
            labels,
            data_dir,
            transform_train,
        )
    elif name == "gtsrb":
        # celebA has a large number of images, avoiding randomcropping.
        transform_train = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.ImageFolder(
            data_dir,
            transform=transform_train,
        )
    elif name == "shadow":
        transform_train = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        )
        train_set = shadow_dataset(
            data_dir,
            transform_train,
        )
    elif name == "shadow_memory":
        transform_train = transforms.Compose(
            [
                transforms.Resize((64, 128)),
                transforms.ToTensor(),
            ]
        )
        train_set = shadow_memory_dataset(
            data_dir,
            transform_train,
        )
    else:
        raise ValueError(f"{name} dataset nor supported!")
    return train_set


def remove_module(d):
    return OrderedDict({(k[len("module.") :], v) for (k, v) in d.items()})


def fix_legacy_dict(d):
    keys = list(d.keys())
    if "model" in keys:
        d = d["model"]
    if "state_dict" in keys:
        d = d["state_dict"]
    keys = list(d.keys())
    # remove multi-gpu module.
    if "module." in keys[1]:
        d = remove_module(d)
    return d
