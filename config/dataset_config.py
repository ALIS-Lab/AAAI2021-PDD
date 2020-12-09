from torchvision import transforms
import torchvision
from .TinyImageNet import TinyImageNet

def getData(dataset):

    if dataset == 'FashionMNIST':

        DATAROOT = '/data/datasets/pytorch_datasets/FashionMNIST/'
        transform_train = transforms.Compose([
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])

        train_data = torchvision.datasets.FashionMNIST(
            root=DATAROOT, train=True, download=False, transform=transform_train)
        test_data = torchvision.datasets.FashionMNIST(
            root=DATAROOT, train=False, download=False, transform=transform_test)
        num_classes = 10

        return num_classes, train_data, test_data

    if dataset == 'CIFAR100':
        DATAROOT = '/data/datasets/pytorch_datasets/CIFAR100/'
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
        ])

        train_data = torchvision.datasets.CIFAR100(
            root=DATAROOT, train=True, download=False, transform=transform_train)
        test_data = torchvision.datasets.CIFAR100(
            root=DATAROOT, train=False, download=False, transform=transform_test)
        num_classes = 100

        return num_classes, train_data, test_data

    if dataset == 'Tiny_Image':

        DATAROOT = '/data/datasets/Tiny_Imagenet/tiny-imagenet-200/'
        normalize = transforms.Normalize(
            mean=[
                0.4802, 0.4481, 0.3975], std=[
                0.2302, 0.2265, 0.2262])
        transform_train = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize
        ])
        transform_test = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.ToTensor(),
            # normalize
        ])

        in_memory = False
        train_data = TinyImageNet(
            root=DATAROOT,
            split='train',
            transform=transform_train,
            in_memory=in_memory)
        test_data = TinyImageNet(
            root=DATAROOT,
            split='val',
            transform=transform_test,
            in_memory=in_memory)

        num_classes = 200

        return num_classes, train_data, test_data

