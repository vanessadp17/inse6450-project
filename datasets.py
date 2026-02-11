
import torchvision
from torch.utils.data import DataLoader
import transforms
from tinyimagenet import TinyImageNet


class CIFAR(object):
    """
    Base class for CIFAR-10/100 datasets.
    
    Args:
        dataset_cls: PyTorch dataset class
        data_path: Path to store/load the dataset
        norm_mean: Tuple of mean values for normalization (per channel)
        norm_std: Tuple of standard deviation values for normalization (per channel)
        batch_size: Number of samples per batch
        use_gpu: Boolean to use GPU
        num_workers: Number of subprocesses for data loading
    """
    def __init__(self, dataset_cls, data_path, norm_mean, norm_std, batch_size, use_gpu, num_workers):
        # Define training transformations with data augmentation
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])
        # Define test transformations
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])

        pin_memory = use_gpu

        # Load training dataset
        trainset = dataset_cls(data_path, train=True, download=True, transform=transform_train)
        self.trainset = trainset
        # Create mappings between class indices and class names
        self.idx_to_class = {v: k for k, v in trainset.class_to_idx.items()}
        self.class_to_idx = trainset.class_to_idx

        # Create training data loader
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.trainloader = trainloader

        # Load test dataset
        testset = dataset_cls(data_path, train=False, download=True, transform=transform_test)
        self.testset = testset
        # Create test data loader
        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.testloader = testloader

        self.num_classes = known_class  # known classes from mismatch

class CIFAR100(CIFAR):
    """
    CIFAR-100 dataset
    """
    def __init__(self, *args, **kwargs):
        super().__init__(
            torchvision.datasets.CIFAR100,
            "./data/cifar100",
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761),
            *args,
            **kwargs
        )


class CIFAR10(CIFAR):
    """
    CIFAR-10 dataset
    """
    def __init__(self, *args, **kwargs):
        super().__init__(
            torchvision.datasets.CIFAR10,
            "./data/cifar10",
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
            *args,
            **kwargs
        )

class MyTinyImageNet(object):
    """
    Class for TinyImageNet dataset.
    
    Args:
        batch_size: Number of samples per batch
        use_gpu: Boolean whether to use GPU
        num_workers: Number of subprocesses for data loading
    """
    def __init__(self, batch_size, use_gpu, num_workers):
        # Define training transformations with data augmentation
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(TinyImageNet.mean, TinyImageNet.std)
        ])
        # Define test transformations 
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(TinyImageNet.mean, TinyImageNet.std)
        ])

        pin_memory = use_gpu

        # Load training dataset
        trainset = TinyImageNet("./data/tinyimagenet", split="train", transform=transform_train, imagenet_idx=True)
        self.trainset = trainset
        # Create mappings between class indices and class names
        self.idx_to_class = {v: k for k, v in trainset.class_to_idx.items()}
        self.class_to_idx = trainset.class_to_idx
        # Create training data loader
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.trainloader = trainloader

        # Load validation dataset
        testset = TinyImageNet("./data/tinyimagenet", split="val", transform=transform_test, imagenet_idx=True)
        self.testset = testset
        # Create validation data loader
        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.testloader = testloader

        self.num_classes = known_class  # known classes from mismatch

__factory = {
    'cifar100': CIFAR100,
    'cifar10': CIFAR10,
    'tinyimagenet': MyTinyImageNet}

def create(name, known_class_, knownclass, batch_size, use_gpu, num_workers):
    """
    Function to create dataset loaders.
    
    Args:
        name: Name of the dataset ('cifar10', 'cifar100', or 'tinyimagenet')
        known_class_: Number of known classes
        knownclass: List of known class indices
        batch_size: Number of samples per batch
        use_gpu: Boolean whether to use GPU 
        num_workers: Number of subprocesses for data loading
    
    Returns:
        Dataset loader instance (CIFAR10, CIFAR100, or MyTinyImageNet)
    """
    global known_class, knownclass_list
    known_class = known_class_  # mismatch ratio
    knownclass_list = knownclass
    if name not in __factory.keys():
        raise KeyError(f"Unknown dataset: {name}")
    return __factory[name](batch_size, use_gpu, num_workers)