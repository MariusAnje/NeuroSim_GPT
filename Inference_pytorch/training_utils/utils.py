import torch
import time
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch import nn
from training_utils.models import QCIFAR, model_from_cfg_vgg8
from torch import optim
import logging

def get_dataset(args, BS, NW):
    if args.model == "CIFAR" or args.model == "Res18" or args.model == "QCIFAR" or args.model == "QRes18" or args.model == "QDENSE" or args.model == "vgg8":
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        transform = transforms.Compose(
        [transforms.ToTensor(),
        #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            normalize])
        train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
                ])
        trainset = torchvision.datasets.CIFAR10(root='~/Private/data', train=True, download=False, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS, shuffle=True, num_workers=4)
        secondloader = torch.utils.data.DataLoader(trainset, batch_size=BS//args.div, shuffle=False, num_workers=4)
        testset = torchvision.datasets.CIFAR10(root='~/Private/data', train=False, download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BS, shuffle=False, num_workers=4)
    elif args.model == "QCIFAR100" or args.model == "QResC100":
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        transform = transforms.Compose(
        [transforms.ToTensor(),
        #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            normalize])
        train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
                ])
        trainset = torchvision.datasets.CIFAR100(root='~/Private/data', train=True, download=False, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS, shuffle=True, num_workers=4)
        secondloader = torch.utils.data.DataLoader(trainset, batch_size=BS//args.div, shuffle=False, num_workers=4)
        testset = torchvision.datasets.CIFAR100(root='~/Private/data', train=False, download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BS, shuffle=False, num_workers=4)
    elif args.model == "TIN" or args.model == "QTIN" or args.model == "QVGG":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        transform = transforms.Compose(
                [transforms.ToTensor(),
                 normalize,
                ])
        train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, 4),
                transforms.ToTensor(),
                normalize,
                ])
        trainset = torchvision.datasets.ImageFolder(root='~/Private/data/tiny-imagenet-200/train', transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS, shuffle=True, num_workers=8)
        secondloader = torch.utils.data.DataLoader(trainset, batch_size=BS//args.div, shuffle=False, num_workers=8)
        testset = torchvision.datasets.ImageFolder(root='~/Private/data/tiny-imagenet-200/val',  transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BS, shuffle=False, num_workers=8)
    elif args.model == "QVGGIN" or args.model == "QResIN":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        pre_process = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
        pre_process += [
            transforms.ToTensor(),
            normalize
        ]

        trainset = torchvision.datasets.ImageFolder('/data/data/share/imagenet/train',
                                transform=transforms.Compose(pre_process))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.ImageFolder('/data/data/share/imagenet/val',
                                transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize
                                ]))
        testloader = torch.utils.data.DataLoader(testset, batch_size=BS,
                                                    shuffle=False, num_workers=4)
    else:
        trainset = torchvision.datasets.MNIST(root='~/Private/data', train=True,
                                                download=False, transform=transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS,
                                                shuffle=True, num_workers=NW)
        secondloader = torch.utils.data.DataLoader(trainset, batch_size=BS//args.div,
                                                shuffle=False, num_workers=NW)

        testset = torchvision.datasets.MNIST(root='~/Private/data', train=False,
                                            download=False, transform=transforms.ToTensor())
        testloader = torch.utils.data.DataLoader(testset, batch_size=BS,
                                                    shuffle=False, num_workers=NW)
    return trainloader, secondloader, testloader

def get_model(args):
    if args.model == "MLP3":
        model = SMLP3()
    elif args.model == "MLP3_2":
        model = SMLP3()
    elif args.model == "MLP4":
        model = SMLP4()
    elif args.model == "LeNet":
        model = SLeNet()
    elif args.model == "CIFAR":
        model = CIFAR()
    elif args.model == "Res18":
        model = resnet.resnet18(num_classes = 10)
    elif args.model == "TIN":
        model = resnet.resnet18(num_classes = 200)
    elif args.model == "QLeNet":
        model = QSLeNet()
    elif args.model == "QCIFAR":
        model = QCIFAR()
    elif args.model == "QCIFAR100":
        model = QCIFAR100()
    elif args.model == "QRes18":
        model = qresnet.resnet18(num_classes = 10)
    elif args.model == "QResC100":
        model = qresnet.resnet18(num_classes = 100)
    elif args.model == "QDENSE":
        model = qdensnet.densenet121(num_classes = 10)
    elif args.model == "QTIN":
        model = qresnet.resnet18(num_classes = 200)
    elif args.model == "QVGG":
        model = qvgg.vgg16(num_classes = 1000)
    elif args.model == "Adv":
        model = SAdvNet()
    elif args.model == "QVGGIN":
        model = qvgg.vgg16(num_classes = 1000)
    elif args.model == "QResIN":
        model = qresnetIN.resnet18(num_classes = 1000)
    else:
        NotImplementedError
    return model

def get_model_cfg(cfg, args):
    return model_from_cfg_vgg8(cfg, args)

def prepare_model(model, device, args):
    model.to(device)
    model.push_S_device()
    model.clear_noise()
    if "TIN" in args.model or "Res" in args.model or "VGG" in args.model or "DENSE" in args.model:
    # if "TIN" in args.model or "Res" in args.model:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [1000])
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60])
    warm_optimizer = optim.SGD(model.parameters(), lr=1e-3)
    return model, optimizer, warm_optimizer, scheduler

def copy_model(old_model, args):
    new_model = get_model(args)
    state_dict = old_model.state_dict()
    for key in state_dict.keys():
        if "weight" in key:
            device = state_dict[key].device
            break
    new_model, optimizer, warm_optimizer, scheduler = prepare_model(new_model, device, args)
    new_model.load_state_dict(old_model.state_dict())
    return new_model, optimizer, warm_optimizer, scheduler

def get_logger(filepath=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)
    if filepath is not None:
        file_handler = logging.FileHandler(filepath+'.log', mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)
    return logger

def UpdateBN(model_group):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    model.train()
    total = 0
    correct = 0
    # model.clear_noise()
    with torch.no_grad():
        # for images, labels in tqdm(testloader, leave=False):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)

def TUpdateBN(t_model_group):
    t_model, criteriaF, t_optimizer, w_optimizer, t_scheduler, device, trainloader, testloader = t_model_group
    for i in range(len(t_model)):
        model = t_model[i]
        model.train()
        total = 0
        correct = 0
        model.clear_noise()
        with torch.no_grad():
            for images, labels in trainloader:
                model.clear_noise()
                # model.set_SPU(s_rate, p_rate, dev_var)
                images, labels = images.to(device), labels.to(device)
                # images = images.view(-1, 784)
                outputs = model(images)

def CEval(model_group):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    model.eval()
    total = 0
    correct = 0
    # model.clear_noise()
    with torch.no_grad():
        # for images, labels in tqdm(testloader, leave=False):
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            if len(outputs) == 2:
                outputs = outputs[0]
            predictions = outputs.argmax(dim=1)
            correction = predictions == labels
            correct += correction.sum()
            total += len(correction)
    return (correct/total).cpu().item()

def NEval(model_group, dev_var):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    model.eval()
    total = 0
    correct = 0
    model.clear_noise()
    with torch.no_grad():
        model.set_noise(dev_var)
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            if len(outputs) == 2:
                outputs = outputs[0]
            predictions = outputs.argmax(dim=1)
            correction = predictions == labels
            correct += correction.sum()
            total += len(correction)
    return (correct/total).cpu().item()

def NEachEval(model_group, dev_var):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    model.eval()
    total = 0
    correct = 0
    model.clear_noise()
    with torch.no_grad():
        for images, labels in testloader:
            model.clear_noise()
            model.set_noise(dev_var)
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            if len(outputs) == 2:
                outputs = outputs[0]
            predictions = outputs.argmax(dim=1)
            correction = predictions == labels
            correct += correction.sum()
            total += len(correction)
    return (correct/total).cpu().item()

def NTrain(model_group, epochs, header, dev_var, verbose=False):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    best_acc = 0.0
    for i in range(epochs):
        model.train()
        running_loss = 0.
        # for images, labels in tqdm(trainloader):
        for images, labels in trainloader:
            model.clear_noise()
            model.set_noise(dev_var)
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)
            loss = criteriaF(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        test_acc = NEachEval(model_group, dev_var)
        # test_acc = CEval()
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"tmp_best_{header}.pt")
        if verbose:
            print(f"epoch: {i:-3d}, test acc: {test_acc:.4f}, loss: {running_loss / len(trainloader):.4f}")
        scheduler.step()

def str2bool(a):
    if a == "True":
        return True
    elif a == "False":
        return False
    else:
        raise NotImplementedError(f"{a}")

if __name__ == "__main__":
    cfg_list = {
        'cifar10': [('C', 128, 3, 'same', 2.0),
                    ('C', 128, 3, 'same', 16.0),
                    ('M', 2, 2),
                    ('C', 256, 3, 'same', 16.0),
                    ('C', 256, 3, 'same', 16.0),
                    ('M', 2, 2),
                    ('C', 512, 3, 'same', 16.0),
                    ('C', 512, 3, 'same', 32.0),
                    ('M', 2, 2)]
    }
    class ARGS():
        def __init__(self) -> None:
            pass
    args = ARGS()
    args.model="QCIFAR"
    args.div = 1
    args.wl_weight = 4
    device = torch.device("cuda:0")
    trainloader, secondloader, testloader = get_dataset(args, 128, 4)
    # model = get_model(args)
    model = get_model_cfg(cfg_list['cifar10'], args)
    model, optimizer, warm_optimizer, scheduler = prepare_model(model, device, args)
    criteriaF = nn.CrossEntropyLoss()
    model_group = model, criteriaF, optimizer, scheduler, device, trainloader, testloader
    NTrain(model_group, 1, 1, 0.01, verbose=True)