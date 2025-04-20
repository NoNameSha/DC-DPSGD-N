"""
download model from https://github.com/bearpaw/pytorch-classification
"""

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from transfer.resnext import resnext
import numpy as np
import os
from sklearn.linear_model import LogisticRegression

from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# trainset = datasets.CIFAR100(root='.data', train=True, download=False, transform=transform_test)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=False, num_workers=4)

# testset = datasets.CIFAR100(root='.data', train=False, download=False, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=4)

data_dir = "./data/imagenette/imagenette2"
num_workers = {"train": 2, "val": 0}
image_size = 112
image_read_func = partial(io.read_image, mode=io.image.ImageReadMode.RGB)
data_transforms = {
    "train": transforms.Compose(
        [   
            transforms.Resize((image_size, image_size)),
            #transforms.RandomRotation(20),
            #transforms.RandomHorizontalFlip(0.5),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]
    ),
    "val": transforms.Compose(
        [   
            transforms.Resize((image_size, image_size)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]
    ),
}
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x], loader=image_read_func) for x in ["train", "val"]
}
dataloaders = {
    x: data.DataLoader(image_datasets[x], batch_size=batchsize, shuffle=True, num_workers=num_workers[x])
    for x in ["train", "val"]
}

trainloader, testloader = dataloaders["train"], dataloaders["val"]

model = resnext(
    cardinality=32,
    widen_factor=4,
)

model = torch.nn.DataParallel(model).cuda()
model.eval()

checkpoint = torch.load("transfer/resnext50-32x4d/model_best.pth.tar")
model.load_state_dict(checkpoint['state_dict'])

with torch.no_grad():
    acc = 0.0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.cuda()
        outputs = torch.argmax(model(inputs), dim=-1)

        acc += torch.sum(outputs.cpu().eq(targets))

    acc /= (1.0 * len(testset))
    acc = (100 * acc).numpy()
    print(f"Test Acc on Imagenet = {acc: .2f}")

model.module.classifier = torch.nn.Identity()

features_imagenet_train = []
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        f = model(inputs.cuda()).cpu().numpy()
        features_imagenet_train.append(f)

features_imagenet_train = np.concatenate(features_imagenet_train, axis=0)
print(features_imagenet_train.shape)

mean_imagenet = np.mean(features_imagenet_train, axis=0)
var_imagenet = np.var(features_imagenet_train, axis=0)

#trainset = datasets.imagenette(root='.data', train=True, download=False, transform=transform_test)
trainset = IMBALANCECIFAR10(root='.data', imb_type="exp", 
                                    imb_factor=0.01, rand_number=0, train=True, 
                                    download=False, transform =transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=False, num_workers=4)

testset = datasets.imagenette(root='.data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=4)

ytrain = np.asarray(trainset.targets).reshape(-1)
ytest = np.asarray(testset.targets).reshape(-1)

features_train = []
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        f = model(inputs.cuda()).cpu().numpy()
        features_train.append(f)

features_train = np.concatenate(features_train, axis=0)
print(features_train.shape)

features_test = []
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        f = model(inputs.cuda()).cpu().numpy()
        features_test.append(f)

features_test = np.concatenate(features_test, axis=0)
print(features_test.shape)

os.makedirs("transfer/features/", exist_ok=True)
np.save("transfer/features/imagenet_resnext_train_lt.npy", features_train)
np.save("transfer/features/imagenet_resnext_test_lt.npy", features_test)
np.save("transfer/features/imagenet_resnext_mean_lt.npy", mean_imagenet)
np.save("transfer/features/imagenet_resnext_var_lt.npy", var_imagenet)

mean = np.mean(features_train, axis=0)
var = np.var(features_train, axis=0)

features_train_norm = (features_train - mean) / np.sqrt(var + 1e-5)
features_test_norm = (features_test - mean) / np.sqrt(var + 1e-5)

features_train_norm2 = (features_train - mean_imagenet) / np.sqrt(var_imagenet + 1e-5)
features_test_norm2 = (features_test - mean_imagenet) / np.sqrt(var_imagenet + 1e-5)

for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
    clf = LogisticRegression(random_state=0, max_iter=1000, C=C).fit(features_train, ytrain)
    print(C, clf.score(features_train, ytrain), clf.score(features_test, ytest))

    clf = LogisticRegression(random_state=0, max_iter=1000, C=C).fit(features_train_norm, ytrain)
    print(C, clf.score(features_train_norm, ytrain), clf.score(features_test_norm, ytest))

    clf = LogisticRegression(random_state=0, max_iter=1000, C=C).fit(features_train_norm2, ytrain)
    print(C, clf.score(features_train_norm2, ytrain), clf.score(features_test_norm2, ytest))

