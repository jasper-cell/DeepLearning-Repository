import glob

import timm
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import torch.nn as nn
from torch import optim
import argparse
import tqdm
from sklearn import metrics
import numpy as np
import torch.nn.functional as F

parser = argparse.ArgumentParser("input information for training model")
parser.add_argument("--epoch", type=int, required=False, default=20, help="training epoch")
parser.add_argument("--lr", type=float, required=False, default=1e-3, help='learning rate')
args = parser.parse_args()


def get_names(paths):
    names = []
    for path in paths:
        image = cv2.imread(path)
        if image.shape[2] != 3:
            continue
        name = path.split("\\")[-2]
        names.append(name)
    # print("mid length: ", len(image_paths))
    return names


# 对数据集进行观察
test_image = cv2.imread("./imagenette2-160/test/cassette_player/ILSVRC2012_val_00008651.JPEG")
print(test_image.shape)

train_custom_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])

test_custom_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_image_path_lists = glob.glob("./imagenette2-160/train/*/*.JPEG")
test_image_path_lists = glob.glob("./imagenette2-160/test/*/*.JPEG")
label_encoder = LabelEncoder()
print("original length: ", len(train_image_path_lists))
# new_train_image_path_lists = []
# new_test_image_path_lists = []
#
# for path in train_image_path_lists:
#     image = cv2.imread(path)
#     # print(image.shape[2])
#     if image.shape[2] != 3:
#         # print("***************")
#         continue
#     else:
#         new_train_image_path_lists.append(path)
#
# for path in test_image_path_lists:
#     image = cv2.imread(path)
#     if image.shape[2] != 3:
#         continue
#     else:
#         new_test_image_path_lists.append(path)

train_names = get_names(train_image_path_lists)
test_names = get_names(test_image_path_lists)
# print("current length: ", len(new_train_image_path_lists))
# exit()
train_image_labels = label_encoder.fit_transform(train_names)
test_image_labels = label_encoder.fit_transform(test_names)


class CustomDataset(Dataset):
    def __init__(self, image_path_lists, label_path_lists, custom_transform=None):
        self.image_path_lists = image_path_lists
        self.label_path_lists = label_path_lists
        self.transform = custom_transform

    def __getitem__(self, index):
        image = Image.open(self.image_path_lists[index]).convert('RGB')
        label = self.label_path_lists[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_path_lists)


train_dataset = CustomDataset(train_image_path_lists, train_image_labels, train_custom_transform)
test_dataset = CustomDataset(test_image_path_lists, test_image_labels, test_custom_transform)

train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

model = timm.create_model('resnet34', pretrained=False, num_classes=10)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = model.to(device)
schedular = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
best_loss = float('inf')
for epoch in range(args.epoch):
    epoch_loss = 0
    epoch_acc = 0

    for batch, (images, labels) in enumerate(train_dataloader):
        model.train()
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        predicts = torch.max(output.data, 1)[1].cpu()

        epoch_loss += loss.cpu().item()
        epoch_acc += metrics.accuracy_score(predicts, labels.cpu())

        if batch % 10 == 0:
            print("Epoch: {}, Loss: {}, Acc: {}".format(epoch + 1, epoch_loss / (batch + 1), epoch_acc / (batch + 1)))


    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss.item()
            labels = labels.cpu().numpy()
            # print("label shape: ", labels.shape)
            predict = torch.max(outputs, 1)[1].cpu().numpy()
            # print("predict shape: ", predict.shape)
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)
    # print("labels all shape: ", labels_all.shape)
    # print("predict all shape: ", predict_all.shape)
    acc = metrics.accuracy_score(labels_all, predict_all)
    print("test acc: ", acc)
    dev_loss = loss_total / len(test_dataloader)
    if dev_loss < best_loss:
        best_loss = dev_loss
        torch.save(model.state_dict(), "./best.pth")

model.load_state_dict(torch.load("./best.pth"))
model.to(device)
model.eval()
loss_total = 0
predict_all = np.array([], dtype=int)
labels_all = np.array([], dtype=int)
with torch.no_grad():
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_total += loss.item()
        labels = labels.data.cpu().numpy()
        predict = torch.max(outputs, 1)[1].cpu().numpy()
        labels_all = np.append(labels_all, labels)
        predict_all = np.append(predict_all, predict)
print(label_encoder.classes_)
print(type(label_encoder.classes_))
report = metrics.classification_report(labels_all, predict_all, target_names=label_encoder.classes_, digits=4)
confusion = metrics.confusion_matrix(labels_all, predict_all)
print("report: \n", report)
print("confusion: \n", confusion)
