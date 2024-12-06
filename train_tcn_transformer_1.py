import copy
from dataset import PoseDataset
from network import TCNTransformer
import numpy as np
import torch
from torch.utils.data import DataLoader
from loss import FocalLoss
from common import AverageMeter, accuracy, balanced_acc
import time
import csv
import torch.nn as nn



def train(train_loader, model, criterion, optimizer, epoch):
    print("training... ", epoch + 1)
    batch_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    model.train()
    end = time.time()
    for i, (pose, target) in enumerate(train_loader):
        pose = pose.float()
        if torch.cuda.is_available():
            pose = pose.cuda()
            target = target.cuda()

        output, _, = model(pose)

        acc = accuracy(output, target)
        accs.update(acc, output.size(0))

        loss = criterion(output, target)
        losses.update(loss.item(), pose.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if (epoch+1) % 10 == 0:
        print('Epoch: [{0}][{1}/{2}]   '
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
              'Loss {loss.val:.6f} ({loss.avg:.6f})   '
              'Acc {accs.val:.3f} ({accs.avg:.3f})   '.format(
            epoch, i, len(train_loader), batch_time=batch_time,
            loss=losses, accs=accs))

    return accs.avg, losses.avg


def validate(test_loader, model, criterion, epoch):
    print("validate... ", epoch + 1)
    confusion_matrix = np.zeros([2, 2], int)
    accs = AverageMeter()
    losses = AverageMeter()
    model.eval()
    for i, (pose, target) in enumerate(test_loader):
        pose = pose.float()
        if torch.cuda.is_available():
            pose = pose.cuda()
            target = target.cuda()
        output, _,  = model(pose)

        acc = accuracy(output, target)
        accs.update(acc, output.size(0))

        loss = criterion(output, target)
        losses.update(loss.item(), pose.size(0))

        # calculate confusion matrix
        _, predicted = torch.max(output.data, 1)  # return 1 dimensional array: tensor([index1, index2])

        for it in range(len(predicted)):
            confusion_matrix[target[it].item(), predicted[it].item()] += 1
    if (epoch+1) % 10 == 0:
        print('Validation Accuracy: {accs.avg:.3f} '.format(accs=accs))
        print('confusion matrix\n', confusion_matrix)
    return accs.avg, losses.avg, confusion_matrix


def main():
    train_x_path = r'E:\Users\Aa\Desktop\UP-Fall-Dataset\YOLOV8\pose_train_test\17_subjects\split_1\train_x_1'
    train_y_path = r'E:\Users\Aa\Desktop\UP-Fall-Dataset\YOLOV8\pose_train_test\17_subjects\split_1\train_y_1'
    test_x_path = r'E:\Users\Aa\Desktop\UP-Fall-Dataset\YOLOV8\pose_train_test\17_subjects\split_1\test_x_1'
    test_y_path = r'E:\Users\Aa\Desktop\UP-Fall-Dataset\YOLOV8\pose_train_test\17_subjects\split_1\test_y_1'
    BATCH_SIZE = 32
    EPOCHS = 100
    best_acc = 0
    best_epoch = 0
    best_model = None
    is_header = False
    train_dataset = PoseDataset(data_path=train_x_path, label_path=train_y_path)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    test_dataset = PoseDataset(data_path=test_x_path, label_path=test_y_path)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = TCNTransformer()

    if torch.cuda.is_available():
        model = model.cuda()

    weights = torch.tensor([1.0, 30.0])
    if torch.cuda.is_available():
        weights = weights.cuda()

    criterion = FocalLoss(weights=weights)
    # criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(EPOCHS):
        # train for one epoch
        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch)
        valid_acc, valid_loss, confusion_matrix = validate(test_loader, model, criterion, epoch)

        if (epoch + 1 >= EPOCHS - 10) and valid_acc > best_acc:
            best_acc = valid_acc
            best_model = copy.deepcopy(model)
            best_epoch = epoch + 1

        if epoch == EPOCHS - 1:
            torch.save(best_model.state_dict(), f"./loss_compare/TCN_transformer_cew_split_1_{best_epoch}.pth")
            print("best acc: ", best_acc)
            print("best epoch: ", best_epoch)

        # 将每个epoch的confusion matrix保存下来
        with open("loss_compare/train_tcn_transformer_cew_split_1.csv", 'a+', newline='') as f:
            rowIni = list()
            my_writer = csv.writer(f)
            if not is_header:
                my_writer.writerow(
                    ['Epoch', "TN", "FP", "FN", "TP", "train_acc", "train_loss", "valid_acc", "valid_loss"])
                is_header = True
            rowIni.append(str(epoch + 1))
            rowIni.extend(confusion_matrix.flatten().tolist())
            rowIni.extend([train_acc, train_loss, valid_acc, valid_loss])
            my_writer.writerow(rowIni)


if __name__ == '__main__':
    print(torch.cuda.is_available())
    main()













