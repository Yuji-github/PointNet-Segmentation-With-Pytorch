import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from pointnet_segmentation import PointNetSeg, Loss
from datasets_loader import Data

root_dir = ""

def data_loader():
    dset = Data(root_dir, transform=None)
    train_num = int(len(dset) * 0.95)
    val_num = int(len(dset) * 0.05)
    if int(len(dset)) - train_num - val_num > 0:
        train_num = train_num + 1
    elif int(len(dset)) - train_num - val_num < 0:
        train_num = train_num - 1

    train_dataset, val_dataset = random_split(dset, [train_num, val_num])
    val_dataset.valid = True

    print('######### Dataset class created #########')
    print('Number of images: ', len(dset))
    print('Sample image shape: ', dset[0]['image'].shape)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64)
    return train_loader, val_loader

if __name__ == "__main__":

    train_loader, val_loader= data_loader()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pointnet = PointNetSeg()
    pointnet.to(device)
    optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.001)
    epochs = 5
    save =True

    for epoch in range(epochs):
        pointnet.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['image'].to(device), data['category'].to(device)
            optimizer.zero_grad()
            outputs, matrix_3x3, matrix_kxk = pointnet(inputs.transpose(1, 2))

            loss = Loss(outputs, labels, matrix_3x3, matrix_kxk)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

        pointnet.eval()
        correct = total = 0

        # validation
        if val_loader:
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data['image'].to(device), data['category'].to(device)
                    outputs, __, __ = pointnet(inputs.transpose(1, 2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0) * labels.size(1)  ##
                    correct += (predicted == labels).sum().item()
            val_acc = 100 * correct / total
            print('Valid accuracy: %d %%' % val_acc)

        # save the model
        if save:
            torch.save(pointnet.state_dict(), root_dir + "/modelsSeg/" + str(epoch) + "_" + str(val_acc))
