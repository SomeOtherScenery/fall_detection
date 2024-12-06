import torch.utils.data as data
import numpy as np



def load_x(file_name):
    temp = np.memmap(file_name, dtype='float32', mode='r')
    temp = np.array(temp)
    x = np.reshape(temp, [-1, 18, 24])
    return x


def load_y(file_name):
    temp = np.memmap(file_name, dtype='int8', mode='r')
    temp = np.array(temp)
    y = np.reshape(temp, [-1, 2])
    return y


def onehot2dense(x):
    y = np.where(x > 0)
    return y

# pose.shape [18, 24]
def pose_scale_norm(pose):
    poseX = pose[:, ::2]
    poseX_min = np.min(poseX, axis=1, keepdims=True)
    poseX_max = np.max(poseX, axis=1, keepdims=True)
    # 20231130  avoid 0 / 0 error
    poseX_norm = (poseX - poseX_min) / (poseX_max - poseX_min+0.0001)

    poseY = pose[:, 1::2]
    poseY_min = np.min(poseY, axis=1, keepdims=True)
    poseY_max = np.max(poseY, axis=1, keepdims=True)
    # 20231130   avoid 0 / 0 error
    poseY_norm = (poseY - poseY_min) / (poseY_max - poseY_min+0.0001)

    pose_norm = np.zeros_like(pose)
    pose_norm[:, ::2] = poseX_norm
    pose_norm[:, 1::2] = poseY_norm
    return pose_norm


class PoseDataset(data.Dataset):
    def __init__(self, data_path, label_path, transform=pose_scale_norm):
        self.poses = load_x(data_path)
        self.labels = load_y(label_path)
        self.transform = transform

    def __getitem__(self, index):
        pose = self.poses[index]
        # do pose normalization, pose.shape = (18, 24)
        if self.transform is not None:
            pose = self.transform(pose)
        pose = pose.transpose((1, 0))
        label = self.labels[index]
        label = onehot2dense(label)
        label = label[0].item()
        return pose, label

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    data_path = r'E:\Users\Aa\Desktop\UP-Fall-Dataset\YOLOV8\pose_train_test\17_subjects\test_x'
    label_path = r'E:\Users\Aa\Desktop\UP-Fall-Dataset\YOLOV8\pose_train_test\17_subjects\test_y'
    train_dataset = PoseDataset(data_path, label_path)
    for item in train_dataset:
        print(item[0])
        # print(item[0].shape)
        break