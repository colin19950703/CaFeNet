from glob import glob
from collections import Counter
from torchvision import transforms
import cv2
import torch.utils.data as data
import numpy as np

def print_number_of_sample(data_set, prefix):
    def fill_empty_label(label_dict):
        for i in range(max(label_dict.keys()) + 1):
            if label_dict[i] != 0:
                continue
            else:
                label_dict[i] = 0
        return dict(sorted(label_dict.items()))

    data_label = [data_set[i][1] for i in range(len(data_set))]
    d = Counter(data_label)
    d = fill_empty_label(d)
    print("%-7s" % prefix, d)
    data_label = [d[key] for key in d.keys()]

    return data_label

def load_data_info(pathname, gt_list=None):
    file_list = glob(pathname)
    label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]

    if gt_list is not None:
        label_list = [gt_list[i] for i in label_list]

    return list(zip(file_list, label_list))

def prepare_colon_data(data_root_dir):
    set_1010711 = load_data_info('%s/1010711/*.jpg' % data_root_dir)
    set_1010712 = load_data_info('%s/1010712/*.jpg' % data_root_dir)
    set_1010713 = load_data_info('%s/1010713/*.jpg' % data_root_dir)
    set_1010714 = load_data_info('%s/1010714/*.jpg' % data_root_dir)
    set_1010715 = load_data_info('%s/1010715/*.jpg' % data_root_dir)
    set_1010716 = load_data_info('%s/1010716/*.jpg' % data_root_dir)
    wsi_00016 = load_data_info('%s/wsi_00016/*.jpg' % data_root_dir)  # benign exclusively
    wsi_00017 = load_data_info('%s/wsi_00017/*.jpg' % data_root_dir)  # benign exclusively
    wsi_00018 = load_data_info('%s/wsi_00018/*.jpg' % data_root_dir)  # benign exclusively

    train_set = set_1010711 + set_1010712 + set_1010713 + set_1010715 + wsi_00016
    valid_set = set_1010716 + wsi_00018
    test_set = set_1010714 + wsi_00017

    print_number_of_sample(train_set, 'Train')
    print_number_of_sample(valid_set, 'Valid')
    print_number_of_sample(test_set, 'Test1')
    return train_set, valid_set, test_set

def prepare_colon_test2_data(data_root_dir):
    gt_list = { 0: 5,  # "BN", #0
                1: 0,  # "TLS", #0
                2: 1,  # "TW", #2
                3: 2,  # "TM", #3
                4: 3,  # "TP", #4
                }

    test_set = load_data_info('%s/*/*/*.png' % data_root_dir, gt_list)

    print_number_of_sample(test_set, 'Test2')
    return test_set

class DatasetSerial(data.Dataset):
    def __init__(self, pair_list, shape_augs=None, input_augs=None):
        self.pair_list = pair_list
        self.shape_augs = shape_augs
        self.input_augs = input_augs

    def __getitem__(self, idx):
        pair = self.pair_list[idx]
        # print(pair)
        input_img = cv2.imread(pair[0])
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        img_label = pair[1]
        # print(input_img.shape)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0., 0., 0.],
                                 std=[1., 1., 1.])
        ])

        if self.shape_augs is not None:
            shape_augs = self.shape_augs.to_deterministic()
            input_img = shape_augs.augment_image(input_img)

        if self.input_augs is not None:
            input_img = self.input_augs.augment_image(input_img)

        input_img = np.array(input_img).copy()
        out_img = np.array(transform(input_img)).transpose(1, 2, 0)

        return np.array(out_img), img_label

    def __len__(self):
        return len(self.pair_list)

if __name__ == '__main__':
    print('\nColoectal')
    prepare_colon_data()
    prepare_colon_test2_data()