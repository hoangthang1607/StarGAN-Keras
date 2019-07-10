import os
import random 
import numpy as np 
import cv2 


class ImageData:

    def __init__(self, data_dir, selected_attrs):
        self.selected_attrs = selected_attrs

        self.data_path = os.path.join(data_dir, 'images')
        self.lines = open(os.path.join(data_dir, 'list_attr_celeba.txt'), 'r').readlines()

        self.train_dataset = []
        self.train_dataset_label = []
        self.train_dataset_fix_label = []

        self.test_dataset = []
        self.test_dataset_label = []
        self.test_dataset_fix_label = []

        self.attr2idx = {}
        self.idx2attr = {}

    def preprocess(self) :
        all_attr_names = self.lines[1].split()
        for i, attr_name in enumerate(all_attr_names) :
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name


        lines = self.lines[2:]
        random.seed(1234)
        random.shuffle(lines)

        for i, line in enumerate(lines) :
            split = line.split()
            filename = os.path.join(self.data_path, split[0])
            values = split[1:]

            label = []

            for attr_name in self.selected_attrs :
                idx = self.attr2idx[attr_name]

                if values[idx] == '1' :
                    label.append(1)
                else :
                    label.append(0)

            if i < 2000 :
                self.test_dataset.append(filename)
                self.test_dataset_label.append(label)
            else :
                self.train_dataset.append(filename)
                self.train_dataset_label.append(label)
            # ['./dataset/celebA/train/019932.jpg', [1, 0, 0, 0, 1]]

        self.test_dataset_fix_label = create_labels(self.test_dataset_label, self.selected_attrs)
        self.train_dataset_fix_label = create_labels(self.train_dataset_label, self.selected_attrs)

        print('\n Finished preprocessing the CelebA dataset...')

def create_labels(c_org, selected_attrs=None):
    """Generate target domain labels for debugging and testing."""
    # Get hair color indices.
    c_org = np.asarray(c_org)
    hair_color_indices = []
    for i, attr_name in enumerate(selected_attrs):
        if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
            hair_color_indices.append(i)

    c_trg_list = []

    for i in range(len(selected_attrs)):
        c_trg = c_org.copy()

        if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
            c_trg[:, i] = 1.0
            for j in hair_color_indices:
                if j != i:
                    c_trg[:, j] = 0.0
        else:
            c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.

        c_trg_list.append(c_trg)

    c_trg_list = np.transpose(c_trg_list, axes=[1, 0, 2]) # [c_dim, bs, ch]

    return c_trg_list

def resize_keep_aspect_ratio(image, width, height):
    (h, w) = image.shape[:2]
    dH = 0
    dW = 0
    if w < h:
        image = cv2.resize(image, (width, int(h*width/w)), interpolation = cv2.INTER_AREA)
        dH = int((image.shape[0] - height) / 2.0)
    else:
        image = cv2.resize(image, (int(w*height/h), height), interpolation = cv2.INTER_AREA)
        dW = int((image.shape[1] - width) / 2.0)
    (h, w) = image.shape[:2]
    image = image[dH:h-dH, dW:w-dW]
    return cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)

def get_loader(filenames, labels, fix_labels, image_size=128, batch_size=16, mode='train'):
    """Build and return a data loader."""
    n_batches = int(len(filenames) / batch_size)
    total_samples = n_batches * batch_size

    for i in range(n_batches):
        batch = filenames[i*batch_size:(i+1)*batch_size]
        imgs = []
        for p in batch:
            image = cv2.imread(p)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = resize_keep_aspect_ratio(image, width = image_size, height = image_size)
            if mode == 'train':
                proba = np.random.rand()
                if proba > 0.5:
                    image = cv2.flip(image, 1)
                            
            imgs.append(image)

        imgs = np.array(imgs)/127.5 - 1
        orig_labels = np.array(labels[i*batch_size:(i+1)*batch_size])
        target_labels = np.random.permutation(orig_labels)
        yield imgs, orig_labels, target_labels, fix_labels[i*batch_size:(i+1)*batch_size], batch