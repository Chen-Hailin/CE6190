import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict
import os, errno
from multiprocessing import Pool
import argparse


data_root = '/export/home/CE6190_ass_1/dataset/games'

original_colors = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [20, 20, 20], [111, 74, 0], [81, 0, 81], [128, 64, 128], [244, 35, 232], [250, 170, 160], [230, 150, 140], [70, 70, 70], [102, 102, 156], [190, 153, 153], [180, 165, 180], [150, 100, 100], [150, 120, 90], [153, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 0, 90], [0, 0, 110], [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 142]]

train_palette = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]

color_to_trainID = defaultdict(lambda: 255)
color_to_trainID.update({str(color):trainId for trainId,color in enumerate(train_palette)})

def pixel_to_trainID(pixel):
    return color_to_trainID[str(original_colors[pixel])]

pixel_to_trainID_map = {pixel:pixel_to_trainID(pixel) for pixel in range(len(original_colors))}
pixel_2_trainID = lambda x: pixel_to_trainID_map[x]
vectorized_pixel_2_trainID = np.vectorize(pixel_2_trainID)

def single_process_l2Id(f_path):
    file_name = os.path.basename(f_path)
    im = np.array(Image.open(f_path))
    new_im = vectorized_pixel_2_trainID(im)
    Image.fromarray(new_im.astype(np.uint8)).save(os.path.join(data_root, 'trainId_labels', file_name))

def multiprocessing_convert_label_to_trainId():
    # convert original label to cityscape-like label ID
    file_paths = glob.glob(os.path.join(data_root, 'source_labels')+'/*.png')
    with Pool(20) as p:
        r = list(tqdm(p.imap(single_process_l2Id, file_paths), total=2500))

def convert_label_to_trainId():
    file_paths = glob.glob(os.path.join(data_root, 'source_labels')+'/*.png')
    print(pixel_to_trainID_map)
    for idx,f_path in tqdm(enumerate(file_paths)):
        file_name = os.path.basename(f_path)
        im = np.array(Image.open(f_path))
        new_im = vectorized_pixel_2_trainID(im)
        new_im = Image.fromarray(new_im.astype(np.uint8)).save(os.path.join(data_root, 'trainId_labels', file_name))


def check_img_size():
    file_paths = glob.glob(os.path.join(data_root, 'source_images')+'/*.png')
    sizes = set()
    for f_path in tqdm(file_paths):
        im = np.array(Image.open(f_path))
        sizes.add(str(im.shape))

    print(f'sizes include: {sizes}') # sizes include: {'(1052, 1914, 3)'}

def check_label_range():
    file_paths = glob.glob(os.path.join(data_root, 'source_labels')+'/*.png')
    labels = Counter()
    for idx,f_path in tqdm(enumerate(file_paths)):
        if idx > 200:
            break
        im = np.array(Image.open(os.path.join(data_root, 'source_labels', f_path)))
        labels.update(im.flatten())

    print(f'labels length: {len(labels)}, counter: {labels}') # labels length: 29, counter: Counter({7: 158735599, 23: 60745163, 11: 43893863, 1: 41123282, 21: 24275459, 8: 12381530, 22: 10545083, 0: 9004240, 26: 8563559, 12: 7128179, 5: 4006175, 17: 3953523, 15: 3689052, 27: 3536722, 16: 3126028, 13: 2653817, 31: 2363884, 28: 1293866, 4: 987702, 14: 440496, 20: 434613, 19: 432669, 24: 416504, 6:357768, 30: 287287, 32: 194817, 25: 143153, 33: 4690, 34: 405})

def split_train_valid():
    for idx in tqdm(range(1, 2501)):
        if idx >= 2300: # val split
            split = 'val'
        else:
            split = 'train'
        
        # image
        img_src = os.path.join(data_root, 'source_images', str(idx).zfill(5)+'.png')
        img_dst = os.path.join(data_root, 'images', split, f'{split}_'+str(idx).zfill(5)+'_image.png')
        os.symlink(img_src, img_dst)

        # labels
        label_src = os.path.join(data_root, 'trainId_labels',str(idx).zfill(5)+'.png')
        label_dst = os.path.join(data_root, 'labels', split, f'{split}_'+str(idx).zfill(5)+'_labelIds.png')
        try:
            os.symlink(label_src, label_dst)
        except Exception as e:
            if e.errno == errno.EEXIST:
                os.remove(label_dst)
                os.symlink(label_src, label_dst)

if __name__ == '__main__':
    # check_label_range()
    multiprocessing_convert_label_to_trainId()
    split_train_valid()