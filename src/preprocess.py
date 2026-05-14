import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils import IMG_SIZE, load_images_from_folder

def load_dataset(train_dir, test_dir):
    person_names = sorted(os.listdir(train_dir))
    X_train, y_train, train_files = [], [], []
    label_map = {}
    for idx, person in enumerate(person_names):
        person_path = os.path.join(train_dir, person)
        if not os.path.isdir(person_path):
            continue
        label_map[person] = idx
        imgs, _, fnames = load_images_from_folder(person_path, label=idx)
        X_train.extend(imgs)
        y_train.extend([idx] * len(imgs))
        train_files.extend(fnames)

    X_test, y_test, test_files = [], [], []
    test_person_names = sorted(os.listdir(test_dir))
    test_label_map = {}
    for person in test_person_names:
        person_path = os.path.join(test_dir, person)
        if not os.path.isdir(person_path):
            continue
        if person not in label_map:
            continue
        test_label_map[person] = label_map[person]
        imgs, _, fnames = load_images_from_folder(person_path, label=label_map[person])
        X_test.extend(imgs)
        y_test.extend([label_map[person]] * len(imgs))
        test_files.extend(fnames)

    X_train = np.array(X_train, dtype=np.float64)
    X_test = np.array(X_test, dtype=np.float64)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print(f"[INFO] Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"[INFO] Image vector size: {X_train.shape[1]}")
    print(f"[INFO] Number of classes: {len(label_map)}")
    print(f"[INFO] Classes: {list(label_map.keys())}")

    return X_train, X_test, y_train, y_test, label_map, train_files, test_files

def load_imposter_images(imposter_dir, img_size=IMG_SIZE):
    imgs, _, fnames = load_images_from_folder(imposter_dir)
    if len(imgs) == 0:
        return np.array([])
    return np.array(imgs, dtype=np.float64), fnames
