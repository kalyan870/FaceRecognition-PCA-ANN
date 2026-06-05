"""
Download and organize the face dataset.
"""
import os
import zipfile
import urllib.request
import shutil

DATASET_URL = "https://github.com/robaita/introduction_to_machine_learning/blob/main/dataset.zip"
DATASET_ZIP = "dataset.zip"
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

def download_dataset():
    if os.path.exists(TRAIN_DIR) and len(os.listdir(TRAIN_DIR)) > 0:
        print("[INFO] Dataset already exists. Skipping download.")
        return True

    print(f"[INFO] Downloading dataset from {DATASET_URL}...")
    try:
        urllib.request.urlretrieve(DATASET_URL, DATASET_ZIP)
        print(f"[INFO] Downloaded {DATASET_ZIP}")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print("[INFO] Please manually download and extract the dataset.")
        return False

    print("[INFO] Extracting dataset...")
    try:
        with zipfile.ZipFile(DATASET_ZIP, 'r') as zf:
            zf.extractall("dataset_raw")
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        return False

    os.remove(DATASET_ZIP)
    print("[INFO] Dataset ready!")
    return True

def organize_dataset():
    """
    This function should organize raw images into:
      dataset/train/person1/img1.jpg
      dataset/train/person2/img1.jpg
      dataset/test/person1/img1.jpg
      dataset/test/person2/img1.jpg
    Adjust the logic based on actual dataset structure after extraction.
    """
    # Placeholder — adjust based on your extracted dataset structure
    if not os.path.exists("dataset_raw"):
        print("[WARN] No dataset_raw directory found. Skipping organization.")
        return False

    print("[INFO] Organizing dataset into train/test (60/40 split)...")
    from sklearn.model_selection import train_test_split
    import glob

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # Assumes structure: dataset_raw/<person>/<images>
    persons = sorted(os.listdir("dataset_raw"))
    for person in persons:
        person_dir = os.path.join("dataset_raw", person)
        if not os.path.isdir(person_dir):
            continue
        images = glob.glob(os.path.join(person_dir, "*"))
        if not images:
            continue

        train_imgs, test_imgs = train_test_split(images, test_size=0.4, random_state=42)

        train_person_dir = os.path.join(TRAIN_DIR, person)
        test_person_dir = os.path.join(TEST_DIR, person)
        os.makedirs(train_person_dir, exist_ok=True)
        os.makedirs(test_person_dir, exist_ok=True)

        for img_path in train_imgs:
            shutil.copy(img_path, train_person_dir)
        for img_path in test_imgs:
            shutil.copy(img_path, test_person_dir)

        print(f"  {person}: {len(train_imgs)} train, {len(test_imgs)} test")

    shutil.rmtree("dataset_raw", ignore_errors=True)
    print("[INFO] Dataset organized successfully!")
    return True

if __name__ == "__main__":
    if download_dataset():
        organize_dataset()
