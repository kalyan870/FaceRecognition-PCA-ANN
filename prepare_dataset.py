"""
Download and prepare the face dataset.
Organizes into train/test (60/40 split) under dataset/.
"""
import os
import shutil
import urllib.request
import zipfile
from sklearn.model_selection import train_test_split

DATASET_URL = "https://github.com/robaita/introduction_to_machine_learning/raw/main/dataset.zip"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
IMPOSTER_DIR = os.path.join(BASE_DIR, "dataset", "imposters")

def main():
    # Create directories
    for d in [TRAIN_DIR, TEST_DIR, IMPOSTER_DIR]:
        os.makedirs(d, exist_ok=True)

    # Download if needed
    zip_path = os.path.join(BASE_DIR, "dataset.zip")
    if not os.path.exists(zip_path):
        print("[INFO] Downloading dataset...")
        urllib.request.urlretrieve(DATASET_URL, zip_path)
        print("[INFO] Download complete.")

    # Extract
    extract_dir = os.path.join(BASE_DIR, "dataset_extracted")
    if not os.path.exists(extract_dir):
        print("[INFO] Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
        print("[INFO] Extraction complete.")

    # Find faces directory
    faces_dir = os.path.join(extract_dir, "dataset", "faces")
    if not os.path.exists(faces_dir):
        # Try alternate structure
        faces_dir = os.path.join(extract_dir, "faces")
    if not os.path.exists(faces_dir):
        print("[ERROR] Could not find faces directory.")
        return

    persons = sorted(os.listdir(faces_dir))
    print(f"[INFO] Found {len(persons)} persons: {persons}")

    for person in persons:
        person_dir = os.path.join(faces_dir, person)
        if not os.path.isdir(person_dir):
            continue

        images = sorted([
            os.path.join(person_dir, f)
            for f in os.listdir(person_dir)
            if os.path.isfile(os.path.join(person_dir, f))
        ])

        if len(images) < 2:
            print(f"[WARN] {person}: only {len(images)} images, skipping.")
            continue

        train_imgs, test_imgs = train_test_split(
            images, test_size=0.4, random_state=42, shuffle=True
        )

        train_person_dir = os.path.join(TRAIN_DIR, person)
        test_person_dir = os.path.join(TEST_DIR, person)
        os.makedirs(train_person_dir, exist_ok=True)
        os.makedirs(test_person_dir, exist_ok=True)

        for img_path in train_imgs:
            shutil.copy2(img_path, train_person_dir)
        for img_path in test_imgs:
            shutil.copy2(img_path, test_person_dir)

        print(f"  {person}: {len(train_imgs)} train, {len(test_imgs)} test")

    print(f"\n[INFO] Dataset ready!")
    print(f"  Train: {TRAIN_DIR}")
    print(f"  Test:  {TEST_DIR}")

    # Cleanup
    shutil.rmtree(extract_dir, ignore_errors=True)
    os.remove(zip_path)

    print("[INFO] Done!")

if __name__ == "__main__":
    main()
