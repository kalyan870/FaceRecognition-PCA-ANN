import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from src.train import train_pipeline
from src.test import test_folder, test_imposters, test_single_image
from src.preprocess import load_dataset
from src.utils import ensure_dir
import joblib

def main():
    parser = argparse.ArgumentParser(description='Face Recognition using PCA + ANN')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['train', 'test', 'full'],
                        help='Operation mode')
    parser.add_argument('--train_dir', type=str, default='dataset/train',
                        help='Training dataset directory')
    parser.add_argument('--test_dir', type=str, default='dataset/test',
                        help='Testing dataset directory')
    parser.add_argument('--imposter_dir', type=str, default='dataset/imposters',
                        help='Imposter images directory')
    parser.add_argument('--image', type=str, default=None,
                        help='Single image path for testing')
    parser.add_argument('--k_values', type=int, nargs='+', default=[5, 10, 20, 30, 40],
                        help='k values for PCA')
    parser.add_argument('--best_k', type=int, default=None,
                        help='Use specific k value for training')
    args = parser.parse_args()

    ensure_dir('outputs/graphs')
    ensure_dir('outputs/eigenfaces')
    ensure_dir('outputs/predictions')
    ensure_dir('outputs/models')

    if args.mode in ('full', 'train'):
        print("=" * 60)
        print("FACE RECOGNITION SYSTEM - PCA + ANN")
        print("=" * 60)
        best_pca, best_ann, label_map, k_values, accuracies = train_pipeline(
            args.train_dir, args.test_dir, args.k_values
        )
        print(f"\n[INFO] Final model saved to outputs/models/")

    if args.mode in ('full', 'test'):
        print("\n" + "=" * 60)
        print("TESTING PHASE")
        print("=" * 60)

        if os.path.exists('outputs/models/final_pca.joblib'):
            pca = joblib.load('outputs/models/final_pca.joblib')
            ann = joblib.load('outputs/models/final_ann.joblib')
            _, _, _, _, label_map, _, _ = load_dataset(args.train_dir, args.test_dir)
        else:
            print("[ERROR] No trained model found. Run with --mode train first.")
            return

        if args.image:
            test_single_image(args.image, pca, ann, label_map)
        else:
            test_folder(args.test_dir, pca, ann, label_map)

        if os.path.exists(args.imposter_dir):
            test_imposters(args.imposter_dir, pca, ann)

if __name__ == '__main__':
    main()
