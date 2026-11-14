import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from src.archive_safety import extract_zip_safely
from src.runtime import normalize_label_names, square_image_size


class RuntimeArtifactTests(unittest.TestCase):
    def test_label_list_is_preserved(self):
        self.assertEqual(normalize_label_names(["A", "B"]), ["A", "B"])

    def test_legacy_label_tuple_uses_names(self):
        self.assertEqual(normalize_label_names((["A", "B"], {"A": 0})), ["A", "B"])

    def test_invalid_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            normalize_label_names("person")

    def test_model_feature_count_determines_image_dimensions(self):
        self.assertEqual(square_image_size(10_000), 100)
        self.assertEqual(square_image_size(4_096), 64)

    def test_non_square_model_input_is_rejected(self):
        with self.assertRaises(ValueError):
            square_image_size(9_999)


class SafeArchiveTests(unittest.TestCase):
    def test_extracts_valid_member(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "safe.zip"
            destination = root / "out"
            with ZipFile(archive, "w") as zf:
                zf.writestr("faces/person/image.jpg", b"image")
            extract_zip_safely(archive, destination)
            self.assertEqual((destination / "faces/person/image.jpg").read_bytes(), b"image")

    def test_rejects_parent_traversal_member(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "unsafe.zip"
            destination = root / "out"
            with ZipFile(archive, "w") as zf:
                zf.writestr("../../outside.txt", b"do not write")
            with self.assertRaisesRegex(ValueError, "Unsafe ZIP member path"):
                extract_zip_safely(archive, destination)
            self.assertFalse((root.parent / "outside.txt").exists())


if __name__ == "__main__":
    unittest.main()
