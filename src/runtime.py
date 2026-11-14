from math import isqrt


def normalize_label_names(value):
    """Normalize supported saved label formats to an ordered list of names."""
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("The labels artifact must contain a non-empty list of names.")
    return [str(label) for label in value]


def square_image_size(feature_count):
    """Return the image width/height represented by a flattened square image."""
    if not isinstance(feature_count, int) or feature_count <= 0:
        raise ValueError("The model must declare a positive input feature count.")
    side = isqrt(feature_count)
    if side * side != feature_count:
        raise ValueError(
            f"Expected a flattened square grayscale image; model expects {feature_count} features."
        )
    return side
