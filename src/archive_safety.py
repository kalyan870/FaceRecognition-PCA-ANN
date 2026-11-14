from pathlib import Path
from zipfile import ZipFile


def extract_zip_safely(archive_path, destination):
    """Extract a ZIP only when every member resolves inside the destination."""
    destination_path = Path(destination).resolve()
    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination_path / member.filename).resolve()
            try:
                target.relative_to(destination_path)
            except ValueError as exc:
                raise ValueError(f"Unsafe ZIP member path: {member.filename!r}") from exc
        archive.extractall(destination_path)
