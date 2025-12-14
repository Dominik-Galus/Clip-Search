import os
import subprocess
from multiprocessing import Pool, cpu_count
from pathlib import Path

PATH: str = "data/test"
TARGET: str = "converted_data/test"


def convert_file(file_info: tuple[str, str]) -> None:
    source_file, target_file = file_info
    subprocess.run([
        "ffmpeg",
        "-i", source_file,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-y",
        target_file
    ])
    print(f"Converted: {source_file} to {target_file}")


def convert_data(path: str) -> None:
    file_list: list[tuple[str, str]] = []

    for root, _, files in os.walk(path):
        dest_path = os.path.join(TARGET, os.path.relpath(root, path))
        os.makedirs(dest_path, exist_ok=True)

        for file in files:
            if file.lower().endswith(".avi"):
                source_file = os.path.join(root, file)
                target_file = os.path.join(dest_path, Path(file).stem + ".mp4")
                file_list.append((source_file, target_file))

    with Pool(processes=cpu_count()) as pool:
        pool.map(convert_file, file_list)


if __name__ == "__main__":
    convert_data(PATH)
