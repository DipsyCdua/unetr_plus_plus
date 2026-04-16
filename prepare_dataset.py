from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable


RAW_ROOT = Path("./TopCoW2024_Data_Release")
IMAGES_TR = RAW_ROOT / "imagesTr"
IMAGES_TS = RAW_ROOT / "imagesVal"
LABELS_TR = RAW_ROOT / "cow_seg_labelsTr"

DATASET_BASE = Path("./DATASET_TopCoW")
TASK_NAME = "Task010_TopCoW"
DEFAULT_MODALITY = "ct"
TRAIN_LIMIT = None
TEST_LIMIT = None

MODALITY_MAP = {
    "ct": "CTA",
    "mr": "MRA",
}

SOURCE_LABEL_SPECS = [
    ("background", 0),
    ("BA", 1),
    ("R_PCA", 2),
    ("L_PCA", 3),
    ("R_ICA", 4),
    ("R_MCA", 5),
    ("L_ICA", 6),
    ("L_MCA", 7),
    ("R_Pcom", 8),
    ("L_Pcom", 9),
    ("ACom", 10),
    ("R_ACA", 11),
    ("L_ACA", 12),
    ("ThirdA2", 15),
]

TARGET_LABELS = {str(new_value): name for new_value, (name, _) in enumerate(SOURCE_LABEL_SPECS)}
SOURCE_TO_TARGET_LABEL = {
    source_value: target_value
    for target_value, (_, source_value) in enumerate(SOURCE_LABEL_SPECS)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TopCoW 2024 data to the raw task layout expected by this UNETR++ repository.",
    )
    parser.add_argument(
        "--modality",
        choices=sorted(MODALITY_MAP),
        default=DEFAULT_MODALITY,
        help="Prepare a single-modality task. Mixing CTA and MRA in one channel is not supported here.",
    )
    parser.add_argument(
        "--task-name",
        default=TASK_NAME,
        help="Output raw task folder name. Must follow the TaskXXX_Name convention.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DATASET_BASE),
        help="Dataset root that will contain unetr_pp_raw/unetr_pp_raw_data/<task-name>.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=TRAIN_LIMIT,
        help="Optional cap on the number of training cases after modality filtering.",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=TEST_LIMIT,
        help="Optional cap on the number of validation/test cases after modality filtering.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output task directory instead of aborting.",
    )
    return parser.parse_args()


def validate_task_name(task_name: str) -> None:
    if not task_name.startswith("Task") or len(task_name) < 9:
        raise ValueError("task_name must follow the TaskXXX_Name convention, for example Task010_TopCoW")
    if task_name[4:7].isdigit() is False or task_name[7] != "_":
        raise ValueError("task_name must follow the TaskXXX_Name convention, for example Task010_TopCoW")


def validate_input_dirs() -> None:
    required_dirs = [IMAGES_TR, IMAGES_TS, LABELS_TR]
    missing = [str(path) for path in required_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing TopCoW input directories: {missing}")


def require_nibabel():
    try:
        import nibabel as nib
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "prepare_dataset.py requires nibabel and numpy. Install the dependencies from requirements.txt first."
        ) from exc
    return nib, np


def case_id_from_image(path: Path) -> str:
    name = path.name
    if not name.endswith(".nii.gz"):
        raise ValueError(f"Unsupported file name: {path.name}")
    name = name[:-7]
    if not name.endswith("_0000"):
        raise ValueError(f"TopCoW image file should end with _0000.nii.gz: {path.name}")
    return name[:-5]


def iter_cases(split_dir: Path, modality: str, limit: int | None = None) -> Iterable[tuple[str, Path]]:
    matching_files = []
    for img_path in sorted(split_dir.glob("*.nii.gz")):
        case_id = case_id_from_image(img_path)
        parts = case_id.split("_")
        if len(parts) != 3:
            raise ValueError(f"Unexpected TopCoW case id format: {case_id}")
        if parts[1] != modality:
            continue
        matching_files.append((case_id, img_path))
    if limit is not None:
        matching_files = matching_files[:limit]
    return matching_files


def remap_label_file(src: Path, dst: Path) -> None:
    nib, np = require_nibabel()
    img = nib.load(str(src))
    data = np.asarray(img.dataobj)
    unique_values = np.unique(data)
    unexpected = [int(v) for v in unique_values if int(v) not in SOURCE_TO_TARGET_LABEL]
    if unexpected:
        raise ValueError(f"Label file {src} contains unexpected values: {unexpected}")

    remapped = data.astype(np.int16, copy=True)
    for old_value, new_value in SOURCE_TO_TARGET_LABEL.items():
        if old_value == new_value:
            continue
        remapped[data == old_value] = new_value

    header = img.header.copy()
    header.set_data_dtype(remapped.dtype)
    nib.save(nib.Nifti1Image(remapped, img.affine, header), str(dst))


def build_dataset_json(modality: str, train_case_ids: list[str], test_case_ids: list[str]) -> dict:
    return {
        "name": "TopCoW",
        "description": f"TopCoW 2024 {MODALITY_MAP[modality]} vessel segmentation prepared for UNETR++",
        "tensorImageSize": "3D",
        "reference": "TopCoW Challenge 2024",
        "licence": "TopCoW Open Data (non-commercial use)",
        "release": "2024",
        "modality": {
            "0": MODALITY_MAP[modality],
        },
        "labels": TARGET_LABELS,
        "numTraining": len(train_case_ids),
        "numTest": len(test_case_ids),
        "training": [
            {
                "image": f"./imagesTr/{case_id}.nii.gz",
                "label": f"./labelsTr/{case_id}.nii.gz",
            }
            for case_id in train_case_ids
        ],
        "test": [
            f"./imagesTs/{case_id}.nii.gz"
            for case_id in test_case_ids
        ],
    }


def prepare_output_dir(dst_root: Path, overwrite: bool) -> None:
    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"{dst_root} already exists. Re-run with --overwrite to avoid mixing old and new prepared files."
            )
        shutil.rmtree(dst_root)
    (dst_root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dst_root / "labelsTr").mkdir(exist_ok=True)
    (dst_root / "imagesTs").mkdir(exist_ok=True)


def copy_split(modality: str, dst_root: Path, train_limit: int | None, test_limit: int | None) -> tuple[list[str], list[str]]:
    train_case_ids: list[str] = []
    test_case_ids: list[str] = []

    for case_id, img_path in iter_cases(IMAGES_TR, modality=modality, limit=train_limit):
        label_path = LABELS_TR / f"{case_id}.nii.gz"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for training case {case_id}: {label_path}")
        shutil.copy2(img_path, dst_root / "imagesTr" / f"{case_id}_0000.nii.gz")
        remap_label_file(label_path, dst_root / "labelsTr" / f"{case_id}.nii.gz")
        train_case_ids.append(case_id)

    for case_id, img_path in iter_cases(IMAGES_TS, modality=modality, limit=test_limit):
        shutil.copy2(img_path, dst_root / "imagesTs" / f"{case_id}_0000.nii.gz")
        test_case_ids.append(case_id)

    return train_case_ids, test_case_ids


def validate_split(train_case_ids: list[str], test_case_ids: list[str], modality: str) -> None:
    if not train_case_ids:
        raise RuntimeError(f"No training cases were found for modality '{modality}'.")
    if not test_case_ids:
        raise RuntimeError(f"No test cases were found for modality '{modality}'.")
    overlap = sorted(set(train_case_ids) & set(test_case_ids))
    if overlap:
        raise RuntimeError(f"Train/test split is invalid because these cases appear in both splits: {overlap}")


def write_dataset_json(dst_root: Path, modality: str, train_case_ids: list[str], test_case_ids: list[str]) -> None:
    dataset_json = build_dataset_json(modality, train_case_ids, test_case_ids)
    with open(dst_root / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    validate_input_dirs()
    validate_task_name(args.task_name)

    output_root = Path(args.output_root)
    dst_root = output_root / "unetr_pp_raw" / "unetr_pp_raw_data" / args.task_name

    prepare_output_dir(dst_root, overwrite=args.overwrite)
    train_case_ids, test_case_ids = copy_split(
        modality=args.modality,
        dst_root=dst_root,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
    )
    validate_split(train_case_ids, test_case_ids, args.modality)
    write_dataset_json(
        dst_root=dst_root,
        modality=args.modality,
        train_case_ids=train_case_ids,
        test_case_ids=test_case_ids,
    )

    print(f"Prepared {args.modality} dataset at {dst_root}")
    print(f"Training cases: {len(train_case_ids)}")
    print(f"Test cases: {len(test_case_ids)}")


if __name__ == "__main__":
    main()
