import json
from pathlib import Path


def convert_coco_to_yolo(coco_json, images_dir, labels_dir):
    labels_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json, "r") as f:
        coco = json.load(f)

    # category_id → 0..N-1
    categories = sorted(coco["categories"], key=lambda x: x["id"])
    cat_map = {c["id"]: i for i, c in enumerate(categories)}

    images = {img["id"]: img for img in coco["images"]}

    anns_by_image = {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for img_id, img in images.items():
        img_w, img_h = img["width"], img["height"]
        img_name = Path(img["file_name"]).stem
        label_file = labels_dir / f"{img_name}.txt"

        lines = []
        for ann in anns_by_image.get(img_id, []):
            cls = cat_map[ann["category_id"]]
            x, y, w, h = ann["bbox"]

            xc = (x + w / 2) / img_w
            yc = (y + h / 2) / img_h
            w /= img_w
            h /= img_h

            lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        label_file.write_text("\n".join(lines))

    print(f"Converted {coco_json.name}")
    print("Classes:")
    for i, c in enumerate(categories):
        print(i, c["name"])


if __name__ == "__main__":
    base = Path("datasets/coco")

    convert_coco_to_yolo(
        base / "annotations" / "instances_train2017.json",
        base / "train2017",
        base / "labels" / "train",
    )

    convert_coco_to_yolo(
        base / "annotations" / "instances_val2017.json",
        base / "val2017",
        base / "labels" / "val",
    )
