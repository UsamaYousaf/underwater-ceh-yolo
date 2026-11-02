# Underwater Object Detection with CEH-YOLO: Principles and Application

**Author:** Usama Yousaf, BTU Cottbus-Senftenberg

Source code accompanying the thesis. It is a fork of Ultralytics YOLOv8 (AGPL-3.0, see [ultralytics/LICENSE](ultralytics/LICENSE)) extended with the CEH-YOLO modules. Trained weights, training run outputs and the dataset images and labels are not included in this repository.

## Key files

| File | Role in the thesis |
|---|---|
| [ultralytics/ultralytics/nn/modules/ceh/DAttention.py](ultralytics/ultralytics/nn/modules/ceh/DAttention.py) | `DAttention`: deformable attention block. `HDA`: reconstruction of CEH-YOLO's HDA module (paper Section 3.2 / Fig. 4); splits the input into two channel segments, applies DAttention to each, and recombines them. Replaces the last C2f block of the YOLOv8n backbone. |
| [ultralytics/ultralytics/nn/modules/ceh/ESPPF.py](ultralytics/ultralytics/nn/modules/ceh/ESPPF.py) | `ESPPF`: enhanced spatial pyramid pooling module, replacing SPPF. |
| [ultralytics/ultralytics/nn/modules/ceh/CompositeDetect.py](ultralytics/ultralytics/nn/modules/ceh/CompositeDetect.py) | `CompositeDetect`: composite detection head module. |
| [ultralytics/ultralytics/nn/modules/\_\_init\_\_.py](ultralytics/ultralytics/nn/modules/__init__.py) | Registers ESPPF, DAttention, HDA and CompositeDetect as Ultralytics modules. |
| [ultralytics/ultralytics/nn/tasks.py](ultralytics/ultralytics/nn/tasks.py) | Imports the CEH modules so that `parse_model()` can build them from the model YAML configs. |
| [yolov8-ceh.yaml](ultralytics/ultralytics/cfg/models/v8/yolov8-ceh.yaml) | CEH-YOLO config: DAttention replaces the last C2f, ESPPF replaces SPPF. |
| [yolov8-ceh-hda.yaml](ultralytics/ultralytics/cfg/models/v8/yolov8-ceh-hda.yaml) | CEH-YOLO config with the HDA reconstruction (640 px input, q_size 20x20). |
| [yolov8-ceh-hda-1280.yaml](ultralytics/ultralytics/cfg/models/v8/yolov8-ceh-hda-1280.yaml) | HDA config for the high-resolution experiment (1280 px input, q_size 40x40). |
| [scripts/coco_to_yolo_detect.py](scripts/coco_to_yolo_detect.py) | Converts the UTDAC2020 COCO-format annotations to YOLO detection labels. |
| [datasets/coco/data.yaml](datasets/coco/data.yaml) | Dataset definition: 4 classes (echinus, starfish, holothurian, scallop). Expects `images/` and `labels/` under `datasets/coco/`. |
| [CEH-YOLO/](CEH-YOLO/) | Early standalone prototypes of DAttention, ESPPF and CompositeDetect with a shape test (`test_cehyolo.py`). The integrated versions used for training are under `ultralytics/ultralytics/nn/modules/ceh/`. |

## Dataset

The UTDAC2020 dataset is publicly available and is not included here; its source is cited in the thesis. Download: https://aistudio.baidu.com/datasetdetail/215376

## Reproducing

Place the images and COCO annotations under `datasets/coco/`, run `scripts/coco_to_yolo_detect.py` to generate labels, then train, for example:

```bash
pip install -e ultralytics
yolo detect train model=ultralytics/ultralytics/cfg/models/v8/yolov8-ceh-hda.yaml data=datasets/coco/data.yaml
```
