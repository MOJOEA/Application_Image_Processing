# Week 1 Summary — Smart Traffic Monitoring (Dataset + Model)

## Dataset
- Source: roboflow:pram/vehicle-counting-flusm/v5
- Classes (4): bus, car, motorcycle, truck
- Total images: 2613 | Total objects: 27649
- Object size: medium 52.5%, small (<32²px) 32.2%, large (>96²px) 15.3%
- Dark images (brightness < 80): 1.4% of train sample

| split | images | background_images | objects | % images | objects/image |
|---|---|---|---|---|---|
| train | 1830 | 0 | 19445 | 70.0 | 10.63 |
| val | 523 | 0 | 5414 | 20.0 | 10.35 |
| test | 260 | 0 | 2790 | 10.0 | 10.73 |

### Objects per class
| cls | train | val | test | total |
|---|---|---|---|---|
| bus | 1090 | 306 | 147 | 1543 |
| car | 6938 | 2017 | 1006 | 9961 |
| motorcycle | 8985 | 2414 | 1264 | 12663 |
| truck | 2432 | 677 | 373 | 3482 |

## Model
- yolo11n.pt fine-tuned from COCO-pretrained weights, imgsz=640, epochs=50 (early stop patience 15), batch=16
- Best epoch: 50

## Evaluation (test set)
| mAP50 | mAP50-95 | Precision | Recall | F1 | Inference ms/img | Best conf (max F1) |
|---|---|---|---|---|---|---|
| 0.9048 | 0.6756 | 0.8578 | 0.8361 | 0.8468 | 239.258 | 0.37 |

### Per class
| cls | test_instances | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| bus | 147 | 0.895 | 0.815 | 0.918 | 0.751 |
| car | 1006 | 0.882 | 0.865 | 0.921 | 0.691 |
| motorcycle | 1264 | 0.845 | 0.822 | 0.887 | 0.561 |
| truck | 373 | 0.809 | 0.842 | 0.893 | 0.699 |

## Image processing robustness test (test set, mAP50)
| condition | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|
| Original | 0.905 | 0.6753 | 0.8708 | 0.8251 |
| Night | 0.7913 | 0.5463 | 0.8116 | 0.6961 |
| Night + Gamma/CLAHE | 0.6234 | 0.4138 | 0.6741 | 0.5751 |
| Noise (salt&pepper) | 0.6109 | 0.3886 | 0.6566 | 0.5897 |
| Noise + Median filter | 0.8881 | 0.658 | 0.8677 | 0.8085 |

- Gamma + CLAHE on night images: -0.168 mAP50
- Median filter on noisy images: +0.277 mAP50

## Files
best.pt, model_info.json, src/preprocessing.py, results/*.png, results/*.csv
