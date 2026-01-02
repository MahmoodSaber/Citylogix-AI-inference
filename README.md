# Citylogix-AI-inference

Run segmentation inference on road/pavement images using multiple AI model architectures.

## Purpose

This repository handles all inference operations for the Citylogix AI pipeline:
- Load trained models (OneFormer, Mask2Former, SegFormer, etc.)
- Process images with flexible naming conventions and folder structures
- Generate predictions in multiple output formats (COCO, CVAT, FiftyOne)
- Support batch processing and sliding window inference for large images

---

## Current Status: `IN DEVELOPMENT`

---

## TODO List

### High Priority

- [ ] **Multi-Model Support**
  - [ ] Create base model adapter class with unified interface
  - [ ] Implement OneFormer adapter (port from existing code)
  - [ ] Implement Mask2Former adapter
  - [ ] Implement SegFormer adapter
  - [ ] Create model registry for easy model switching
  - [ ] Support loading models from local path or HuggingFace

- [ ] **Flexible Image Input**
  - [ ] Support any image naming convention (not just `*.jpg`)
  - [ ] Support nested folder structures (`**/*.jpg`, `**/*.png`)
  - [ ] Support case-insensitive extensions (`.JPG`, `.jpeg`, `.PNG`)
  - [ ] Auto-detect image format and validate before processing
  - [ ] Handle images of any resolution (not hardcoded sizes)

- [ ] **Remove Hardcoded Values**
  - [ ] Make `crop_top=1360` configurable (currently hardcoded for dashcam)
  - [ ] Make processor size configurable (currently hardcoded 800)
  - [ ] Make class names configurable via config file (not in code)
  - [ ] Remove hardcoded paths (`/usr/src/segmentation/models/`, `/mnt/data/`)

- [ ] **Configuration System**
  - [ ] YAML-based configuration for all parameters
  - [ ] CLI argument overrides for config values
  - [ ] Environment variable support for paths
  - [ ] Validation of config before running

### Medium Priority

- [ ] **Inference Modes**
  - [ ] Full image inference (current macro models)
  - [ ] Sliding window inference with configurable crop size
  - [ ] Configurable window overlap/stride
  - [ ] Voting mechanism for overlapping predictions

- [ ] **Output Formats**
  - [ ] COCO JSON format (segmentation masks)
  - [ ] CVAT XML format (for annotation tools)
  - [ ] FiftyOne dataset export
  - [ ] Simple visualization (overlay masks on images)
  - [ ] GeoJSON (if georeferenced images)

- [ ] **Performance**
  - [ ] Batch processing with configurable batch size
  - [ ] GPU memory optimization
  - [ ] Progress logging and ETA estimation
  - [ ] Resume interrupted inference runs

### Low Priority

- [ ] **Advanced Features**
  - [ ] Ensemble inference (combine multiple models)
  - [ ] Confidence thresholding
  - [ ] Post-processing (morphological operations, small object removal)
  - [ ] ONNX runtime support for faster inference

---

## Project Structure

```
Citylogix-AI-inference/
├── README.md
├── pyproject.toml
├── config/
│   ├── default.yaml          # Default configuration
│   ├── models/               # Model-specific configs
│   │   ├── oneformer.yaml
│   │   ├── mask2former.yaml
│   │   └── segformer.yaml
│   └── classes.yaml          # Class definitions
│
├── src/citylogix_ai_inference/
│   ├── __init__.py
│   ├── predictor.py          # Main inference interface
│   ├── config.py             # Configuration loading
│   │
│   ├── models/               # Model adapters
│   │   ├── __init__.py
│   │   ├── base.py           # Abstract base class
│   │   ├── oneformer.py      # OneFormer adapter
│   │   ├── mask2former.py    # Mask2Former adapter
│   │   ├── segformer.py      # SegFormer adapter
│   │   └── registry.py       # Model registry
│   │
│   ├── processors/           # Image processing
│   │   ├── __init__.py
│   │   ├── image_loader.py   # Flexible image loading
│   │   ├── sliding_window.py # Sliding window logic
│   │   └── post_process.py   # Post-processing
│   │
│   └── exporters/            # Output formats
│       ├── __init__.py
│       ├── coco.py           # COCO JSON export
│       ├── cvat.py           # CVAT XML export
│       └── visualization.py  # Image overlays
│
├── scripts/
│   ├── infer.py              # CLI entry point
│   └── batch_infer.py        # Batch processing
│
└── tests/
    ├── test_models.py
    ├── test_image_loading.py
    └── test_exporters.py
```

---

## Usage (Planned)

```bash
# Basic inference
python -m citylogix_ai_inference.infer \
  --model oneformer \
  --checkpoint /path/to/model.pt \
  --input /path/to/images \
  --output /path/to/results

# With custom config
python -m citylogix_ai_inference.infer \
  --config config/custom.yaml \
  --input /path/to/images

# Sliding window inference for large images
python -m citylogix_ai_inference.infer \
  --model oneformer \
  --checkpoint /path/to/model.pt \
  --input /path/to/images \
  --crop-size 800 \
  --crop-overlap 100
```

---

## Dependencies

- Python >= 3.11
- PyTorch >= 2.0
- transformers (HuggingFace)
- Pillow
- OpenCV
- pycocotools
- typer (CLI)
- pydantic (config validation)

---

## Related Repositories

- [Citylogix-AI-training-prep](../Citylogix-AI-training-prep) - Data preparation
- [Citylogix-AI-training](../Citylogix-AI-training) - Model training
- [Citylogix-AI-evaluation](../Citylogix-AI-evaluation) - Performance evaluation
- [Citylogix-AI-orchestrator](../Citylogix-AI-orchestrator) - Pipeline orchestration

---

## Migration Notes (From Existing Code)

Files to port from `streetscan_segmentation`:
- `infer_multiple_macro_models.py` → Split into `predictor.py` + model adapters
- `model_tools.py` → `models/base.py`, `models/oneformer.py`
- `utils/tools.py` (crop functions) → `processors/sliding_window.py`

Key changes from original:
1. Remove SageMaker dependencies
2. Make all paths configurable
3. Support multiple model architectures
4. Flexible image input handling
