# Citylogix-AI-inference

Run segmentation inference on road/pavement images using multiple AI model architectures.

## Purpose

This repository handles all inference operations for the Citylogix AI pipeline:
- Load trained models (OneFormer - PyTorch & ONNX)
- Process images with flexible naming conventions and folder structures
- Generate predictions in multiple output formats (COCO JSON, CVAT XML, FiftyOne)
- Support batch processing and sliding window inference for large images
- Run multiple specialized models together for comprehensive defect detection

---

## Current Status: `IN DEVELOPMENT`

---

## Requirements Summary

### Input Structure

```
project_folder/
├── session_175731/
│   ├── 10070-28530_726frames/
│   │   └── *.jpg (any naming: image1.jpg, 2025-10-12.jpg, image_240.jpg, etc.)
│   └── 29720-53960_453frames/
│       └── *.jpg
└── session_198456/
    └── task_folder/
        └── *.jpg
```

### Image Requirements

| Requirement | Value |
|-------------|-------|
| Minimum size | **1024 x 2024 pixels** |
| Supported formats | `.jpg`, `.jpeg`, `.png`, `.JPG`, `.JPEG`, `.PNG` |
| Naming convention | Any (no restrictions) |

### Output

- **COCO JSON** - Segmentation masks with RLE encoding
- **CVAT XML** - For annotation tool compatibility
- **FiftyOne Dataset** - Optional (configurable)
- Output structure mirrors input folder structure

---

## Architecture

### Model Registry Pattern

All models defined in YAML config - easy to add 7th, 8th model:

```yaml
defaults:
  processor_size: 800
  crop_top: 1360
  crop_size: 400
  window_slide_h: -1      # -1 = same as crop_size
  window_slide_v: -1
  batch_size: 5
  min_image_width: 1024
  min_image_height: 2024

image_patterns:
  - "*.jpg"
  - "*.jpeg"
  - "*.png"
  - "*.JPG"

normalization:
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

output:
  coco_json: true
  cvat_xml: true
  fiftyone: false         # Optional

error_handling:
  on_image_error: skip    # "skip" or "stop"

models:
  # Macro models (full image inference)
  - name: alligator
    path: models/alligator_model.pt
    classes: [Alligator cracking]
    mode: macro
    separate: true        # Connected components separation

  - name: block
    path: models/block_model.pt
    classes: [Block cracking]
    mode: macro
    separate: true

  - name: patches
    path: models/patches_model.pt
    classes: [Patches]
    mode: macro
    separate: true

  - name: potholes
    path: models/potholes_model.pt
    classes: [Potholes]
    mode: macro
    separate: true

  # Sliding window models (cropped inference with voting)
  - name: cracks
    path: models/cracks_model.pt
    classes: [Cracks]
    mode: sliding_window

  - name: cracks_sealed
    path: models/cracks_sealed_model.pt
    classes: [Cracks sealed]
    mode: sliding_window
```

### Processing Modes

| Mode | Use Case | Description |
|------|----------|-------------|
| `macro` | Alligator, Block, Patches, Potholes | Full image inference |
| `sliding_window` | Cracks, Cracks sealed | Crop top portion, sliding window with voting |

### Configurable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `processor_size` | 800 | Model input size |
| `crop_top` | 1360 | Pixels to crop from top (dashcam removal) |
| `crop_size` | 400 | Sliding window crop size |
| `window_slide_h` | -1 | Horizontal slide (-1 = crop_size) |
| `window_slide_v` | -1 | Vertical slide (-1 = crop_size) |
| `batch_size` | 5 | Batch size for inference |
| `on_image_error` | skip | Error handling: "skip" or "stop" |

---

## Goals

| Goal | Description |
|------|-------------|
| **Faster** | Load image once, optimize batch processing, reduce redundant operations |
| **Robust** | Validate early, clear error messages, configurable error handling, comprehensive logging |
| **Preserve functions** | Port existing logic from streetscan_segmentation, don't reinvent |
| **Extensible** | Easy to add new models via config (7th, 8th, etc.) |
| **Simple** | No over-engineering, no edge case bloat |

---

## TODO List

### Phase 1: Core Infrastructure

- [ ] **Configuration System**
  - [ ] Pydantic models for config validation
  - [ ] YAML config loader
  - [ ] CLI argument overrides
  - [ ] Early validation with clear error messages

- [ ] **Folder Scanner**
  - [ ] Recursive traversal (Project → Session → Task → Images)
  - [ ] Support configurable image patterns
  - [ ] Validate minimum image size (1024x2024)
  - [ ] Clear error messages for invalid inputs

### Phase 2: Model Infrastructure

- [ ] **Model Adapter**
  - [ ] Port `PyTorchModel` from `model_tools.py`
  - [ ] Port `ONNXModel` from `model_tools.py`
  - [ ] Model registry for loading from config
  - [ ] Support both macro and sliding_window modes

- [ ] **Image Processing**
  - [ ] Port `crop_image_windows` from `utils/tools.py`
  - [ ] Port `assemble_crops_with_voting` from `utils/tools.py`
  - [ ] Port `batch_array` from `utils/tools.py`
  - [ ] Make crop_top configurable (not hardcoded 1360)

### Phase 3: Inference Pipeline

- [ ] **Predictor (Orchestrator)**
  - [ ] Load all models from config
  - [ ] Process images through all models
  - [ ] Optimize: load image once, share across models
  - [ ] Progress bars (tqdm) + logging
  - [ ] Configurable error handling (skip/stop)

- [ ] **Output Exporters**
  - [ ] COCO JSON with RLE encoding
  - [ ] CVAT XML conversion
  - [ ] FiftyOne dataset (optional)
  - [ ] Mirror input folder structure in output

### Phase 4: CLI & Integration

- [ ] **CLI Entry Point**
  - [ ] `citylogix-infer` command
  - [ ] `--project`, `--output`, `--config` arguments
  - [ ] Progress display and summary

- [ ] **Testing**
  - [ ] Config validation tests
  - [ ] Image loading tests
  - [ ] Model adapter tests
  - [ ] End-to-end inference test

---

## Project Structure

```
Citylogix-AI-inference/
├── README.md
├── pyproject.toml
├── config/
│   └── default.yaml              # Default configuration
│
├── models/                       # Model weights (.pt, .onnx files)
│   └── *.pt                      # Place your model files here
│
├── src/citylogix_ai_inference/
│   ├── __init__.py
│   ├── cli.py                    # Entry point (typer)
│   ├── config.py                 # Pydantic config models
│   ├── predictor.py              # Main orchestrator
│   │
│   ├── adapters/                 # Model adapters (loading & inference)
│   │   ├── __init__.py
│   │   ├── base.py               # ModelAdapter base class
│   │   ├── pytorch.py            # PyTorchModel (from model_tools.py)
│   │   ├── onnx.py               # ONNXModel (from model_tools.py)
│   │   └── registry.py           # Load models from config
│   │
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── folder_scanner.py     # Recursive folder traversal
│   │   ├── image_loader.py       # Load + validate images
│   │   └── sliding_window.py     # crop_image_windows, voting
│   │
│   └── exporters/
│       ├── __init__.py
│       ├── coco.py               # COCO JSON output
│       └── cvat.py               # CVAT XML output
│
├── scripts/
│   └── infer.py                  # CLI script
│
└── tests/
    ├── test_config.py
    ├── test_folder_scanner.py
    ├── test_models.py
    └── test_exporters.py
```

---

## Usage (Planned)

```bash
# Basic usage with config file
citylogix-infer \
  --project /path/to/project_folder \
  --output /path/to/output \
  --config config/default.yaml

# Short form
citylogix-infer -p /path/to/project -o /path/to/output -c config.yaml

# With CLI overrides
citylogix-infer \
  --project /path/to/project \
  --output /path/to/output \
  --config config.yaml \
  --batch-size 10 \
  --on-error skip
```

---

## Dependencies

- Python >= 3.11
- PyTorch >= 2.0
- transformers >= 4.44 (HuggingFace)
- Pillow >= 10.0
- OpenCV >= 4.8
- pycocotools >= 2.0
- typer >= 0.12 (CLI)
- pydantic >= 2.0 (config validation)
- loguru >= 0.7 (logging)
- tqdm >= 4.66 (progress bars)
- numpy >= 1.24
- onnxruntime (optional, for ONNX models)

---

## Migration Notes (From Existing Code)

### Files to Port

| Source | Destination | Description |
|--------|-------------|-------------|
| `model_tools.py` → `PyTorchModel` | `adapters/pytorch.py` | PyTorch model loading & inference |
| `model_tools.py` → `ONNXModel` | `adapters/onnx.py` | ONNX model loading & inference |
| `utils/tools.py` → `crop_image_windows` | `processors/sliding_window.py` | Sliding window cropping |
| `utils/tools.py` → `assemble_crops_with_voting` | `processors/sliding_window.py` | Voting mechanism |
| `utils/tools.py` → `batch_array` | `processors/sliding_window.py` | Batch array utility |
| `infer_multiple_macro_models.py` | `predictor.py` | Multi-model orchestration |
| COCO/CVAT export logic | `exporters/` | Output format handlers |

### Key Changes from Original

1. **Remove hardcoded values** - All in config (crop_top, processor_size, etc.)
2. **Remove SageMaker dependencies** - Local execution only
3. **Model registry** - YAML-based, easy to extend
4. **Flexible input** - Any image naming, recursive folder structure
5. **Configurable error handling** - Skip or stop on error
6. **Progress tracking** - tqdm + logging

---

## Related Repositories

- [Citylogix-AI-training-prep](https://github.com/MahmoodSaber/Citylogix-AI-training-prep) - Data preparation
- [Citylogix-AI-training](https://github.com/MahmoodSaber/Citylogix-AI-training) - Model training
- [Citylogix-AI-evaluation](https://github.com/MahmoodSaber/Citylogix-AI-evaluation) - Performance evaluation
- [Citylogix-AI-orchestrator](https://github.com/MahmoodSaber/Citylogix-AI-orchestrator) - Pipeline orchestration
