# Cricket Ball Detection and Tracking  
**EdgeFleet.AI – Round 1 AI/ML Assessment**

This repository contains the complete implementation for the **cricket ball detection and tracking pipeline** developed as part of **Round 1 of the EdgeFleet.AI recruitment process**. The solution combines multimodal grounding-based localization with video-level segmentation and tracking to robustly track a cricket ball from a single fixed-camera broadcast video.

---

## Repository Structure

```bash
submission/
├── code/
│ ├── molmo/
│ │ ├── config.yaml
│ │ └── molmo_infer_uniform_sampling.py
│ │
│ └── sam3/
│ └── edgefleet/
│ ├── config.yaml
│ ├── process_video.py
│ ├── sam3_direct_prompitng_captions_to_videos.py
│ └── utils.py
│
├── results/
│ └── collage.mp4
│
├── processing.log
└── README.md
```

---

## Overview

The pipeline follows a **multi-stage, multi-model design**:

1. **Video preprocessing** (cropping to remove broadcast overlays)
2. **Initial ball localization** using a multimodal grounding model (MOLMO)
3. **Video-level segmentation and tracking** using SAM3
4. **Cross-model refinement**, where MOLMO point predictions are injected into SAM3 to improve tracking robustness
5. **Visualization and aggregation** of results into a single collage video

This design improves robustness against motion blur, occlusion, and broadcast artifacts while maintaining reproducibility and modularity.

---

## Requirements

- Python 3.10+
- PyTorch
- OpenCV
- NumPy
- CUDA-enabled GPU (recommended)

All experiments were conducted assuming a Linux environment. Please refer [requirements.txt](requirements.txt) for the complete list of dependencies.

---

## Running MOLMO (Grounding-Based Localization)

MOLMO is used to obtain **point-based ball localization** using a natural language prompt.

### Configuration
Edit parameters in: `code/molmo/config.yaml`.


### Run MOLMO inference
```bash
cd code/molmo
python molmo_infer_uniform_sampling.py --config config.yaml
```
This step produces point-based predictions of the cricket ball across uniformly sampled frames, which are later used to refine SAM3 tracking.

## Running SAM3 (Segmentation and Tracking)

SAM3 is responsible for video-level segmentation and propagation of the cricket ball.

### Configuration

Edit the parameters in: `code/sam3/edgefleet/config.yaml`.


### Main Script to Run

```bash
cd code/sam3/edgefleet
python sam3_direct_prompitng_captions_to_videos.py --config config.yaml
```
Script Functionality

This script performs the following steps:

Prompts SAM3 with the object label “ball”

Selects the most confident object track across frames

Injects MOLMO point predictions to refine the selected track

Generates trajectory overlays and per-video visualizations

## Results

[results/collage.mp4](results/collage.mp4) contains the final collage video showing the tracked cricket ball across multiple test videos.

The final outputs from multiple runs are aggregated into a single collage video for easy qualitative inspection.
