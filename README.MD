# Better Hey - Part 1 (Backend) - Current Build

## Overview
This repository is **Part 1 (backend)** of the **Better Hey** project.

It focuses on:
- data collection and preprocessing from sign-language videos,
- feature extraction and motion/gesture modeling,
- training and exporting a `.pt` model file used by **Part 2 (mobile app)** for real-time comparison and inference.

The project is **non-profit**, **offline-first**, and designed for privacy and accessibility.

## Why This Project Exists
Better Hey aims to improve communication between:
- people using sign language,
- and people speaking directly.

Key motivation:
- reliable communication in environments with limited/no internet,
- private on-device operation without cloud media storage.

## Scope of This Repository
This backend project has 2 internal tracks:

1. **Windows recognition & analysis system (UI prototype)**
- real-time camera ingestion,
- gesture/character detection visualization,
- analysis UI for tuning thresholds and behavior,
- foundation for future Windows software packaging.

2. **Video extraction + machine learning system (`tools/ml_tuner.py`)**
- collect/labeled-video workflow,
- feature extraction pipeline,
- model training and validation,
- export `models/gesture_model_best.pt` for mobile usage.

## Relation to Full Better Hey Program
- **Windows side**: create data and build core recognition logic.
- **Mobile side (Part 2)**: real-time two-way conversion:
1. camera gesture -> voice/text output,
2. incoming voice -> text + gesture simulation/visual output.

## Core Principles
- Fully offline inference/training workflows where possible.
- No ad/tracking model.
- Privacy-oriented: no location permission, no behavioral tracking.
- Lightweight target: model package under ~200MB for mid-range phones.

## Recommended Dataset Layout
Use label-based folders:

```text
textDataset/
├── again/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── what/
│   ├── video1.mp4
│   └── ...
├── where/
└── ...
```

`tools/ml_tuner.py` supports assigning labels per video and storing clips under `textDataset/<label>/`.

## Data Pipeline (Video -> ML Model)
1. **Input video**
- local clips or downloaded clips (tool-assisted).

2. **Landmark extraction**
- pose + left/right hand landmarks per frame.

3. **Feature encoding**
- per-frame vector encoding (`core/ml_features.py`),
- temporal sequence building with padding/trimming.

4. **Dataset build**
- `X` (sequence tensors), `y` (class ids), labels JSON mapping.

5. **Model training**
- Temporal Convolutional Network (`core/ml_model.py`),
- augmentation + class balancing in training utilities.

6. **Export**
- `models/gesture_model_best.pt`
- consumed later by mobile/frontend integration.

## About “MML” Conversion
In this project, video is converted into a structured machine-readable motion representation:
- frame-level landmarks,
- normalized kinematic features,
- temporal sequence tensors.

This intermediate representation is then used by the ML stack to train/export `.pt` models.
If your team uses the term **MML** internally, this corresponds to the structured motion-feature stage before model inference.

## Current Limitations
- recognition can degrade on specific edge cases:
  - hands briefly occluded near chest,
  - left/right hand confusion in hard poses,
  - sparse data for certain actions.
- limited dataset scale for robust generalization.
- single-developer resource constraints slow iteration speed.

## Roadmap
1. **Part 1 - Backend**: recognition + ML training pipeline.
2. **Part 2 - Frontend/Mobile**: integrate `.pt` model with voice->text and gesture UX.
3. **Part 3 - Testing & Balancing**: quality tuning, false-positive/false-negative balancing.
4. **Part 4 - Official Website + Community Data Contribution**.
5. **Part 5 - Public Release**.

## Highlights for Recruiters
This project demonstrates:
- end-to-end ownership (data -> model -> runtime integration path),
- practical ML engineering under resource constraints,
- iterative improvement mindset (UI tooling, dataset workflow, runtime debugging),
- product thinking with accessibility + privacy + offline-first constraints,
- ability to turn a single-developer prototype into a staged roadmap.

