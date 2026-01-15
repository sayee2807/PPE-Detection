# PPE Detection System using YOLOv8

A real-time computer vision system that detects whether construction workers are wearing required
Personal Protective Equipment (PPE) such as Hardhat, Safety Vest, and Mask. The system highlights
compliance with green bounding boxes and violations with red bounding boxes.

--------------------------------------------------

## FEATURES
- Detects PPE classes:
  Hardhat / NO-Hardhat
  Safety Vest / NO-Safety Vest
  Mask / NO-Mask
- Green box indicates PPE present
- Red box indicates PPE missing
- Works on video files and webcam
- Custom-trained YOLOv8 model

--------------------------------------------------

## TECHNOLOGIES USED
- Python 3.10
- YOLOv8 (Ultralytics)
- OpenCV
- CVZone
- PyTorch

--------------------------------------------------

## PROJECT STRUCTURE

The Videos directory contains sample construction site videos used for testing and demonstration of the PPE detection system.

The Model folder stores the custom-trained YOLOv8 model file (best.pt) used for PPE compliance inference.

The requirements.txt file lists all Python dependencies required to set up and run the project environment.

The run.py script serves as the main entry point for executing the real-time PPE compliance detection pipeline.

--------------------------------------------------

HOW TO RUN

Run on video file:
   python main.py

--------------------------------------------------

PERFORMANCE

Accuracy: 63% mAP@0.5
Inference Speed: ~17–20 FPS on CPU
