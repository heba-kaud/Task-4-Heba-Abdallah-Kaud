# AI Image Detection

An AI-based image analysis project that combines Optical Character Recognition (OCR) and Object Detection to analyze images.

## Project Overview

This project contains two main modules:

1. Text Extraction using Tesseract OCR
   - Reads text from images.
   - Applies image preprocessing to improve OCR results.
   - Extracts and displays the detected text.

2. Object Detection using MobileNet-SSD
   - Detects objects inside images.
   - Uses a pretrained MobileNet-SSD model.
   - Displays detected objects with bounding boxes and confidence scores.

## Technologies Used

- Python
- OpenCV
- Tesseract OCR
- MobileNet-SSD
- NumPy

## Models

The object detection module uses a pretrained MobileNet-SSD model with:

- MobileNetSSD_deploy.prototxt
- MobileNetSSD_deploy.caffemodel

The model is used for object detection without training a new dataset.

## Project Structure

`text
project_4/
│
├── image_detection_(AI)_CodeLabs_Training.py
├── MobileNetSSD_deploy.prototxt
├── MobileNetSSD_deploy.caffemodel
├── test_image.png
└── object_test.jpg
