# Comparative Analysis of Object Detection Architectures for Driver Phone Usage

This project evaluates the performance of legacy (SSD MobileNet V2) and modern (YOLOv8) object detection architectures for real-time driver safety monitoring.

## 1. PREPARE THE ENVIRONMENT
This project use Anaconda as Enviroment

Use Python 3.9 or higher for the modern implementation. Install the required libraries using the command below.

pip install ultralytics streamlit opencv-python roboflow

## 2. DATASET ACCESS
The system uses 4,700 frames of labeled data sourced from Kaggle and custom smartphone recordings.
1. Open Instruction.ipynb to view each step of Process.
2. Execute the Roboflow download cell to pull the dataset.

## 3. MODEL TRAINING
Run the training cells in the Instruction.ipynb notebook.
1. The YOLOv8 Nano model would achieves 86.1% mAP.
2. The training process saves the final weights as best.pt.

## 4. QUANTITATIVE LOG ANALYSIS
The system generates raw CSV logs to prove detection stability.
1. Review the scatter plots in the (graph copy.ipynb) notebook.
2. YOLOv8 maintains confidence scores above 70% throughout the 4,700 frames.

## 5. LAUNCH THE SAFETY INTERFACE
Use the Streamlit web application to test the model with video files.

streamlit run app.py

---

## !!! THE DINOSAUR TRAP WARNING (LEGACY SSD)
The SSD MobileNet V2 implementation is included for comparison but is NOT recommended for production.
1. Requires Python 3.8 or lower.
2. Requires legacy TensorFlow 2.10.0.
3. This model displayed significant instability and frequent detection misses.
