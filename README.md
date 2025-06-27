# PV-Fault-Classifier: Fault Detection and Classification for PV Panels based on Thermal Imaging and Machine Learning
This repository contains the official code implementation of a Final Year Project on automated fault detection and classification of photovoltaic (PV) modules using thermal infrared (TIR) imagery and deep learning techniques. The project integrates image preprocessing, deep learning model training (custom CNN and transfer learning), and deployment of a real-time graphical user interface (GUI) for field applications.

**Project Overview**
This project presents a complete pipeline for PV fault classification from thermal images:
1. Thermal Image Preprocessing: Corner detection, perspective transformation, orientation adjustment, and tight cropping.
2. Model Training:
   - Custom CNN: Designed for lightweight yet accurate inference.
   - Transfer Learning: Evaluated using ResNet50, EfficientNetB0, and MobileNetV2.
   - Optimizations include class weighting (ENS), learning rate schedulers (Cosine Annealing), and early stopping.
3. Graphical User Interface (GUI): A Tkinter-based desktop application for real-time prediction, result saving, and batch analysis.

**Repository Structure**
- preprocessing/ - Image preprocessing scripts
- models/ - Custom CNN architecture and trassfer learning model development scripts
- gui/ - Source code for the standalone desktop GUI application

**Dataset**
This project uses the publicly available PVF-10 dataset:

Wang, B., Chen, Q., Wang, M., Chen, Y., Zhang, Z., Liu, X., Gao, W., Zhang, Y., & Zhang, H. (2024). PVF-10: A high-resolution unmanned aerial vehicle thermal infrared image dataset for fine-grained photovoltaic fault classification. Applied Energy, 376, 124187. https://doi.org/10.1016/j.apenergy.2024.124187

**Deployment**
The GUI can be packaged as a standalone desktop application using PyInstaller for ease of use in field applications:

_pyinstaller --clean --onefile --windowed --icon=app_icon.ico image_classifier_app.py_

**Citation**
If you find this repository useful in your research or application, please cite this work and the PVF-10 dataset as described above.
