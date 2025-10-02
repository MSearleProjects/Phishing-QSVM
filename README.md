# Phishing Detection - Quantum Support Vector Classifier for URL Classification
## Overview
This repository documents the development of a **Phishing URL Detection System** using **Quantum Machine Learning (QML)**. The approach utilises the **Quantum Support Vector Classifier (QSVC)** to classify URLs as either `phish` or `legitimate`.
The model processes **11 features** from the URLs structure. Using an enhanced feature map, the QSVC aims to achieve superior results compared to classical methods and demonstrate a computational advantage in cyber threat detection.
***
## Quantum Methodology and Technology
This section details the specific QML approach and the technology stack used to execute the project.

**Technology Stack**
* **Quantum Framework**: Qiskit (`qiskit-machine-learning`, `qiskit-aer`).
* **Programming Language**: Python
* **Data Science**: Pandas, NumPy, Scikit-learn.
* **Execution Environment**: Jupyter Notebook (`Phishing_QSVM_V3.ipynb`)

**Quantum Compenents**
The classification is performed using a quantum kernel made from the following circuit structure:
* **Feature Map**: `ZZFeatureMap`
* **Input Dimension**: 11 Qubits (one per feature)
* **Circuit Complexity**: 2 Repetitions
* **Entanglement**: Linear
* **Training Backend**: Qiskit Aer Qasm Simulator
***
## Walkthrough
