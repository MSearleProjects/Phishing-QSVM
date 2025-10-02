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
The workflow is contained within the `Phishing_QSVM_V3.ipynb` notebook and requires the data file `phishing_data.csv` to be present.

**Prerequisites**
* **Python**: Version 3.9 or higher.
* **Jupyter**: A Jupyter Notebook environment (Lab or classic).

**Code Explained**
The `Phishing_QSVM_V3` notebook executes the detection task in 4 main stages:
1. **Data Loading and Preprocessing:**
   * The `phishing_data.csv` file is loaded into a Pandas DataFrame.
   * The features and the target variable (`status`) are seperated. The categorical status column is converted to numerical labels (`0` for legitimate, `1` for phish).
   * The dataset is split into 70% training and 30% testing subsets, using stratified sampling to maintain class balance (`random_state=109`).
   * All 11 features are scaled using a `MinMaxScaler` to the range **(-1, 1)**, which is the required input range for the quantum feature map.

2. **Quantum Kernel Definition:**
   * A `ZZFeatureMap` is instantiated using 11 qubits and 2 repetitions. This map encodes the classical feature vector x into a quantum state ∣ψ(x)⟩ through a series of Hadamard, rotation (Rz​), and controlled-Z (CZ) gates.
   * A `FidelityQuantumKernel` is created using the feature map. This kernel calculates the similarity between the 2 encoded quantum states, ∣ψ(xi​)⟩ and ∣ψ(xj​)⟩, which serves as the core of the quantum classification model.

3. **Model Training (QSVC):**
   * The **QSVC** is initialised using the defined quantum kernel.
   * The model is trained by calling `qsvc.fit(X_train, y_train)`. During this stage, the simulator calculates the training kernel matrix K, where Kij​=∣⟨ψ(xi​)∣ψ(xj​)⟩∣2. This matrix is then used by the classical SVM solver to find the optimal seperating hyperplane.

4. **Prediction and Evaluation:**
   * The trained model makes predicitons on the held-out test set (`X_test`).
   * Standard metrics (Accuracy, recall, F1-Score) are computed using `scikit-learn` to quantify the model's performance.
***
## Performance Results
