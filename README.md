# 🧠 Anomaly Detection with Autoencoders

**Author**: Harsh Diwaker  
**Roll Number**: 210108015

---

## 📌 Overview

This project implements an **unsupervised anomaly detection system** using autoencoders—a special type of neural network that learns to reconstruct normal patterns in data. By training the autoencoder on normal data, the system identifies anomalies as those inputs which are reconstructed poorly, leading to a high reconstruction error.

The notebook simulates a real-world fraud detection task on tabular retail transaction data, highlighting the practical power of autoencoders in detecting hidden patterns without labeled anomalies.

---

## 🧩 Background & Relevance

Autoencoders are rooted in unsupervised learning and dimensionality reduction, evolving from shallow models like PCA to deep architectures capable of learning rich representations. Their encoder-decoder structure allows them to filter "normal" inputs and flag deviations—a property that's foundational in both **anomaly detection** and **multimodal learning systems**.

---

## 🚧 Project Pipeline

The project follows a modular and pedagogical structure:

### 1. **Setup and Dataset Preparation**
- Libraries: `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`, `TensorFlow`, `Scikit-learn`
- Dataset is loaded, inspected, and preprocessed for training
- GPU acceleration (T4) is leveraged if available

### 2. **Data Preprocessing**
- Features are normalized using Min-Max scaling
- Dataset is split into training and test sets, ensuring only *normal* data in the training phase

### 3. **Autoencoder Architecture**
- Consists of:
  - **Encoder**: Compresses input to a latent representation
  - **Decoder**: Reconstructs the original input from this representation
- Activation functions like `ReLU` are used
- Loss Function: **Mean Squared Error (MSE)** is employed to capture reconstruction fidelity

### 4. **Model Training**
- Autoencoder is trained only on non-anomalous (normal) samples
- Training and validation losses are plotted for monitoring convergence

### 5. **Evaluation**
- Reconstruction error is computed for all data points
- Threshold is selected based on the distribution of errors
- Data points with error above threshold are flagged as anomalies
- Performance is evaluated using:
  - Confusion matrix
  - Precision, Recall, F1-score
  - ROC-AUC score

### 6. **Conclusion**
- Autoencoders are effective in filtering out normal patterns
- High reconstruction error helps in detecting unknown, unstructured anomalies
- Observations and learnings from results are shared in detail

---

## 📊 Results Summary

- The model demonstrated high precision in detecting anomalies with minimal false positives.
- ROC-AUC score was strong, indicating reliable separation between normal and anomalous points.
- Visualizations clearly showed the threshold line and anomaly clustering.

---

## 📌 Key Takeaways

- Autoencoders are a scalable, domain-agnostic approach to anomaly detection.
- Ideal for unsupervised environments with scarce labeled anomaly data.
- Architecture and threshold selection play a critical role in balancing sensitivity and specificity.

---

## 🚀 Future Extensions

- Implement Variational Autoencoders (VAEs) or Sequence Autoencoders for temporal anomalies
- Explore adaptive thresholding using statistical methods
- Deploy the model in real-time anomaly detection pipelines (e.g., network intrusion, transaction fraud)

---

## 📁 Files

- `anomaly-detection-with-autoencoders-6.ipynb`: Full implementation notebook
- `card_transdata.csv`: Dataset used
- `README.md`: Project documentation

---

