### 🧬 Breast Cancer Classification using Deep Learning

This project is a web-based application built with **Streamlit** that classifies breast cancer tumors as **benign** or **malignant**. The model analyzes histopathological images to assist in early detection and diagnosis of breast cancer.

---

### 🚀 Try the App

👉 **[Launch the App](https://breastcancerclassification-fkyl7yjknvkjk4wdqcgdgg.streamlit.app/)**

---

### 🧠 About the Model

The model is a Convolutional Neural Network (CNN) trained on labeled mammography and histopathology images. It was designed to detect whether a given breast image shows characteristics of a **benign** or **malignant** tumor.

---

### 📁 Dataset

This model was trained using a combination of two public datasets:

* **[Mammography Breast Cancer Detection](https://www.kaggle.com/datasets/gauravduttakiit/mammography-breast-cancer-detection)**
  Contains mammographic images labeled with tumor types for detection tasks.

* **[Breast Cancer Detection](https://www.kaggle.com/datasets/hayder17/breast-cancer-detection)**
  Includes histopathological image data used for binary classification (benign vs. malignant).

---

### 📦 Installation & Run Locally

1. Clone the repo:

   ```bash
   git clone https://github.com/your-username/breast-cancer-classification.git
   cd breast-cancer-classification
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the app:

   ```bash
   streamlit run app.py
   ```

---

### 💡 Features

* Upload breast cancer image scans.
* Predict tumor type: **Benign** or **Malignant**.
* Simple, interactive web interface for medical insights.

---

### 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* Streamlit
* PIL, NumPy, Matplotlib
