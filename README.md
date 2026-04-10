# 🎭 The Bard Bot: Shakespearean Text Generation

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

This project leverages **Recurrent Neural Networks (RNNs)** to generate text in the unmistakable literary style of William Shakespeare. By training on over 100,000 lines of dialogue, the model captures the essence of 16th-century prose through word-level sequence prediction.

---

## 📌 Project Overview
The core objective is to simulate the creative process of "the Bard." Using **TensorFlow** and **Keras**, the model processes raw theatrical dialogue and employs a hidden-state memory (SimpleRNN) to learn the sequential dependencies and rhythmic patterns unique to Shakespearean English.

---

## 📊 Dataset Insights
The model is powered by the **Shakespeare Plays** dataset, a comprehensive collection of his theatrical works.

| Feature | Details |
| :--- | :--- |
| **Source** | `kingburrito666/shakespeare-plays` (Kaggle) |
| **Size** | 111,000+ lines of dialogue |
| **Key Plays** | *Henry IV*, *Hamlet*, *Macbeth*, and more |
| **Target Feature** | `PlayerLine` (The spoken dialogue) |

---

## ⚙️ The Methodology
The pipeline follows a rigorous NLP workflow designed for high-fidelity sequence modeling.

### 1. Data Preprocessing
* **Text Sanitization**: Conversion to lowercase and removal of special characters via Regex.
* **Tokenization**: Mapping unique vocabulary to specific integer IDs using the `Tokenizer` class.

### 2. Sequence Engineering
We utilize a **Sliding Window** approach to create our training samples:
* **Input Sequence**: 5 consecutive words ($X$).
* **Target Label**: The subsequent 6th word ($y$).
* **Encoding**: Target labels are transformed via **One-Hot Encoding** for categorical compatibility.

### 3. Model Architecture
> **The Neural Engine**
> * **Embedding Layer**: Maps indices to a 100-dimensional dense vector space.
> * **SimpleRNN Layer**: 150 hidden units to process temporal information.
> * **Dense Output**: Softmax activation across the entire `vocab_size`.

---

## 📈 Results & Observations
The model is optimized using **Adam** and **Categorical Cross-Entropy**. 

* **Capabilities**: Successfully captures short-term dependencies and common word pairings (e.g., "thou art").
* **Optimization**: While effective, the SimpleRNN structure provides a foundation that can be further enhanced.

> [!TIP]
> **Future Roadmap**: To generate longer, more coherent passages, consider evolving this architecture into an **LSTM** or **GRU** to mitigate the vanishing gradient problem.

---

## 🛠️ Tech Stack
* **Language**: ![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54)
* **ML Framework**: ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)
* **Data Science**: ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
* **Environment**: Google Colab / T4 GPU

---

## ▶️ Getting Started

### 1. Setup
```bash
# Clone the repository
git clone <your-repo-link>
cd <repo-name>

# Install required dependencies
pip install tensorflow pandas numpy kagglehub
