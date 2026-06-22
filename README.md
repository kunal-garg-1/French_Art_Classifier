# 🎨 The AI Curator: French Art Classifier

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![Fast.ai](https://img.shields.io/badge/Fast.ai-Transfer%20Learning-FFD43B.svg)](https://www.fast.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B.svg)](https://streamlit.io/)

**Live Web Application:** [Launch The AI Curator](https://french-art-curator.streamlit.app/)

## 📌 Project Overview
The AI Curator is an end-to-end computer vision web application trained to differentiate between the highly similar brushstrokes of four French Masters: **Claude Monet, Pierre-Auguste Renoir, Edgar Degas, and Paul Cézanne.** Unlike standard tutorial projects that use pre-packaged Kaggle datasets, this project demonstrates a complete, real-world Machine Learning Engineering lifecycle: from automated web scraping and data sanitation to transfer learning and cloud deployment.

---

## ⚙️ The Engineering Pipeline

This project was built from scratch using the following pipeline:

### 1. Data Sourcing & Web Scraping
* Sourced the raw dataset using automated web scraping scripts (`bing-image-downloader`) to query search engines for specific artists.
* Overcame API rate limits and bot-blocking protocols to successfully construct a raw dataset of French Impressionist and Post-Impressionist art.

### 2. Data Cleaning & Integrity
* Programmatically verified image headers to locate and purge corrupted file downloads.
* Utilized `fast.ai`'s `ImageClassifierCleaner` GUI widget inside a Jupyter Notebook to manually review the highest-loss images, purging irrelevant search results (e.g., modern logos, movie posters, portraits) to create a pristine, highly-specialized dataset.

### 3. Model Architecture & Transfer Learning
* **Brain:** Utilized **ResNet18**, a state-of-the-art Convolutional Neural Network (CNN) pre-trained on the ImageNet dataset.
* **Pipeline:** Built a custom `DataBlock` to resize images into uniform 128x128 tensors, automatically labeling data based on directory structures, and splitting the data for validation.
* **Fine-Tuning:** Froze the core layers to retain edge/shape detection, attached a custom output layer for the 4 specific artist categories, and fine-tuned the model to achieve high accuracy on a notoriously difficult classification task (differentiating Monet from Renoir).

### 4. Production Inference (The PyTorch Bypass)
* **The Challenge:** Encountered a known serialization bug between Python 3.13 and the high-level `fast.ai` inference pipeline when handling in-memory Streamlit image objects.
* **The Solution:** Bypassed the wrapper library entirely. Extracted the raw PyTorch model weights (`model.eval()`), built a custom `torchvision.transforms` pipeline to enforce RGB conversion and normalization, and utilized pure PyTorch to calculate the `softmax` probabilities.

### 5. Deployment
* Designed a responsive, user-friendly frontend using **Streamlit**.
* Packaged the architecture and weights into a compressed `.pkl` file.
* Managed large asset tracking using **Git LFS** (Large File Storage) to bypass standard GitHub upload limits.
* Deployed the final inference engine to **Streamlit Community Cloud**.

---

## 💻 How to Run Locally

If you want to run this application on your own machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR-USERNAME/French_Art_Classifier.git](https://github.com/YOUR-USERNAME/French_Art_Classifier.git)
   cd French_Art_Classifier
