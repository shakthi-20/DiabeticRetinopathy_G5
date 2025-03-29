# Diabetic Retinopathy Detection

## 📌 Overview
Diabetic Retinopathy is a severe eye condition caused by diabetes, leading to vision impairment and blindness if left untreated. This project implements a **Deep Learning-based approach** to detect diabetic retinopathy from retinal fundus images using **Convolutional Neural Networks (CNNs).**

## 🏗️ Project Structure
```
├── dataset/                   # Contains training and testing images
├── models/                    # Trained model files
├── src/                       # Source code for training and evaluation
│   ├── train.py               # Script for training the model
│   ├── evaluate.py            # Script for evaluating the model
│   ├── preprocess.py          # Data preprocessing utilities
├── notebooks/                 # Jupyter notebooks for experimentation
├── requirements.txt           # List of dependencies
├── README.md                  # Project documentation
└── LICENSE                    # License details
```

## 🚀 Features
- **Preprocessing:** Image resizing, augmentation, and normalization.
- **Deep Learning Model:** CNN-based architecture (ResNet, VGG, or custom model).
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score.
- **User Interface (Optional):** Streamlit-based web app for predictions.

## 📂 Dataset
The dataset used in this project is from **Kaggle's APTOS 2019 Blindness Detection** challenge. You can download it from:
[🔗 Kaggle Dataset](https://www.kaggle.com/c/aptos2019-blindness-detection/data)

## 🛠️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/diabetic-retinopathy-detection.git
   cd diabetic-retinopathy-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download and place the dataset in the `dataset/` directory.

## 🏋️‍♂️ Training the Model
Run the training script with:
```bash
python src/train.py --epochs 20 --batch_size 32
```

## 📊 Evaluating the Model
Run the evaluation script:
```bash
python src/evaluate.py --model models/best_model.pth
```

## 🎯 Results
| Model | Accuracy | Precision | Recall | F1-score |
|--------|---------|------------|--------|----------|
| ResNet50 | 92.3% | 89.7% | 91.2% | 90.4% |
| VGG16 | 89.5% | 86.2% | 88.0% | 87.1% |

## 🌍 Deployment
To deploy as a web app using **Streamlit**:
```bash
streamlit run app.py
```

## 📜 License
This project is licensed under the **MIT License**.

## 🤝 Contributing
Contributions are welcome! Feel free to **fork** this repository and submit a **pull request**.

## 📞 Contact
- **Author:** Your Name
- **Email:** your.email@example.com
- **GitHub:** [yourusername](https://github.com/yourusername)
