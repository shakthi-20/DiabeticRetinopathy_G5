# Diabetic Retinopathy Classification using MobileNetV2

## Overview
This project aims to classify diabetic retinopathy severity using deep learning techniques. We utilize the MobileNetV2 architecture for efficient and accurate image classification. The dataset consists of retinal images labeled with different severity levels of diabetic retinopathy.

## Features
- **Deep Learning Model**: Uses MobileNetV2 for efficient feature extraction.
- **Transfer Learning**: Pretrained weights are fine-tuned on the dataset.
- **Data Augmentation**: Applied to improve model generalization.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, and Kappa Score.
- **Loss Function**: Sparse Categorical Crossentropy.
- **Deployment Ready**: Model can be deployed for real-world use cases.

## Dataset
The dataset used in this project is sourced from Kaggle:
[Diabetic Retinopathy Detection](https://www.kaggle.com/competitions/diabetic-retinopathy-detection)

The dataset comprises retinal images categorized into different classes based on diabetic retinopathy severity:
- No DR
- Mild DR
- Moderate DR
- Severe DR
- Proliferative DR

A `train.csv` file is used to map image IDs to their respective diagnosis labels.

Ensure the dataset is structured correctly before training.

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/diabetic-retinopathy-classification.git
cd diabetic-retinopathy-classification
pip install -r requirements.txt
```

### Dependencies
Ensure the following Python packages are installed:
```bash
pip install numpy pandas matplotlib seaborn opencv-python tensorflow keras scikit-learn kaggle
```

## Running the Code
The entire workflow is contained in a single Jupyter Notebook. Execute the following command to open the notebook:
```bash
jupyter notebook Code_Demonstration.ipynb
```

## Evaluation
The model performance is evaluated using the Kappa Score, along with accuracy, precision, recall, and F1-score. The loss function used is **Sparse Categorical Crossentropy**.

## Results
The model achieves high classification accuracy with balanced precision and recall. Performance can be further improved with hyperparameter tuning and data augmentation.

## Future Improvements
- Enhance dataset with more high-quality images.
- Experiment with other CNN architectures.
- Implement model quantization for mobile deployment.

## Contributing
Feel free to contribute by opening an issue or submitting a pull request.

## License
This project is licensed under the MIT License.

## Acknowledgments
Thanks to the open-source community and datasets that made this research possible.

## Contributors
Batch No – 05
AMAL KRISHNA S CB.EN.U4EEE22003
SHAKTHI S CB.EN.U4EEE22044
ASHNI S CB.EN.U4EEE22060
PRIYASHREE P CB.EN.U4EEE22125

