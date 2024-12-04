# Grape Leaf Disease Detection: CNN vs. Vision Transformers

# Project Overview

## Project Title
Grape Leaf Disease Detection: CNN vs. Vision Transformers

## Introduction
Grapevine cultivation plays a significant role in the economy, particularly in regions such as British Columbia, where the grape industry faces challenges due to diseases affecting grape leaves. These diseases can severely impact both crop yield and quality. Traditional methods, such as visual inspection, are time-consuming and prone to errors, making early disease detection difficult. Recent advancements in machine learning, specifically Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs), offer a more reliable and efficient approach for grape leaf disease detection. This research explores the use of these advanced techniques, focusing on comparing their effectiveness in diagnosing grape leaf diseases like Black Rot, ESCA, and Blight. By employing these state-of-the-art models, the project aims to improve early-stage disease detection, enhance disease management strategies, and provide a robust, data-driven solution to support the viticulture industry. Results from this study indicate that machine learning models, particularly CNNs integrated with data augmentation and transfer learning techniques, significantly outperform traditional methods, with up to 99% accuracy in disease classification, contributing to better disease prediction and management in grapevine cultivation.

## Dataset
The dataset for this study consists of 9,023 high-resolution images of grape leaves, sourced from publicly available repositories such as PlantVillage and Kaggle, along with additional images collected from vineyards. The dataset is evenly distributed across four categories: 2,340 images of Black Rot, 2,350 images of Esca, 2,183 images of Leaf Blight, and 2,150 healthy images.

### Key Features
- **Black Rot**
- **Esca**
- **Leaf Blight**
- **Healthy**

## How to Access and Run the Project
Kindly note, the file used to develop and train the models is saved as grape_cnn.ipynb and grape_vit.ipynb for CNN and Vision Transformer respectively. 

### Prerequisites
- Python. 
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `joblib`, `pymongo`, `pyspark`,  `streamlit`, `torch`, `transformers`, `torchvision`, `matplotlib`, `seaborn`

### Steps to Access the Project
1. **Clone the Repository**
   ```bash
    git clone https://github.com/Joshuaogaga/Grape-Disease.git
    cd Grape-Disease
   ```

2. **Install Required Libraries**
   ```bash
   conda env create -f genv.yaml
   conda activate  genv
   ```

3. **Run the Project**
    - For convenience sake, our project was deployed using streamlit where you can easily switch models to test its predictive ability. You can run the project by running the following command in the terminal:
    ```bash
     streamlit run app.py
     ```
## Evaluation Metrics
- **Accuracy**: Overall correct predictions
- **Precision and Recall**: To handle class imbalance
- **F1 Score**: Balance between precision and recall.