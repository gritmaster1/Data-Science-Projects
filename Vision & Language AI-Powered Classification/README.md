# Vision & Language: AI-Powered Classification

## Project Overview
This project explores the fusion of computer vision and natural language processing (NLP) in modern machine learning. It leverages pre-trained foundation models to analyze images and text, applies dimensionality reduction, and evaluates multiple classification approaches.

## Datasets Used
- **CIFAR-10**: A dataset of 60,000 32x32 color images categorized into 10 classes (e.g., airplanes, cars, birds, and cats).
- **AG News**: A dataset containing news articles classified into four categories: World, Sports, Business, and Science/Technology.

## Key Tasks
### 1. Image Classification using ResNet-18
- Loaded the CIFAR-10 dataset and split it into training (50,000 images) and testing (10,000 images) sets.
- Extracted embeddings using the ResNet-18 model after removing its final classification layer.

### 2. Visualization with t-SNE
- Reduced high-dimensional image embeddings into 2D using t-SNE.
- Plotted the visualization to analyze how well similar images cluster.

### 3. Nearest Neighbor Classification
- Computed class centroids and classified test images based on their nearest centroid.
- Used cosine similarity and Euclidean distance for classification.
- Generated a confusion matrix to evaluate classification performance.
- Identified outlier images farthest from their class centroids.

### 4. Supervised Image Classification Model
- Used the extracted embeddings as features for classification.
- Trained logistic regression and random forests models.
- Compared performance against nearest-neighbor classification.

### 5. Dimensionality Reduction with PCA/SVD
- Reduced embeddings to 10 and 50 dimensions.
- Evaluated classification performance and compared it with full embeddings.

### 6. NLP Task with DistilBERT
- Loaded AG News dataset and extracted text embeddings using the DistilBERT model.
- Repeated the same classification tasks as CIFAR-10, including t-SNE visualization, nearest neighbor classification, and supervised classification.

### 7. Evaluating AI Code Generation Tools
- Used ChatGPT, CoPilot, or Google Gemini to generate solutions for this assignment.
- Analyzed the effectiveness of AI-generated code, identifying strengths and weaknesses.
- Reflected on the role of AI in coding and its impact on problem-solving skills.

## Technologies Used
- Python (pandas, numpy, scikit-learn, matplotlib, seaborn, torchvision, torch, transformers)
- Jupyter Notebook / Google Colab
- Pre-trained models: ResNet-18 (torchvision), DistilBERT (Hugging Face)
- t-SNE, PCA, SVD for dimensionality reduction
- Nearest neighbor classification (cosine similarity, Euclidean distance)

## Results & Learnings
- **Image classification**: ResNet-18 embeddings improved classification accuracy compared to raw images.
- **Dimensionality reduction**: PCA/SVD with 50 dimensions preserved classification performance while reducing computational cost.
- **NLP classification**: DistilBERT embeddings significantly enhanced classification accuracy compared to traditional text vectorization.
- **AI-generated code evaluation**: While AI tools provided useful code snippets, debugging and understanding their outputs were necessary.

## How to Run the Project
1. Clone the repository or download the Jupyter Notebook.
2. Install dependencies using Anaconda or pip:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn torchvision torch transformers
   ```
3. Download CIFAR-10 and AG News datasets using the provided scripts.
4. Open the Jupyter Notebook and run the cells sequentially.

## Future Work
- Experiment with deeper models (ResNet-50, ViTs) for better feature extraction.
- Apply advanced NLP techniques such as fine-tuning DistilBERT for text classification.
- Extend classification models with ensemble learning for improved accuracy.

## Acknowledgments
This project was completed as part of the CSE 519 - Data Science course under Prof. Steven Skiena. The datasets were obtained from Kaggle and Hugging Face.

