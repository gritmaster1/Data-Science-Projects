# Zillow Housing Price Prediction

## Project Overview

This project focuses on housing price prediction using data from the Zillow Prize Challenge on Kaggle. The primary objective is to integrate various datasets, explore key features, and develop a predictive model to estimate real estate prices. The project involves multiple data science techniques, including feature engineering, clustering, and model evaluation.

## Dataset

The dataset comes from the Zillow Prize Challenge and includes a variety of property attributes such as location, tax assessments, and structural details. An additional external dataset was incorporated to enhance predictive capabilities.(https://www.kaggle.com/c/zillow-prize-1/data)

## Key Tasks

1. **House Desirability Scoring**

   - Developed a scoring function to rank houses based on desirability.
   - Identified the most and least desirable properties based on relevant attributes.

2. **Pairwise Distance Function for Properties**

   - Created a function to measure the similarity between properties.
   - Used geographical and property-specific features to refine the metric.

3. **Clustering Analysis**

   - Applied clustering algorithms (k-means, DBScan) to group similar properties.
   - Visualized clusters on a map to analyze patterns in housing attributes.

4. **Integration of External Data**

   - Incorporated external datasets related to financial, geographic, and economic factors.
   - Evaluated the impact of additional data on the model’s performance.

5. **Data Visualization**

   - Created multiple visualizations to explore trends in housing data.
   - Used matplotlib and seaborn to present insights effectively.

6. **Predictive Model Development**

   - Built machine learning models to predict logerror in Zillow estimates.
   - Experimented with different algorithms to optimize accuracy.
   - Submitted predictions to Kaggle and evaluated performance.

7. **Statistical Validation**

   - Conducted a permutation test to assess whether the model outperforms a random assignment of logerror values.

## Technologies Used

- Python (pandas, numpy, scikit-learn, matplotlib, seaborn)
- Jupyter Notebook / Google Colab
- Kaggle API
- Clustering techniques (k-means, DBScan)

## Results & Learnings

- Identified key factors influencing property prices.
- Discovered patterns through clustering and visualization.
- Improved model performance by incorporating external datasets.
- Gained insights into real estate price prediction challenges.

## How to Run the Project

1. Clone the repository or download the Jupyter Notebook.
2. Install dependencies using Anaconda or pip:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Download the dataset from Kaggle and place it in the project directory.
4. Open the Jupyter Notebook and run the cells sequentially.

## Future Work

- Explore deep learning models for better price estimation.
- Experiment with additional external datasets.
- Deploy the model as a web application for interactive predictions.

## Acknowledgments

This project was completed as part of the CSE 519 - Data Science course under Prof. Steven Skiena. The Zillow dataset was obtained from Kaggle.should 
