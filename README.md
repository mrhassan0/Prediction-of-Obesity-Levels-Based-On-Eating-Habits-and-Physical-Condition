# Prediction-of-Obesity-Levels-Based-On-Eating-Habits-and-Physical-Condition
The "Estimation of Obesity Levels Based on Eating Habits and Physical Condition" dataset aims to predict an individual's obesity level based on features like eating habits, physical condition, and lifestyle factors. The target variable has multiple classes, making it a classification problem focused on identifying patterns for accurate predictions.
1. Introduction
The "Estimation of Obesity Levels Based on Eating Habits and Physical Condition" dataset is designed to solve a classification problem that aims to predict an individual's obesity level based on various features. These features include factors related to eating habits, physical conditions, and other lifestyle factors. The target variable is the obesity level, which is categorized into different classes, providing an opportunity to build a model that classifies individuals into these obesity categories. This problem involves extracting meaningful patterns from these features to make accurate predictions.

2. Dataset Description
Source
Link: Dataset on ScienceDirect https://www.sciencedirect.com/science/article/pii/S2352340919306985?via%3Dihub
Reference: All references

Dataset Description
Number of Features: 17 features (including the target variable NObeyesdad).


Type of Problem: Classification problem. The target variable contains categorical labels representing obesity levels.


Number of Data Points: 2111 data points.


Type of Features:


Quantitative Features: Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE.
Categorical Features: Gender, family_history_with_overweight, FAVC, CAEC, SMOKE, SCC, CALC, MTRANS, NObeyesdad.
Correlation of Features: The heatmap below displays the correlation matrix for quantitative features:






Imbalanced Dataset
Class Distribution:  The chart represents the seven unique classes (N=7) in the dataset, showing the number of instances for each class.
The target variable (NObeyesdad) has the following class distribution:


Obesity_Type_I: 351 instances
Obesity_Type_III: 324 instances
Obesity_Type_II: 297 instances
Overweight_Level_I: 290 instances
Overweight_Level_II: 290 instances
Normal_Weight: 287 instances
Insufficient_Weight: 272 instances

The dataset is moderately balanced, but not all unique classes have an equal number of instances. This variation in class sizes could impact model performance, especially for underrepresented classes. Addressing this imbalance might involve techniques such as resampling or weighting the classes during training.


3. Dataset pre-processing
Problem 1: Null Values
Identification:
Upon inspection, no null values were identified in the dataset. This simplifies pre-processing as no missing data imputation is required.
Problem 2: Categorical Values
Identification:
Categorical variables such as Gender, family_history_with_overweight, FAVC, CAEC, SMOKE, SCC, CALC, MTRANS, and NObeyesdad need to be encoded for model compatibility.
Solutions:
One-Hot Encoding:
Applied to multi-class categorical variables such as CAEC, MTRANS, and NObeyesdad.
Procedure: Each unique category in a column is represented as a separate binary column (1 if the category is present for a row, otherwise 0). For instance, CAEC with values like Sometimes, Frequently, and No is transformed into columns CAEC_Sometimes, CAEC_Frequently, and CAEC_No.
Justification: Avoids introducing false ordinal relationships and ensures fair representation of categories.
Example: The dataset expanded significantly with one-hot encoding, ensuring all categorical variables were appropriately converted for machine learning compatibility【6:1†source】.
Problem 3: Irrelevant Columns
Identification:
Columns such as SMOKE, SCC, TUE, and MTRANS were identified as less relevant for predicting obesity levels.
Solutions:
Column Removal:
Dropped columns with little predictive value to simplify the dataset.
Justification: Reduces dimensionality and computational cost.
Problem 4: Feature Transformation
Identification:
The Height and Weight columns were more meaningful as a single feature, BMI (Body Mass Index).
Solutions:
BMI Calculation:
Calculated as BMI = Weight / (Height^2) and replaced the original Height and Weight columns.
Justification: BMI provides a standardized measure for assessing obesity.
Problem 5: Feature Scaling
Identification:
Numerical features (Age, FCVC, NCP, CH2O, FAF, and BMI) exhibited varying ranges, potentially impacting model performance.
Solutions:
Standard Scaling:
Transformed features to have a mean of 0 and a standard deviation of 1.
Procedure: Each value in a numerical column is scaled using the formula: z=(X−mean)standard deviationz = \frac{(X - \text{mean})}{\text{standard deviation}}. This ensures uniformity across different scales.
Justification: Standardization is essential for distance-based algorithms like KNN and logistic regression to ensure that features contribute equally
Example: Numerical columns such as Age, BMI, and FAF were scaled uniformly, ensuring improved model interpretability and performance.
4. Feature scaling
Feature scaling ensures equal contribution from numerical features in machine learning models. Without scaling, algorithms like logistic regression and KNN may produce biased results.
Standard Scaling
Purpose: Transforms data to have a mean of 0 and standard deviation of 1.
Formula: z=x−μσz = \frac{x - \mu}{\sigma}, where μ\mu is the mean and σ\sigma is the standard deviation.
Implementation
Library: StandardScaler from sklearn.preprocessing.
Columns Scaled: Age, BMI, FCVC, CH2O, FAF.
Code:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['Age', 'BMI', 'FCVC', 'CH2O', 'FAF']])

Result
Before scaling, features like Age (14-61) and BMI had varied ranges. After scaling, all features followed a standard normal distribution. This improved model accuracy, and convergence, and ensured fair contributions from all features.

5. Dataset Splitting : 
We use a total of 70% of data in training and 30% of data in testing:
Features Training : (1477,15)
Features Testing :  (634,15)

Target Training: (1477,1)
Target Testing : (634,1)

6. Model and Testing: 
As this is a multi-class classification problem, we use four models here : 
KNN
Logistic Regression
Decision Tree
Naive Bayes

7. Model Selection and Comparison Analysis

Accuracy levels of KNN, Logistic Regression, Decision Tree, and Naive Bayes are shown below:

Recall and Precision Comparison
Recall: focuses on identifying all actual positive instances. 
Precision: focuses on the accuracy of positive predictions.


Confusion Matrix:
Rows represent the true labels (what the actual category is).
Columns represent the predicted labels (what the model guessed).

8. Conclusion
With proper preprocessing, feature selection, and classification techniques, this model can be used to predict obesity risk based on lifestyle factors, potentially supporting health professionals in preventive healthcare and personalized treatment plans. The results may offer valuable insights into the role of eating habits and physical conditions in obesity classification, ultimately contributing to public health efforts.





