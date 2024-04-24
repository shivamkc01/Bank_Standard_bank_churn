---

# Building a Next Best Action Model for Standard Bank

## Overview:

Modern customer analytics and personalization systems aim to understand and quantify customer preferences and intent to enhance marketing strategies. However, optimizing immediate interactions with customers using metrics like click-through rate (CTR) or conversion rate (CR) may not suffice, especially in industries like retail banking and telecom, where customer relationships span long periods. The "Building a Next Best Action Model for Standard Bank" project addresses this issue by combining a high-performing classification engine with a recommendation engine. This README provides an in-depth overview of the project, including methodologies employed, steps taken, and results obtained.

## Project Stages:

### Stage 1: Classification Engine

1. **Data Preprocessing:**
   - **Column Names:** Renamed columns for clarity and consistency.
   - **Feature Selection:** Removed irrelevant features to improve model efficiency.
   - **Handling Missing Values:** Utilized K-Means clustering to impute missing values, ensuring data integrity.
   - **Categorical Features:** Encoded categorical features using LabelEncoder for model compatibility.
   - **Outlier Detection:** Employed Isolation Forest to identify and remove outliers, enhancing model robustness.
   - **Feature Scaling:** Applied Quantile Transformer for feature scaling, ensuring uniformity in feature distributions.
   - **Data Balancing:** Used SMOTE method to balance the dataset, addressing class imbalance.

2. **EDA:**
   ![](https://github.com/shivamkc01/Bank_Standard_bank_churn/blob/main/plots/ppts_figs/categorical_features_dist_with_status.png)
   
   ![](https://github.com/shivamkc01/Bank_Standard_bank_churn/blob/main/plots/ppts_figs/monthly_income_dist_with_status.png)
 
   ![](https://github.com/shivamkc01/Bank_Standard_bank_churn/blob/main/plots/ppts_figs/year_with_us_dist_with_status.png)
 
   ![](https://github.com/shivamkc01/Bank_Standard_bank_churn/blob/main/plots/ppts_figs/Mean_values_features_churned_and_not_churned_customers.jpg)
   
   ![](https://github.com/shivamkc01/Bank_Standard_bank_churn/blob/main/plots/ppts_figs/target_class_dist.png)

   ![](https://github.com/shivamkc01/Bank_Standard_bank_churn/blob/main/plots/corr.jpg)
4. **Modeling:**

   ![](https://www.mdpi.com/mathematics/mathematics-10-02379/article_deploy/html/images/mathematics-10-02379-g001.png)

   - I have created a **StratifiedKFold** by using that I check the class distrubtion in each folds
   ![](https://github.com/shivamkc01/Bank_Standard_bank_churn/blob/main/plots/dist_of_class_in_folds.jpg)
   <center><b><u><span style="color:#ff6600">OBSERVATIONS</span></u></b></center>

   - In each **fold**, there is a noticeable class imbalance between **ACTIVE** and **CHURN** classes. The **ACTIVE** class appears more frequently than the **CHURN** class.

   - It seems like after using **StratifiedKFold** to create your folds, our target classes are distributed badly I would say. However, if the majority class (class 1.0) is significantly larger than the minority classes (classes 2.0 and 3.0), we still end up with `imbalanced folds`.

   **we might want to consider techniques such as class weighting, `resampling method`s (e.g., `oversampling` the minority class), or using `evaluation metrics` that are sensitive to class imbalance to ensure that our model effectively learns from both classes and generalizes well**
   ## <center>`Lets understand **Undersampling** and **Oversampling**`</center>

**Oversampling:** In oversampling, you increase the number of instances of the minority class (in our case, "CHURN") to balance the class distribution. This can be done using techniques like `SMOTE (Synthetic Minority Over-sampling Technique)`, `ADASYN (Adaptive Synthetic Sampling)`, or simply by duplicating instances from the minority class.

**Undersampling:** In undersampling, you decrease the number of instances of the majority class (in this case, "ACTIVE") to balance the class distribution. This can involve randomly removing instances from the majority class until a more balanced distribution is achieved.

<div style="text-align:center;">
    <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*7xf9e1EaoK5n05izIFBouA.png" alt="Image" />
</div>


<hr>

### <b><u><span style="color:#ff6600">Each technique has its own set of advantages and disadvantages:</span></u></b></center></span></u></b>
#### `Undersampling:`

**Pros:**

- **`Reduces Training Time`**: By reducing the number of instances in the majority class, undersampling can significantly decrease training time, especially for algorithms sensitive to large datasets.
- **`Simplifies Model Interpretation`**: With a more balanced dataset, models may be simpler and easier to interpret, as they don't need to learn the nuances of the majority class as extensively.
- **`May Improve Performance on the Minority Class`**: By focusing more on the minority class, models trained on undersampled data may achieve higher precision, recall, or other performance metrics for the minority class.

**Cons:**

- **`Information Loss`**: Removing instances from the majority class can lead to the loss of potentially valuable information, which may degrade the model's ability to generalize to unseen data.
- **`Risk of Overfitting`**: Undersampling may increase the risk of overfitting, particularly if the dataset is already limited in size.
- **`Biased Representation`**: The reduced representation of the majority class may not accurately reflect its true distribution, leading to biased model predictions.

<hr>

#### `Oversampling:`

**Pros:**

- **`Preserves Information`**: Oversampling techniques generate synthetic samples or replicate existing minority class instances, preserving information from the minority class and potentially improving model performance.
- **`Balances Class Distribution`**: By increasing the representation of the minority class, oversampling can help balance the class distribution, reducing bias in model predictions.
- **`Reduces Risk of Overfitting`**: Oversampling increases the amount of training data available to the model, which can reduce the risk of overfitting, especially in cases of severe class imbalance.

**Cons:**

- **`Potential Overfitting`**: Oversampling can introduce synthetic data points that may not accurately represent the true distribution of the minority class, leading to overfitting, especially if not carefully implemented.
- **`Increased Computational Cost`**: Generating synthetic samples or replicating minority class instances can increase computational cost, especially for large datasets.
- **`Model Sensitivity`**: Some oversampling techniques may introduce noise or patterns that are not representative of the underlying data distribution, potentially leading to decreased model performance.


``**Note**:In our case Oversampling the "CHURN" class might be a better choice. By generating synthetic samples or duplicating instances from the "CHURN" class, you can increase its representation in the dataset and balance the class distribution. This approach can help prevent the model from being biased towards the majority class and improve its ability to learn patterns from the minority class. But we will try both and we'll see which one is better for our case``


## **Result with using above folds** </br>

Bias Towards Majority Class

Misrepresentation of Minority Class

Difficulty in Detecting Minority Class

Overfitting to Majority Class
**`ROC Score  TRAINING SET=0.9566197629615178, TESTING SET=0.8303359683794465`**
`
<hr>

   ## <u>After using SMOTE:</u></br>
   ## **Result with SMOTE folds** </br>
   **`ROC Score TRAINING SET=0.9464331518139426, TESTING SET=0.9051433261292333`**
   
   ![](https://github.com/shivamkc01/Bank_Standard_bank_churn/blob/main/plots/dist_of_class_using_smote_in_folds.jpg)
   <center><b><u><span style="color:#ff6600">OBSERVATIONS✍️</span></u></b></center>

   The class distribution after applying `SMOTE oversampling technique`. It seems that the class distribution is now balanced in each fold, with both classes (0 and 1) having approximately the same number of instances.

    **Reasons why❓, in my perspective, it works**:
    - It helps address the issue of class imbalance in your dataset. By generating synthetic samples for the minority class (1), SMOTE has effectively increased its representation to match that of the majority class (0), resulting in a more balanced dataset.
    - A balanced class distribution is beneficial for training machine learning models, as it helps prevent bias towards the majority class and allows the model to learn from both classes equally. This can lead to improved model performance and better generalization to unseen data.


   - Implemented the following classification algorithms:
     - ## Logistic Regression
     ```python
       linear_model.LogisticRegression(
              penalty='l2', 
              C=1.0, 
              solver='liblinear'
        )
     ```
     ![](https://github.com/shivamkc01/Bank_Standard_bank_churn/blob/main/plots/roc_auc_curve/roc_curve_training_testing_lr.png)

     - ## Decision Tree
       ```python
         tree.DecisionTreeClassifier(
            max_depth=10, 
            min_samples_split=50, 
            min_samples_leaf=10
          )
      ```
  ![](https://github.com/shivamkc01/Bank_Standard_bank_churn/blob/main/plots/roc_auc_curve/roc_curve_training_testing_rf.png)
      
     - Random Forest
     - XGBoost
   - Utilized GridSearchCV for hyperparameter tuning to optimize model performance.

6. **Model Evaluation:**
   - Evaluated models using precision, recall, and ROC-AUC metrics.
   - Selected XGBoost as the best-performing model based on high ROC-AUC scores.

### Stage 2: Recommendation Engine

1. **Data Preparation:**
   - Created a dummy dataset related to types of churn customers and product recommendations.
   - Recommendations aimed to increase conversion rates and retain churn customers.

## Results:

- **Model Performance:**
  - XGBoost emerged as the top-performing model with ROC-AUC scores of 0.9952 (train) and 0.9584 (test).
  - Precision, recall, and ROC-AUC metrics were used to assess model effectiveness.

## Conclusion:

The project successfully integrated a classification engine with a recommendation engine to address churn in the Standard Bank dataset. By employing advanced preprocessing techniques and fine-tuning multiple classification algorithms, we achieved a high-performing model. XGBoost demonstrated robust performance, indicating its effectiveness in predicting and retaining churn customers.

## Future Work:

- **Recommendation Engine Refinement:** Further enhancement of the recommendation engine to offer personalized product suggestions.
- **Advanced Techniques:** Exploration of advanced machine learning techniques and real-time data integration to improve model accuracy and responsiveness.

---

