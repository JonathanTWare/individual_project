# <a name="top"></a>Wine Project - Finding Drivers of Wine Quality
![]()


***
[[Project Description](#project_description)]
[[Project Planning](#planning)]
[[Key Findings](#findings)]
[[Data Dictionary](#dictionary)]
[[Data Acquire and Prep](#wrangle)]
[[Data Exploration](#explore)]
[[Statistical Analysis](#stats)]
[[Modeling](#model)]
[[Conclusion](#conclusion)]
___



## <a name="project_description"></a>Project Description:

In this project we will be using the data from the California Wine Institute. Exploring the data, we will find drivers of wine's quality. The goal is to find these drivers while utilizing clusters and presenting how it affects a machine learning model.

[[Back to top](#top)]

***
## <a name="planning"></a>Project Planning: 
[[Back to top](#top)]


### Objective
Identify Key Features: Determine the features within the dataset that exhibit strong correlations with wine quality.
Build Predictive Model: Develop a high-performing model that can accurately estimate the quality of wine based on the selected features.
Share Insights: Communicate the findings and model insights to the data science team for further analysis and decision-making.

Find the main drivers of what dictates wine quality.



### Target 
The target of this project is quality.


### Need to haves (Deliverables):
-Need to explore the data.
-Run features through statistical tests.
-Select features for modeling
-Run features through atleast 4 different algorythms.



### Nice to haves (With more time):
Further feature explore to see if additional groups can be identified before modeling.


***

## <a name="findings"></a>Key Findings:
- Proof is highly correlated with wine quality.
- Citric acid is moderately correlated with wine quality.
- Free sulfur dioxide had a week correlation with wine quality.

[[Back to top](#top)]



## <a name="dictionary"></a>Data Dictionary  
[[Back to top](#top)]

### Data Used
---
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
|fixed acidity| Amount of non-volatile acids in a substance |float|
|volatile acidity| Amount of volatile acids in a substance |float|
|citric acid| Amount of citric acid in a substance |float|
|residual sugar| Amount of residual sugar in a substance |float|
|chlorides| Concentration of chlorides in a substance |float|
|free sulfur dioxide| Level of free sulfur dioxide in a substance |float|
|total sulfur dioxide|Total sulfur dioxide content in a substance |float|
|density| The density of a substance |float|
|pH| The pH level of a substance |float|
|sulphates| Amount of sulfates in a substance|float|
|proof|  twice the percentage of alcohol by volume | float |
|quality| The quality rating of a substance |float|
|strain| Type of wine | object|
**

## <a name="wrangle"></a>Data Acquisition and Preparation
[[Back to top](#top)]

![]()

### Reproduce Project

- Install necessary python packages.
- Clone the wine_clustering_project repository.
- Download files from https://data.world/food/wine-quality
- Unzip and store the csv files in the wine_clustering_project folder.
- Ensure the wrangle.py, explore.py and model.py files are in the same folder as the wine_final_report.ipynb notebook.


### Wrangle steps: 
- dropped duplicate rows.
- created dummies for certain features
- created function to acquire and prep data
- function created to scale certain features
- renamed acohol column to 'proof'


*********************

## <a name="explore"></a>Data Exploration:
[[Back to top](#top)]
- Python files used for exploration:
    - explore.py
    - wrangle.py
    
    
    
    


### Takeaways from exploration:
- Three features were chosen for statistical testing: proof, citric acid, free sulfur dioxide.


***

## <a name="stats"></a>Statistical Analysis
[[Back to top](#top)]

### Stats Test 1: Pearson's R

Pearson's correlation coefficient (Pearson's R) is a statistical measure used to assess the strength and direction of the linear relationship between two continuous variables.

By calculating Pearson's R, we aim to determine whether there is a significant linear association between the independent variable and the dependent variable. The coefficient helps us quantify the extent to which the variables vary together and provides insight into the direction (positive or negative) and strength (magnitude) of the relationship.

To calculate Pearson's R in Python, we can use the corrcoef function from the numpy module. This function takes the two variables as input and returns the correlation matrix, where the coefficient of interest is the element in the [0, 1] or [1, 0] position. Pearson's R ranges from -1 to 1, where -1 indicates a perfect negative linear relationship, 0 indicates no linear relationship, and 1 indicates a perfect positive linear relationship.


### Hypothesis  Initial hypotheses and/or questions you have of the data, ideas:


In summary, the hypotheses for the PearsonsR test can be stated as follows:

### 1st Hypothesis Does Proof  Affect Wine Quality?


Null Hypothesis (H0): proof does not have a correlation with wine quality.
Alternative Hypothesis (H1): proof has a correlation with wine quality.

### 2nd Hypothesis Does Free Sulfur Dioxide Affect Wine Quality?


Null Hypothesis (H0): Free sulfur dioxide does not have a correlation with wine quality.
Alternative Hypothesis (H1): Free sulfur dioxide has a correlation with wine quality.

### 3rd Hypothesis Does Citric Acid Affect Wine Quality?


Null Hypothesis (H0): Citric acid does not have a correlation with wine quality.
Alternative Hypothesis (H1): Citric acid has a correlation with wine quality.

#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05

#### Results:

| Feature | Corellation Value | Correlation Strength|
| ---- | ---- | ---- |
| proof|0.48  |strong |
| citric acid|0.11 | moderate |
| free sulfur dioxide |0.06  | weak |



#### Summary: 
Proof had the highest correlation with wine quality, while citric acid and free sulfur dioxide had a correlation but a much weaker one.




***

## <a name="model"></a>Modeling:
[[Back to top](#top)]


### Baseline
    
- Baseline Results: 

| Model | Train Score | 
| ---- | ---- | 
| Baseline | 0.448137 | 

- Selected features to input into models:
    - features = bathroom_count, bedroom_count, calc_sqr_ft, yearbuilt, and all encoded county codes.

***

### Top 3 Models

    
#### Model 1: 3 Feature Logistic Regression(LogReg)


##### The 3 Feature Logistic Regression model had a validation accuracy of 52% which was 7% better than baseline's 45% accuracy



### Model 2 : cluster_k6 with proof, free sulfur dioxide, citric acid LogReg


 
##### cluster_k6 with proof, free sulfur dioxide, citric acid LogReg model had a validation accuracy of 52% which was 7% better than baseline's 45% accuracy



### Model 3 : cluster_k5 with proof, free sulfur dioxide, citric acid LogReg



##### cluster_k5 with proof, free sulfur dioxide, citric acid LogReg model had a validation accuracy of 52% which was 7% better than baseline's 45% accuracy



## Selecting the Best Model:

### Use Table below as a template for all Modeling results for easy comparison:

|model |train |validate|
| ---- | ----| ---- |
|3 Feature LogReg|0.530424|0.521812|
|cluster_k6 with proof, free sulfur dioxide, citric acid LogReg|0.528745|0.52518|
|cluster_k5 with proof, free sulfur dioxide, citric acid LogReg model had a validation accuracy of 52% which was 7% better than baseline's 45% accuracy|0.52916|0.52684|


##### The 3 Feature Logistic Regression model preformed best


## Testing the Model

- Model Testing Results

|model |train |validate| Test |
| ---- | ----| ---- | ---- |
|3 Feature LogReg|0.530424|0.521812|0.50|



***

## <a name="conclusion"></a>Conclusion:

#### Based on the information provided, it seems that the The 3 Feature Logistic Regression model has the highest accuracy with lowest amount of features at 50% test accuracy , which is 5% better than baseline. 
#### 
#### On the other hand, the "cluster_k6 with proof, free sulfur dioxide, citric acid LogReg" model had the same train and validate score but included an additional cluster feature.
#### 
#### The "cluster_k5 with proof, free sulfur dioxide, citric acid LogReg model" also had an additional feature with the same train and validate scores.
#### 
#### Considering all models, as they did all beat baseline, the 3 Feature LogReg model was picked as most optimal as it contained the lowest amount of features for the same scoring outputs.
####


***
## <a name="sources"></a>Sources:


#### 1. Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. Decision Support Systems, 47(4), 547-553. Retrieved from http://dx.doi.org/10.1016/j.dss.2009.05.016.
#### 
#### 2. Preda, C., Lopes, G. R. V. D., & Rodrigues, L. M. C. (2009). Wine Quality Data Set. data.world. Retrieved from https://data.world/food/wine-quality.
#### 



[[Back to top](#top)]