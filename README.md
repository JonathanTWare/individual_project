# <a name="top"></a>Individual Project - Time Series Project on Predicting Crime Counts
![]()


***
[[Project Description](#project_description)]
[[Project Planning](#planning)]
[[Key Findings](#findings)]
[[Data Dictionary](#dictionary)]
[[Data Acquire and Prep](#wrangle)]
[[Data Exploration](#explore)]
[[Modeling](#model)]
[[Conclusion](#conclusion)]
___



## <a name="project_description"></a>Project Description:

In this project we will be using the data from kaggle's chicago crime data set in order to predict specific crime counts for next year.

[[Back to top](#top)]

***
## <a name="planning"></a>Project Planning: 
[[Back to top](#top)]


### Objective
To achieve accurate predictions, I will develop a predictive model using machine learning techniques. I will explore different algorithms suitable for time series forecasting. Additionally, regression models like linear regression may be used to estimate crime counts based on the selected features.



### Target 
There are 3 Targets to this project. 'ASSAULT', 'BATTERY', and 'CRIMINAL DAMAGE'


### Need to haves (Deliverables):
- Need to explore the data.
- run autocorrelation visualizations
- Select features for modeling
- Run features through atleast 5 different algorythms.



### Nice to haves (With more time):
Further feature exploration to see if model prediction can increase. Look for data to add to the TSA such as population and weather.

***

## <a name="findings"></a>Key Findings:
- Assault had a week correlation with time
- Battery had a weak correlation with time
- Criminal Damages, although there. also had a weak correlation
- using 10 crime types, only 3 made the cut for modeling

[[Back to top](#top)]



## <a name="dictionary"></a>Data Dictionary  
[[Back to top](#top)]

### Data Used
---
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
|Date| Dates in order by day |int|
|THEFT| Amount of thefts that has occured in a particular day |int|
|BATTERY|Amount of battery that havs occured in a particular day |int|
|ASSAULT| Amount of assault that has occured in a particular day |int|
|CRIMINAL DAMAGE| Amount of criminal damage that has occured in a particular day |int|
|MOTOR VEHICLE THEFT| Amount of motor vehicle theft that has occured in a particular day |int|
|NARCOTICS|Amount of narcotic related crimes that have occured in a particular day |int|
|HOMICIDE| Amount of homicides that have occured in a particular day|int|
|HUMAN TRAFFICKING| Amount of human trafficking that has occured in a particular day |int|
|OFFENSE INVOLVING CHILDREN| Amount of offenses involving children that have occured in a particular day|int|
|KIDNAPPING| Amount of kidnapping that has occured in a particular day |int|
**
    

## <a name="wrangle"></a>Data Acquisition and Preparation
[[Back to top](#top)]

![]()

### Reproduce Project

- Install necessary python packages and kaggle pip install.
- Clone the individual_project repository.
- Use Wrangle function to acquire zip and extract data into current directory.
- Ensure the wrangle.py and model.py files are in the same folder as the crime_final_report.ipynb notebook.


### Wrangle steps: 
- dropped duplicate rows.
- dropped nulls.
- created function to acquire and prep data
- function created to split data into train, validate and test



*********************

## <a name="explore"></a>Data Exploration:
[[Back to top](#top)]
- Python files used for exploration:
    - wrangle.py
    
    
    
    


### Takeaways from exploration:
- Three crime types out of ten were chosen for seperate targets for modeling using autocorrelation. .



***

## <a name="model"></a>Modeling:
[[Back to top](#top)]

***

### 'ASSAULT' modeling

#### The best model for ASSAULT (Simple Exponential Smoothing RMSE 9.84) did beat baseline(Moving Average Baseline RMSE 9.86), and when running it through a test score it obtained a 9.94 RMSE which failed to beat baseline.


### 'BATTERY' modeling

#### The best model for BATTERY (Holts Linear Trend Forecasting RMSE 29.83) did not beat baseline(Moving Average Baseline RMSE 25.32), and when running it through a test score it obtained a 139 RMSE which failed to beat baseline.

 
### 'CRIMINAL DAMAGE' modeling

#### The best model for CRIMINAL DAMAGE (Holts Linear Trend Forecasting RMSE 18.39) did not beat baseline(Moving Average Baseline RMSE 12.23), and when running it through a test score it obtained a 15.24 RMSE which failed to beat baseline.

***

## <a name="conclusion"></a>Conclusion:

#### Based on the information provided, it seems that the Moving Avg Baseline did better in modeling overall.
#### 
#### For the 10 crimes that went through autocorrelation, only 3 of them showed any correlation and still were not as predictable as baseline
#### Considering all models, as none beat baseline, Further exploring and feature engineering maybe needed in order to predict crimes using time. It may be that time does not influence crime type count daily.
#### One next step would be to add population as a feature to see if that would help the model predict these crime counts more accurately.

[[Back to top](#top)]