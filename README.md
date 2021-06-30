# BDT_MWC_Hackathon
Google Colab that became the winner in the individual phase of the Data Science category from the national Hackathon competition held by NUWE and Barcelona Digital Talent for the Mobile World Congress 2021.

The challenge was based on predict the type of the particle that each collision in the LHC particle accelerator from Switzerland produces. The dataset was made of 127,321 observations and 11 features (plus the type of the particle as variable target, with 8 types in total). The aim of the challenge was to generate a Machine Learning model that allows to predict which particle has been produced after the collision of hadrons from the measurements taken by the LHC sensors

**Stack:** Python, Scikit-Learn, Pandas, Numpy and Matplotlib.

The assessment of the projects were based on the F1-Score metric with the 'Macro' criterion, as well as structure and code explanation and documentation.

The approach was based on performing the following steps:

**1) Exploratory Analysis**, which led to:
    - Removing 1 variable with total Collinearity.
    - Removing NA values through iterative MICE (Multivariate Imputation by Chained Equations) algorithms.
    - Balancing class proportions through a combined approach using unbalancing and SMOTE algorithm.

**2) Feature engineering**, using 'Featuretools' library (traditionally used for Relational Databases), adapted to be used in a standalone DataFrame.

**3) Feature Selection**, using a statistical test of 'F' of Fisher with regard to the Response Variable.

**4) Standarization** of the variables and **PCA** as dimensionality reduction statistical technique to obtain the highest possible explanatory and uncorrelated variables.


After these steps with the data ready to be tackled, simple linear models were used as a baseline to build upon them, using Non-Linear approaches further on. In summary, the following statistical and Machine Learning techniques were applied:

**· Linear Discriminant Analysis (LDA)**

**· Quadratic Discriminant Analysis (QDA)**

**· Support Vector Machine (SVM) with RBF Kernel**

**· Random Forest**

**· Gradient Boosted Trees (XGBoost)**

Eventually, since the classes to predict happened to be defined by **Multivariate Gaussian distributions with different covariances matrices**, the most suitable method that achieved the highest F1-Score metric ended up being QDA, with a final **F1-Score of 99.45%**, although some Data Leakage influenced this value.
