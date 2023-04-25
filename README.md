# Telco Project

# Project Description
The purpose of this project is to discover drivers of churn in Telco customers and use those drivers to develop a machine learning model that can predict if a customer will churn.

# Project Goal
 
* the goal of this project is to develop a model that can predict if a customer will churn.
* Deliver a notebook that documents your process from start to finish
* Find drivers of churn
* Include detailed comments that explain your process and decisions
* Clearly call out the questions you are answering, the decisions you are making, and the processes you are using
* Document your key takeaways and conclusions
* Create module wrangle.py that contains functions that make your process repeateable

# Initial Thoughts

My initail thoughts are that the drivers of churn will be:
* tenure
* monthly charges
* fiber optic

# The Plan
 
* Aquire data from the Codeup Database and create a pandas dataframe
 
* Prepare data
   * Clean data
    * Handle missing values
    * Handle erroneous data
    * Handle outliers
    * Encode variables
    * Split data into train, validate, test
 
* Explore data in search of drivers of churn
   * Answer the following initial questions
       * Is fiber optic a driver of churn?
       * Is there a price threshold for specific services where customers are more likely to churn?
       * Does tenure correlate with higher or lower churn?
       * Are customers with dependents more likely to churn than those without?
      
* Develop a Model to predict if a customer will churn
  * Use drivers identified in explore to build predictive models of different types
  * Evaluate models on train and validate data
  * Select the best model based on highest accuracy
  * Evaluate the best model on test data
* Draw conclusions
 
 
# Steps to Reproduce
1) Clone this repo.
2) Acquire the data from the Codeup database using the wrangle.py script in this repository.
3) Put the data in the file containing the cloned repo.
4) Run notebook.

# Takeaways and Conclusions
* 
 
# Recommendations
* 