# Telco Project

# Project Description
The purpose of this project is to discover drivers of churn in Telco customers and use those drivers to develop a machine learning model that can predict if a customer will churn.

# Project Goal
 
* To develop a model that can predict if a customer will churn.
* Find drivers of churn
* Create a model that performs better than the baseline
* Document code and process well enough to be presented or read like a report

# Data Dictionary
| **Column**          | **Description**                                           |
|---------------------|-----------------------------------------------------------|
| **Customer ID**     | Customer ID                                               |
| **Gender**          | Whether the customer is a male or a female                |
| **SeniorCitizen**   | Whether the customer is a senior citizen or not           |
| **Partner**         | Whether the customer has a partner or not                 |
| **Dependents**      | Whether the customer has dependents or not                |
| **Tenure**          | Number of months the customer has stayed with the company |
| **PhoneService**    | Whether the customer has a phone service or not           |
| **Multiplelines**   | Whether the customer has multiple lines or not            |
| **InternetService** | Customer’s internet service provider                      |
| **OnlineSecurity**  | Whether the customer has online security or not           |
| **OnlineBackup**    | Whether the customer has online backup or not             |
| **DeviceProtection** | Whether the customer has device protection or not         |
| **TechSupport**     | Whether the customer has tech support or not              |
| **StreamingTV**     | Whether the customer has streaming TV or not              |
| **StreamingMovies** | Whether the customer has streaming movies or not          |
| **Contract**        | The contract term of the customer                         |
| **PaperlessBilling** | Whether the customer has paperless billing or not         |
| **PaymentMethod**   | The customer’s payment method                             |
| **MonthlyCharges**  | The amount charged to the customer monthly                |
| **TotalCharges**    | The total amount charged to the customer                  |
| **Churn**           | Whether the customer churned or not                       |


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
       * Is fiber optic price a driver of churn?
       * Is contract type a driver of churn?
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
* The drivers of churn are:
    * tenure
    * monthly charges
    * fiber optic
    * contract type
    * dependents
* The best model is the Logistic Regression model 
* All performance metrics are better than the baseline of 73.46%

 
# Recommendations
 * Offer incentives to customers with fiber optic to switch to DSL
 * Offer incentives to customers with monthly charges above the average to switch to a lower monthly charge
 * Offer incentives to customers with month-to-month contracts to switch to a one or two year contract
 * Offer incentives to customers without dependents to add dependents to their account