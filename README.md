# SMS-Classifier-Using-Logistic-Regression
The repository contains Python code for a binary SMS classifier using logistic regression. It includes data preprocessing, feature extraction, model training, evaluation, and prediction functionalities, providing a comprehensive solution for classifying SMS messages as spam or not spam. 

![image](https://github.com/lekshmiij/SMS-Classifier-Using-Logistic-Regression/assets/141242851/97be0474-5e30-4841-9454-52ef706812bd)

here's a comprehensive overview of the processes involved in the SMS Classifier code:
### Data Loading and Preprocessing:
* Loaded SMS data from a CSV file using Pandas.
* Handled missing values by replacing them with a null string.
* Performed label encoding to represent spam as 0 and ham as 1.

### Data Splitting and Feature Extraction:
* Split the data into training and testing sets using train_test_split from sklearn.
* Utilized TfidfVectorizer to convert text data into numerical form for model training.

### Model Training and Evaluation:
* Trained a logistic regression model on the training data to classify SMS as spam or ham.
* Evaluated the model's performance using accuracy_score on both training and testing data to ensure generalization.

### Prediction:
* Applied the trained model to make predictions on new SMS messages, determining whether they are spam or ham.

