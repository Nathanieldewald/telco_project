from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

def split_xy(train, validate, test):
    x_train = train.drop(["churn", "customer_id","above_avg_monthly_charge","contract_type_month_to_month"], axis=1)
    y_train = train.churn
    x_validate = validate.drop(["churn", "customer_id"], axis=1)
    y_validate = validate.churn
    x_test = test.drop(["churn", "customer_id"], axis=1)
    y_test = test.churn
    return x_train, y_train, x_validate, y_validate, x_test, y_test
def validate_model_test(model, x_train, y_train, x_validate, y_validate):
    '''
    This function will take in a model, x_train, y_train, x_validate, y_validate
    and will return the accuracy of the model on the test data
    '''
    # fit the model using the training data
    model.fit(x_train, y_train)
    # predict churn using training data
    y_pred = model.predict(x_train)
    # evaluate the model on training data
    accuracy = model.score(x_train, y_train)
    # print out report
    print('------------------------------------')
    print(f'Model: {model}')
    print('Accuracy of model on training set: {:.3f}'
          .format(accuracy))
    # print out confusion matrix
    print(confusion_matrix(y_train, y_pred))
    # print out classification report
    print(classification_report(y_train, y_pred))

    # predict churn using validate data
    y_pred = model.predict(x_validate)
    # evaluate the model on validate data
    print('------------------------------------')
    print(f'Model: {model}')
    accuracy = model.score(x_validate, y_validate)
    print('Accuracy of model on validate set: {:.3f}'
          .format(accuracy))
    # print out confusion matrix
    print(confusion_matrix(y_validate, y_pred))
    # print out classification report
    print(classification_report(y_validate, y_pred))

def test_model_test(model, x_train, y_train, x_test, y_test):
    '''
    This function will take in a model, x_train, y_train, x_test, y_test
    and will return the accuracy of the model on the test data
    '''
    # fit the model using the training data
    model.fit(x_train, y_train)
    # predict churn using training data
    y_pred = model.predict(x_train)
    # evaluate the model on training data
    accuracy = model.score(x_train, y_train)
    # print out report
    print('------------------------------------')
    print(f'Model: {model}')
    print('Accuracy of model on training set: {:.3f}'
          .format(accuracy))
    # print out confusion matrix
    print(confusion_matrix(y_train, y_pred))
    # print out classification report
    print(classification_report(y_train, y_pred))

    # predict churn using test data
    y_pred = model.predict(x_test)
    # evaluate the model on test data
    print('------------------------------------')
    print(f'Model: {model}')
    accuracy = model.score(x_test, y_test)
    print('Accuracy of model on test set: {:.3f}'
          .format(accuracy))
    # print out confusion matrix
    print(confusion_matrix(y_test, y_pred))
    # print out classification report
    print(classification_report(y_test, y_pred))

def test_model_test2(x_train, y_train, x_test, y_test):
    model = LogisticRegression(penalty='l1', random_state=123, solver='liblinear', C=1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = round(model.score(x_test, y_test),3)
    print('------------------------------------')
    print(f'Model: {model}')
    print(f'Accuracy of model on test set: {score}')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def test_model_test2_short(x_train, y_train, x_test, y_test):
    model = LogisticRegression(penalty='l1', random_state=123, solver='liblinear', C=1)
    model.fit(x_train, y_train)
    score = round(model.score(x_test, y_test),3)
    print('------------------------------------')
    print(f'Model: {model}')
    print(f'Accuracy of model on test set: {score}')





def validate_model_test_short(model, x_train, y_train, x_validate, y_validate):
    '''
    This function will take in a model, x_train, y_train, x_validate, y_validate
    and will return the accuracy of the model on the test data
    '''
    # fit the model using the training data
    model.fit(x_train, y_train)
    # predict churn using training data
    y_pred = model.predict(x_train)
    # evaluate the model on training data
    accuracy = model.score(x_train, y_train)
    # print out report
    print(f'Accuracy of {model} on training set: {accuracy}')
    # predict churn using validate data
    y_pred = model.predict(x_validate)
    # evaluate the model on validate data
    accuracy = model.score(x_validate, y_validate)
    print(f'Accuracy of {model} on validate set: {accuracy}')



def visualize_model(x_train, y_train, x_validate, y_validate):

    cvalue = [.0001, .001, .01, .1, 1, 10, 100, 1000, 10000]
    train_accuracy = []
    validate_accuracy = []
    for c in cvalue:
        logit = LogisticRegression(penalty='l1', C=c, random_state=123, solver='liblinear')
        logit.fit(x_train, y_train)
        train_accuracy.append(logit.score(x_train, y_train))
        validate_accuracy.append(logit.score(x_validate, y_validate))
    plt.figure(figsize=(13, 7))
    plt.plot(cvalue, train_accuracy, label='Train', marker='o')
    plt.plot(cvalue, validate_accuracy, label='Validate', marker='o')
    plt.xticks(cvalue)
    plt.xscale('log')
    plt.ylabel('Accuracy')
    plt.xlabel('C Value in Log Scale')
    plt.title('Accuracy of Train and Validate Data Sets')
    plt.legend()