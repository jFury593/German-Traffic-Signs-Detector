# Model 1 - Logistic regression

I'll use Logistic regression to classify the images in this dataset.

##  Logistic regression

Logistic regression is named for the function used at the core of the method, the logistic function. In linear regression, the outcome (dependent variable) is continuous. It can have any one of an infinite number of possible values. In logistic regression, the outcome (dependent variable) has only a limited number of possible values. Logistic Regression is used when response variable is categorical in nature.

### Logistic regression in Scikit-learn architecture:
```
Tolerance for stopping criteria = 0.0001
Maximum number of iterations = 1000

```
Number of inputs = 1024, that correspond to the flatten image.

Number of outputs = 43, thath correspond to each class.
```
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=1000, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
```


---

### Model Training and Evaluation

For **model 1** we have to put the following comand:
```
python app.py train -m model_1 -d images/train
```
The output will be the accuracy of model 1:
```
Model is logistic regression with scikit-learn
Train accuracy of model 1 is: 85.77405857740585
```
We've been able to reach a maximum accuracy of **85.77%** on the validation set.

### Testing the Model using the Test Set

For **model 1** we have to put the following comand:
```
python app.py test -m model_1 -d images/test
```
The output will be the accuracy of model 1:
```
Model is logistic regression with scikit-learn
Test accuraccy of model 1 is: 89.03654485049833
```
We've been able to reach a maximum accuracy of **89.03%** on the test set.

