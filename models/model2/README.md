# Model 2 - Logistic regression

I'll use Logistic regression to classify the images in this dataset.

##  Logistic regression

Logistic regression is named for the function used at the core of the method, the logistic function. In linear regression, the outcome (dependent variable) is continuous. It can have any one of an infinite number of possible values. In logistic regression, the outcome (dependent variable) has only a limited number of possible values. Logistic Regression is used when response variable is categorical in nature.

---

### Logistic regression in TensorFlow:

<figure>
 <img src="./screenshots/softmax.png" width="600" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>

```
# Parameters
learning_rate = 0.05
training_epochs = 1000
batch_size = 64
display_step = 1
```
```
-  Input layer: Must be 32x32 = 1024 inputs.
-  Output layer: Must be 43 output that represent the classes.
-  Weight matriz: The dimension will be [1024,43]
-  Bias: The dimension will be [43]
```

---

### Model Training and Evaluation

For **model 2** we have to put the following comand:
```
python app.py train -m model_2 -d images/train
```
The output will be the accuracy of model 2:
```
Epoch: 0 cost=6.138635533196585 Accuracy of epoch is:0.016736401
Epoch: 50 cost=1.5487551603998457 Accuracy of epoch is:0.64853555
Epoch: 100 cost=0.7987359251294818 Accuracy of epoch is:0.76987445
Epoch: 150 cost=0.5192108750343323 Accuracy of epoch is:0.8117155
Epoch: 200 cost=0.37372345903090076 Accuracy of epoch is:0.83682007
Epoch: 250 cost=0.28807814099958967 Accuracy of epoch is:0.8535565
Epoch: 300 cost=0.23311320479427067 Accuracy of epoch is:0.8535565
Epoch: 350 cost=0.19514039903879166 Accuracy of epoch is:0.8535565
Epoch: 400 cost=0.16744607846651757 Accuracy of epoch is:0.8577406
Epoch: 450 cost=0.14643785410693713 Accuracy of epoch is:0.8619247
Optimization Finished!
Train accuracy of model 2 is:86.6108775138855
Elapsed time: 15.973098754882812 seconds
Model saved to: models/model2/saved/log_reg_tF.ckpt
```
We've been able to reach a maximum accuracy of **86.61%** on the validation set.

---

### Testing the Model using the Test Set

For **model 2** we have to put the following comand:
```
python app.py test -m model_2 -d images/test
```
The output will be the accuracy of model 2:
```
Model restored
Test accuraccy of model 2 is: 89.36877250671387
```
We've been able to reach a maximum accuracy of **89.36%** on the test set.

