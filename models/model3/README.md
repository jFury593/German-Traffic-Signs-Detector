# Model 3 - LeNet-5

LeNet-5 is a convolutional network designed for handwritten and machine-printed character recognition in the year 1998. It was introduced by Yann LeCun, in his paper [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf). 

---

## Design a Model Architecture

I'll use Logistic regression and Convolutional Neural Networks to classify the images in this dataset.

**LeNet-5 architecture:**

In the paper we can see this structure propose by Yann LeCun:

<figure>
 <img src="./screenshots/lenet.png" width="600" alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>

Input => Convolution => ReLU => Pooling => Convolution => ReLU => Pooling => FullyConnected => ReLU => FullyConnected

Layer 1 (Convolutional): The output shape should be 28x28x6.

Activation. Your choice of activation function.

Pooling. The output shape should be 14x14x6.

Layer 2 (Convolutional): The output shape should be 10x10x16.

Activation. Your choice of activation function.

Pooling. The output shape should be 5x5x16.

Flattening: Flatten the output shape of the final pooling layer such that it's 1D instead of 3D.

Layer 3 (Fully Connected): This should have 120 outputs.

Activation. Your choice of activation function.

Layer 4 (Fully Connected): This should have 84 outputs.

Activation. Your choice of activation function.

Layer 5 (Fully Connected): This should have 10 outputs.


---

### Model Training and Evaluation

For **model 3** we have to put the following comand:
```
python app.py train -m model_3 -d images/train
```
The output will be the accuracy of model 3:
```
Training LaNet network...
EPOCH 0 Validation Accuracy :4.602510460251046
EPOCH 10 Validation Accuracy :37.23849351186633
EPOCH 20 Validation Accuracy :68.20083716923223
EPOCH 30 Validation Accuracy :76.56903795617394
EPOCH 40 Validation Accuracy :80.75313827482726
EPOCH 50 Validation Accuracy :91.21338922109564
EPOCH 60 Validation Accuracy :94.5606695059453
EPOCH 70 Validation Accuracy :87.86610878661088
EPOCH 80 Validation Accuracy :93.7238494222633
EPOCH 90 Validation Accuracy :93.30543943030067
EPOCH 100 Validation Accuracy :94.14225951398268
EPOCH 110 Validation Accuracy :94.14225951398268
EPOCH 120 Validation Accuracy :94.56066955582368
EPOCH 130 Validation Accuracy :94.56066955582368
EPOCH 140 Validation Accuracy :94.97907959766468
EPOCH 150 Validation Accuracy :94.97907959766468
EPOCH 160 Validation Accuracy :94.97907959766468
EPOCH 170 Validation Accuracy :94.97907959766468
EPOCH 180 Validation Accuracy :94.97907959766468
EPOCH 190 Validation Accuracy :94.97907959766468
EPOCH 199 Validation Accuracy :94.97907959766468
Train accuracy of model 3 is: 94.97907959766468
Model saved in models/model3/saved/LeNet_tF.ckpt
```
We've been able to reach a maximum accuracy of **94.97%** on the validation set.

---

### Testing the Model using the Test Set

For **model 3** we have to put the following comand:
```
python app.py test -m model_3 -d images/test
```
The output will be the accuracy of model 3:
```
Model restored
Test accuraccy of model 3 is: 96.3455149501661
```
We've been able to reach a maximum accuracy of **96.34%** on the test set.

