import click
import urllib.request
import zipfile
import cv2
import csv
import numpy as np

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf
    
from matplotlib import cm
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
import os

import skimage.morphology as morp
from skimage.filters import rank
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pickle
from tensorflow.contrib.layers import flatten


import time

global labels
labels = []
global model
model = ''
global path
path = ''
global path1
path1 = ''
# Create global variables for train model    
global classes
classes = np.arange(0, 43)
global X    # data
X = []  
global X_train  # images for train model
X_train = []
global y_train  # labels for train data
y_train = []
global X_test   # images for test model
X_test = []
global y_test   # label for test data
y_test = []
global X_validation   # images for test model
X_validation = []
global y_validation   # label for validation data
y_validation = []
global state
state = 0       # 0=Train 1=Test 2=Validation
global images_ext
images_ext = []
global images_ext2
images_ext2 = []

global count_images_ext
count_images_ext = 0
global label_images_ext
label_images_ext = []
global label_images_ext2
label_images_ext2 = []
global new_img
new_img = []
global infer_bool 
infer_bool = False
global plot
plot = False
global X_user  # images for train model
X_user = []
    


@click.group()

def cli():    
    pass

@click.option('-p', default=False, help="True to plot samples of augmented dataset")

@cli.command()
def download(p):
    '''
    Download the GTSDB dataset
    '''
    global plot
    plot = bool(p)
    click.echo('Downloading dataset')    
    
    # URL of the Zip file for train images
    click.echo('Downloading train images and test images... (This process can take several minutes)')
    data_zip=urllib.request.urlretrieve("http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip","images/data.zip")    

    click.echo('Download complete!')
    click.echo('Extracting files')
    with zipfile.ZipFile('images/data.zip',"r") as z:
                   z.extractall("images/")

    os.rename(os.path.join("images/", 'FullIJCNN2013'), os.path.join("images/", 'data'))
        
    size_img = (32,32)
    count_train = 0
    count_test = 0
    num_files = []
    augment = False
    # Load the train data and assign to train array
    with open('images/test/test_file.csv', 'w', newline='') as out_file:
        writer = csv.writer(out_file,delimiter=';', quotechar='|')
        with open('images/train/train_file.csv', 'w', newline='') as out_file:
            writer1 = csv.writer(out_file,delimiter=';', quotechar='|')
            click.echo('Saving files on train and test folder')                                
            for c in range(len(classes)):
                mypath = 'images/data' + '/'+ format(classes[c], '02d') + '/'
                path0 = 'images/train' + '/'
                files = [f for f in listdir(mypath) if isfile(join(mypath, f))]                         
                #print(files, len(files))
                if len(files)==2:
                    split = 2
                else:
                    split = int(len(files)*0.82)
                if len(files) <= 15:                    
                    augment = True
                  
                for images_train in range(0,split):                
                    im = cv2.imread(mypath + str(files[images_train]),0) 
                    im = cv2.resize(im, size_img)
                    # Save the crop image to the train folder                
                    cv2.imwrite('images/train/' + str(format(count_train, '04d'))+'.ppm',im)                    
                    lines = [str(format(count_train, '04d'))+'.ppm',str(c)]                
                    writer1.writerow(lines)                
                    count_train += 1
                    if augment==True:
                        img_extend = extend_images(im,c,1)                       
                        cv2.imwrite('images/train/' + str(format(count_train, '04d'))+'.ppm',img_extend)                    
                        lines = [str(format(count_train, '04d'))+'.ppm',str(c)]                
                        writer1.writerow(lines)                
                        count_train += 1
                        img_extend = extend_images(im,c,0)                       
                        cv2.imwrite('images/train/' + str(format(count_train, '04d'))+'.ppm',img_extend)                    
                        lines = [str(format(count_train, '04d'))+'.ppm',str(c)]                
                        writer1.writerow(lines)                
                        count_train += 1
                   
                for images_test in range(split,len(files)):                                                                            
                    im1 = cv2.imread(mypath + str(files[images_test]),0)                
                    im = cv2.resize(im, (32,32))                    
                    # Save the crop image to the test folder                
                    cv2.imwrite('images/test/' + str(format(count_test, '03d'))+'.ppm',im)                    
                    lines = [str(format(count_test, '03d'))+'.ppm',str(c)]                    
                    writer.writerow(lines)                
                    count_test += 1
                    if augment==True:
                        img_extend = extend_images(im,c,1)                       
                        cv2.imwrite('images/test/' + str(format(count_test, '03d'))+'.ppm',img_extend)                    
                        lines = [str(format(count_test, '03d'))+'.ppm',str(c)]                
                        writer.writerow(lines)                
                        count_test += 1
                        img_extend = extend_images(im,c,0)                       
                        cv2.imwrite('images/test/' + str(format(count_test, '03d'))+'.ppm',img_extend)                    
                        lines = [str(format(count_test, '03d'))+'.ppm',str(c)]                
                        writer.writerow(lines)                
                        count_test += 1
                        
                augment = False
            click.echo("Train folder: "+str(count_train)+"  Test folder: "+ str(count_test))
              
def extend_images(img, label_ext,num_create,angle=15, pixels=2,gamma=2):

    if num_create==1: 
        global new_img    
        new_img.append(img)
    
    hist,bins = np.histogram(img.ravel(),256,[0,256])    
    hist_avg = np.argmax(hist)

    # Augment borders of image
    top = int(0.1 * img.shape[0])  # shape[0] = rows
    bottom = top
    left = int(0.1 * img.shape[1])  # shape[1] = cols
    right = left    
    dst = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE, None)
    resized_image = cv2.resize(dst, (80, 80))

    # Rotate
    if num_create==1: angle=14
    else: angle = -10
    M = cv2.getRotationMatrix2D((16, 16), angle, 1)
    image1 = cv2.warpAffine(src=resized_image, M=M, dsize=(80, 80))

    # Translate
    tx = 1
    if num_create==1: ty = -1
    else: ty = 1    
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    image2 = cv2.warpAffine(src=image1, M=M, dsize=(80, 80))

    # Bright
    if np.argmax(hist)>=190: gamma = 0.6180
    else: gamma = 1.618033
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
    image3 = cv2.LUT(image2, table)

    # Crop image and resize 
    if num_create==1:
        image_new = image3[0:65,15:]
    else:
        image_new = image3[12:78,2:72]
        
    resized_image = cv2.resize(image_new, (32, 32))

    global count_images_ext
    count_images_ext += 1
    if num_create==1:
        global images_ext
        images_ext.append(resized_image)
        global label_images_ext
        label_images_ext.append(label_ext)        
    else:
        global images_ext2
        images_ext2.append(resized_image)
        global label_images_ext2
        label_images_ext2.append(label_ext)    
    
    if count_images_ext % 10 == 0 and plot==True:        
        # Plot samples
        num = int((count_images_ext / 10))-1
        sp = 1
        fig = plt.figure(figsize=(4,8))
        fig.suptitle("Normal img - Create img", fontsize=14)
        for r in range(5):
            for c in range(3):                
                ax = plt.subplot(5, 3, sp)
                img_ext_1=new_img[r+(5*num)]
                img_ext_2=images_ext[r+(5*num)]
                img_ext_3=images_ext2[r+(5*num)]
                if c==0:
                    ax.imshow(img_ext_1, cmap=cm.Greys_r)
                elif c==1:
                    ax.imshow(img_ext_2, cmap=cm.Greys_r)
                else:
                    ax.imshow(img_ext_3, cmap=cm.Greys_r)                    
                ax.axis('off')
                #ax.set(xlabel='Hola', ylabel=str(label_images_ext[r]))
                #ax.set_title("Class: "+str(label_images_ext[r]),fontsize=10)
                sp += 1
        
        plt.show()
    return resized_image
            

    

@click.option('-p', default=False, help="True to plot samples of images during the preprocess")
@click.option('-d', default="images/train", help="Here you choose the path of directory")
@click.option('-m', default="model_1", help="Here you choose the model (model_1, model_2, model_3)")

@cli.command()
def train(m,d,p):
    '''
    Train model 
    '''

    global model
    model = str(m)
    global path
    path = str(d)
    global plot
    plot = bool(p)
    
    if model=='model_1':
        click.echo("Model is logistic regression with scikit-learn")
        global state
        state = 0
        model1(test=False)
    elif model=='model_2':
        click.echo("Model is logistic regression with tensorflow")        
        state = 0
        model2(test=False)
    elif model=='model_3':
        click.echo("Model is LeNet with tensorflow")
        model3(test=False)     
    else:
        click.echo("The choosen model is incorrect")

@click.option('-p', default=False, help="True to plot samples of images during the preprocess")
@click.option('-d', default="images/test", help="Here you choose the path of directory")
@click.option('-m', default="model_1", help="Here you choose the model (model_1, model_2, model_3)")

@cli.command()
def test(m,d,p):
    '''
    Test model 
    '''
    global model
    model = str(m)
    global path
    path = str(d)
    global plot
    plot = bool(p)

    if model=='model_1':
        click.echo("Model is logistic regression with scikit-learn")
        global state
        state = 2
        model1(test=True)
    elif model=='model_2':
        click.echo("Model is logistic regression with tensorflow")        
        state = 2
        model2(test=True)
    elif model=='model_3':
        click.echo("Model is LeNet with tensorflow")  
        model3(test=True)  
    else:
        click.echo("the choosen model is incorrect")

@click.option('-d', default="images/user", help="Here you choose the path of directory")
@click.option('-m', default="model_1", help="Here you choose the model (model_1, model_2, model_3)")

@cli.command()
def infer(m,d):
    '''
    Infer model 
    '''
    global model
    model = str(m)
    global path1
    path1 = str(d)

    if model=='model_1':
        click.echo("Model is logistic regression with scikit-learn")
        global infer_bool
        infer_bool = True        
        model1(test=True)
        infer_bool = False 
    elif model=='model_2':
        click.echo("Model is logistic regression with tensorflow")          
        infer_bool = True       
        model2(test=True)
        infer_bool = False
    elif model=='model_3':
        click.echo("Model is LeNet with tensorflow")          
        infer_bool = True 
        model3(test=True) 
        infer_bool = False 
    else:
        click.echo("the choosen model is incorrect")
           

def model1(test=False):
    '''
    Here we manage the data set, we make the train array and the test array.
    - Load the data and assign to train array and test array
    - We preprocess the dataset (Grayscaling, Local Histogram Equalization and normalization).    
    - Train classifier.
    - Score the classifier.
    '''
    if test==False:
        load_train_data()

        global X_train        
        global X_test
        
        
        # Convert shape of input (856x32x32) to (856x1024)
        nsamples, nx, ny = X_train.shape
        X_train = X_train.reshape((nsamples,nx*ny))  #Reshape X_train: 853x32x32 to 853x1024

        nsamples, nx, ny = X_test.shape
        X_test = X_test.reshape((nsamples,nx*ny))  #Reshape X_train: 853x32x32 to 853x1024
         
        
        model_1 = LogisticRegression(C=1.0,tol=0.0001, max_iter=1000, multi_class='ovr', n_jobs=1)            
        model_1 = model_1.fit(X_train, y_train)
        # check the accuracy on the training set
        click.echo("Train accuracy of model 1 is: "+str(model_1.score(X_test, y_test)*100))
        # save the model to disk
        filename = 'models/model1/saved/log_reg_sci(model_1).p'
        pickle.dump(model_1, open(filename, 'wb'))
    else:
        
        if infer_bool == True:
            load_user_data()
            global X_user                                                   
            
            nsamples, nx, ny = X_user.shape
            X_user = X_user.reshape((nsamples,nx*ny)) 

            # load the model from folder model1/saved/
            loaded_model = pickle.load(open('models/model1/saved/log_reg_sci(model_1).p', 'rb'))    
            predicted = loaded_model.predict(X_user)
            click.echo(str(predicted))
                        
            nsamples, nx = X_user.shape
            X_user = X_user.reshape((nsamples,int(nx/32),int(nx/32)))            

            for new in range(len(X_user)):
            	plt.imshow(X_user[new], cmap=cm.Greys_r)
            	plt.axis('off')
            	plt.title("Class: "+str(dictionary(predicted[new])))
            	#plt.set_title("Class: "+str(predicted[new]),fontsize=10)    
            	plt.show()
        
        else:
            load_test_data()
            global X_validation    
            # Convert shape of input (856x32x32) to (856x1024)
            nsamples, nx, ny = X_validation.shape
            X_validation = X_validation.reshape((nsamples,nx*ny))  #Reshape X_train: 853x32x32 to 853x1024

            # load the model from folder model1/saved/
            loaded_model = pickle.load(open('models/model1/saved/log_reg_sci(model_1).p', 'rb'))    
            predicted = loaded_model.predict(X_validation)
            click.echo("Test accuraccy of model 1 is: "+str(accuracy_score(y_validation, predicted)*100))

def reshape_labels(dataset):
    y_train_converted = []
    new_label = np.zeros((len(classes)), dtype=int)
    for i in range(len(dataset)):
        new_label[dataset[i]] = 1
        y_train_converted.append(new_label)
        new_label = np.zeros((len(classes)), dtype=int)
    return np.array(y_train_converted)


def model2(test=False):
    '''
    Here we manage the data set, we make the train array and the test array.
    - Load the data and assign to train array and test array
    - We preprocess the dataset (Grayscaling, Local Histogram Equalization and normalization).    
    - Train classifier.
    - Score the classifier.
    '''
   
        
    if test==False:
        load_train_data()
        global X_train
        global y_train
        
        y_train = reshape_labels(y_train)

        global X_test
        # Convert shape of input (856x32x32) to (856x1024)
        nsamples, nx, ny = X_train.shape
        X_train = X_train.reshape((nsamples,nx*ny))  #Reshape X_train: 853x32x32 to 853x1024

        nsamples, nx, ny = X_test.shape
        X_test = X_test.reshape((nsamples,nx*ny))  #Reshape X_train: 853x32x32 to 853x1024


        # Parameters
        learning_rate = 0.05
        training_epochs = 500
        batch_size = 128
        display_step = 1
        
        # tf Graph Input
        x = tf.placeholder(tf.float32, [None, 1024]) # DSTDB data image of shape 32x32=1024
        y = tf.placeholder(tf.float32, [None, 43]) # 0-43 traffic signs => 43 classes
        
        # Set model weights
        W = tf.Variable(tf.zeros([1024, 43]))
        b = tf.Variable(tf.zeros([43]))
        
        # Construct model
        pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

        # Minimize error using cross entropy
        cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
        # Gradient Descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        tiempo_inicial = time.time()
        # Start training
        with tf.Session() as sess:
            sess.run(init)
            num_examples = len(y_train)
            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(len(y_train)/batch_size)              
                # Loop over all batches
                for offset in range(0, num_examples, batch_size):
                    end = offset + batch_size
                    batch_xs, batch_ys = X_train[offset:end],y_train[offset:end]                    
                    # Fit training using batch data
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                                  y: batch_ys})
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
                if epoch % 50 == 0 or epoch==training_epochs-1:                    
                    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    click.echo("Epoch: "+str(epoch)+ " cost=" +str(avg_cost)+" Accuracy of epoch is:"+str(accuracy.eval({x: X_test, y: reshape_labels(y_test)})))
                    
            click.echo("Optimization Finished!")

            print(X_test.shape)
            saver = tf.train.Saver()
            save_path = saver.save(sess, 'models/model2/saved/log_reg_tF.ckpt')
            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # Calculate accuracy for 210 examples
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            click.echo("Train accuracy of model 2 is:"+str( accuracy.eval({x: X_test, y: reshape_labels(y_test)})*100))
            tiempo_final = time.time()
            click.echo("Elapsed time: "+str(tiempo_final-tiempo_inicial) +" seconds")            
            click.echo("Model saved to: "+ str(save_path))
    else:
    	# Parameters
        learning_rate = 0.05
        training_epochs = 1000
        batch_size = 64
        display_step = 1
        
        # tf Graph Input
        x = tf.placeholder(tf.float32, [None, 1024]) # mnist data image of shape 28*28=784
        y = tf.placeholder(tf.float32, [None, 43]) # 0-43 traffic signs => 43 classes
        
        # Set model weights
        W = tf.Variable(tf.zeros([1024, 43]))
        b = tf.Variable(tf.zeros([43]))
        
        # Construct model
        pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
        
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy for 210 examples
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        production = tf.argmax(pred,1)

        if infer_bool == True:
            load_user_data()
            global X_user 
            nsamples, nx, ny = X_user.shape
            X_user = X_user.reshape((nsamples,nx*ny)) 

            # load the model from folder model1/saved/
            saver = tf.train.Saver()
            print(X_user.shape)
            print(X_user[0].shape)
            
            with tf.Session() as sess:
                saver.restore(sess,"models/model2/saved/log_reg_tF.ckpt")
                click.echo('Model restored')
                prediction_user = production.eval({x: X_user})
                click.echo(prediction_user)                

                nsamples, nx = X_user.shape
                X_user = X_user.reshape((nsamples,int(nx/32),int(nx/32)))   
                for new in range(len(X_user)):
                    plt.imshow(X_user[new], cmap=cm.Greys_r)
                    plt.axis('off')
                    plt.title("Class: "+str(dictionary(prediction_user[new])))                    
                    plt.show()
        else:
            load_test_data()
            global X_validation    
            # Convert shape of input 
            nsamples, nx, ny = X_validation.shape
            X_validation = X_validation.reshape((nsamples,nx*ny))  

            print(X_validation.shape)
                
            saver = tf.train.Saver()
            with tf.Session() as sess:
                saver.restore(sess,"models/model2/saved/log_reg_tF.ckpt")
                click.echo('Model restored')
                click.echo('Test accuraccy of model 2 is: '+str(accuracy.eval({x: X_validation, y: reshape_labels(y_validation)})*100))            

            
            
           
    	
        

class LaNet:  
    def __init__(self, x, y, n_out=43, mu=0, sigma=0.1, learning_rate=0.001):
        
        # Hyperparameters
        self.mu = mu
        self.sigma = sigma

        # Layer 1 (Convolutional): Input = 32x32x1. Output = 28x28x6.
        self.filter1_widht = 5
        self.filter1_height = 5
        self.input1_channels = 1
        self.conv1_output = 6
        # Weight and bias
        self.conv1_weight = tf.Variable(tf.truncated_normal(
            shape=(self.filter1_widht, self.filter1_height, 
                   self.input1_channels, self.conv1_output),
                   mean = self.mu, stddev = self.sigma))
        self.conv1_bias = tf.Variable(tf.zeros(self.conv1_output))
        # Apply Convolution
        self.conv1 = tf.nn.conv2d(x, self.conv1_weight, strides=[1, 1, 1, 1], padding='VALID') + self.conv1_bias
        
        # Activation:
        self.conv1 = tf.nn.relu(self.conv1)
        
        # Pooling: Input = 28x28x6. Output = 14x14x6.
        self.conv1 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        
        # Layer 2 (Convolutional): Output = 10x10x16.
        self.filter2_width = 5
        self.filter2_height = 5
        self.input2_channels = 6
        self.conv2_output = 16
        # Weight and bias
        self.conv2_weight = tf.Variable(tf.truncated_normal(
            shape=(self.filter2_width, self.filter2_height, self.input2_channels, self.conv2_output),
            mean = self.mu, stddev = self.sigma))
        self.conv2_bias = tf.Variable(tf.zeros(self.conv2_output))
        # Apply Convolution
        self.conv2 = tf.nn.conv2d(self.conv1, self.conv2_weight, strides=[1, 1, 1, 1], padding='VALID') + self.conv2_bias
        
        # Activation:
        self.conv2 = tf.nn.relu(self.conv2)
        
        # Pooling: Input = 10x10x16. Output = 5x5x16.
        self.conv2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        
        # Flattening: Input = 5x5x16. Output = 400.
        self.fully_connected0 = flatten(self.conv2)
        
        # Layer 3 (Fully Connected): Input = 400. Output = 120.
        self.connected1_weights = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = self.mu, stddev = self.sigma))
        self.connected1_bias = tf.Variable(tf.zeros(120))
        self.fully_connected1 = tf.add((tf.matmul(self.fully_connected0, self.connected1_weights)), self.connected1_bias)
        
        # Activation:
        self.fully_connected1 = tf.nn.relu(self.fully_connected1)
    
        # Layer 4 (Fully Connected): Input = 120. Output = 84.
        self.connected2_weights = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = self.mu, stddev = self.sigma))
        self.connected2_bias = tf.Variable(tf.zeros(84))
        self.fully_connected2 = tf.add((tf.matmul(self.fully_connected1, self.connected2_weights)), self.connected2_bias)
        
        # Activation.
        self.fully_connected2 = tf.nn.relu(self.fully_connected2)
    
        # Layer 5 (Fully Connected): Input = 84. Output = 43.
        self.output_weights = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = self.mu, stddev = self.sigma))
        self.output_bias = tf.Variable(tf.zeros(43))
        self.logits =  tf.add((tf.matmul(self.fully_connected2, self.output_weights)), self.output_bias)

        # Training operation
        self.one_hot_y = tf.one_hot(y, n_out)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.one_hot_y)
        self.loss_operation = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        self.training_operation = self.optimizer.minimize(self.loss_operation)

        # Accuracy operation
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # Saving all variables
        self.saver = tf.train.Saver()
    
    def y_predict(self,x, y, keep_prob,keep_prob_conv, X_data, BATCH_SIZE=64):
        num_examples = len(X_data)
        y_pred = np.zeros(num_examples, dtype=np.int32)
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x = X_data[offset:offset+BATCH_SIZE]
            y_pred[offset:offset+BATCH_SIZE] = sess.run(tf.argmax(self.logits, 1), 
                               feed_dict={x:batch_x, keep_prob:1, keep_prob_conv:1})
        return y_pred
    
    def evaluate(self,x,y,keep_prob,keep_prob_conv, X_data, y_data, BATCH_SIZE=64):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            accuracy = sess.run(self.accuracy_operation, 
                                feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0, keep_prob_conv: 1.0 })
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples




def model3(test=False):
    '''
    Here we manage the data set, we make the train array and the test array.
    - Load the data and assign to train array and test array
    - We preprocess the dataset (Grayscaling, Local Histogram Equalization and normalization).    
    - Train classifier.
    - Score the classifier.
    '''
    x = tf.placeholder(tf.float32, (None, 32,32,1))
    y = tf.placeholder(tf.int32, (None))
    
    keep_prob = tf.placeholder(tf.float32)       # For fully-connected layers
    keep_prob_conv = tf.placeholder(tf.float32)  # For convolutional layers                

    EPOCHS = 200
    BATCH_SIZE = 64 

    if test==False:
        load_train_data()
        global X_train
        global y_train                

        global X_test
        # Convert shape of input (856x32x32) to (856x1024)
        nsamples, nx, ny = X_train.shape
        X_train = X_train.reshape((nsamples,nx*ny))  #Reshape X_train: 853x32x32 to 853x1024

        nsamples, nx, ny = X_test.shape
        X_test = X_test.reshape((nsamples,nx*ny))  #Reshape X_train: 853x32x32 to 853x1024        

        # Parameters
        learning_rate = 0.005
        sigma = 0.1    
        n_out = 43
        mu = 0                       

        LeNet_Model = LaNet(x,y,n_out = 43)
        model_name = "LeNet"

        # Validation set preprocessing
        X_valid_preprocessed = np.reshape(X_test, (-1, 32, 32, 1))
        X_train1 = np.reshape(X_train, (-1, 32, 32, 1))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(y_train)            
            print("Training LaNet network...")            

            for i in range(EPOCHS):
                normalized_images, y_train = X_train1, y_train
                for offset in range(0, num_examples, BATCH_SIZE):
                    end = offset + BATCH_SIZE
                    batch_x, batch_y = normalized_images[offset:end], y_train[offset:end]                    
                    #print(len(batch_x),batch_x.shape,len(batch_y),batch_y.shape,offset,end)
                    sess.run(LeNet_Model.training_operation, feed_dict={x: batch_x, y:batch_y})
                    
                if i % 10 == 0 or i==EPOCHS-1:
                    validation_accuracy = LeNet_Model.evaluate(x, y, keep_prob, keep_prob_conv, X_valid_preprocessed, y_test)
                    click.echo("EPOCH "+str(i)+" Validation Accuracy :"+str(validation_accuracy*100))
            validation_accuracy = LeNet_Model.evaluate(x, y, keep_prob, keep_prob_conv, X_valid_preprocessed, y_test)
            click.echo("Train accuracy of model 3 is: "+str(validation_accuracy*100))
            
            
            saver = tf.train.Saver()
            save_path = saver.save(sess, 'models/model3/saved/LeNet_tF.ckpt')
            click.echo("Model saved in "+str(save_path))



    else:

        LeNet_Model = LaNet(x,y,n_out = 43)
        
        if infer_bool == True:
            load_user_data()
            global X_user 
            nsamples, nx, ny = X_user.shape
            X_user = X_user.reshape((nsamples,nx*ny)) 

            # load the model from folder model1/saved/
            saver = tf.train.Saver()
            print(X_user.shape)
            print(X_user[0].shape)

            X_user1 =  np.reshape(X_user, (-1, 32, 32, 1))
            
            with tf.Session() as sess:
                saver.restore(sess, 'models/model3/saved/LeNet_tF.ckpt')
                click.echo('Model restored')
                y_pred = LeNet_Model.y_predict(x, y, keep_prob, keep_prob_conv, X_user1)
                print(y_pred)
                #click.echo(prediction_user)                

                nsamples, nx = X_user.shape
                X_user = X_user.reshape((nsamples,int(nx/32),int(nx/32)))   
                for new in range(len(X_user)):
                    plt.imshow(X_user[new], cmap=cm.Greys_r)
                    plt.axis('off')
                    plt.title("Class: "+str(dictionary(y_pred[new])))             
                    plt.show()
            
        else:
            load_test_data()
            global X_validation    
            # Convert shape of input (856x32x32) to (856x1024)
            nsamples, nx, ny = X_validation.shape
            X_validation = X_validation.reshape((nsamples,nx*ny))  #Reshape X_train: 853x32x32 to 853x1024

            X_validation =  np.reshape(X_validation, (-1, 32, 32, 1))
            

            saver = tf.train.Saver()
            with tf.Session() as sess:
                saver.restore(sess, 'models/model3/saved/LeNet_tF.ckpt')
                click.echo('Model restored')
                y_pred = LeNet_Model.y_predict(x, y, keep_prob, keep_prob_conv, X_validation)
                test_accuracy = sum(y_validation == y_pred)/len(y_validation)
                click.echo("Test accuraccy of model 3 is: "+str(test_accuracy*100))



def load_train_data():
    '''
    Here we load training data
    '''
    size_img = (32,32)
    mypath = path + '/'
    first_plot = 0
    with open(mypath+'train_file.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=';')
        for row in readCSV:
            id_image = row[0]            
            im = cv2.imread(mypath+str(id_image),0)
            if plot==True and first_plot == 0:
                click.echo(str(mypath)+str(id_image)+str(row[1]))
                cv2.imshow("image",im)
                if cv2.waitKey(33) == ord('c'):
                    first_plot = 1
                    cv2.destroyAllWindows()
            global X
            X.append(im)
            global labels
            labels.append(int(row[1]))

    # Make X_train data and X_test (Test = 20%)    
    hist, bins = np.histogram(labels, bins=43)    
    id_classes = 0
    hist_sum = hist[0]
    split_train = int(hist[0]*0.852-1)
    #print(hist)
    for c in range(len(X)):        
        if c <= split_train:            
            global X_train
            global y_train
            X_train.append(X[c])            
            y_train.append(labels[c])            
        else:
            global X_test
            X_test.append(X[c])
            global y_test
            y_test.append(labels[c])
            if c==len(X)-1:
                break
            if c==hist_sum-1:                
                id_classes += 1
                hist_sum = hist[id_classes]+hist_sum
                split_train = int(hist[id_classes]*0.852)+c-1
                if hist[id_classes]==2:                    
                    split_train = int(hist[id_classes]*0.852)+c
        
    X_train = preprocess(X_train,"Train dataset")
    global state
    state = 1
    X_test = preprocess(X_test,"Test dataset")           

    X_train = np.array(X_train)         #Convert to numpy array
    y_train = np.array(y_train)         #Convert to numpy array
    X_test = np.array(X_test)           #Convert to numpy array
    y_test = np.array(y_test)           #Convert to numpy array 
    
    # Ask if plot hostogram
    if plot==True:
        plot_histogram(y_train,"Train")
        plot_histogram(y_test,"Test")


def load_user_data():
    # Open csv of the test folder and save the data to X_test and y_test
    mypath = path1+'/'
    print(mypath)
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    print(files)
    for images_infer in range(0,len(files)):            
        img_infer = cv2.imread(mypath+files[images_infer],0)
        im = cv2.resize(img_infer, (32,32))       
        global X_user  # images for train model
        X_user.append(im)       
    
    X_user = preprocess(X_user,"Validation dataset")
    X_user = np.array(X_user) 


def load_test_data():
    # Open csv of the test folder and save the data to X_test and y_test
    with open(path+'/test_file.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=';')
        for row in readCSV:          
            im = cv2.imread(path+'/'+str(row[0]),0)
            im = cv2.resize(im, (32, 32))            
            global X_validation
            X_validation.append(im)
            global y_validation
            y_validation.append(int(row[1]))

    X_validation = preprocess(X_validation,"Validation dataset")
    X_validation = np.array(X_validation)         #Convert to numpy array
    y_validation = np.array(y_validation)         #Convert to numpy array  

def plot_histogram(dataset1,text):
    # Print histogram for Train images
    histo, bins = np.histogram(dataset1, bins=43)        
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, histo, align='center', width=width)
    plt.xlabel(text+" set")
    plt.ylabel("Image count")
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    plt.show()    

def preprocess(dataset,text):    
    global sample_idx   # For plot porpose
    sample_idx = np.random.randint(len(dataset), size=18)  
    # Local Histogram Equalization 
    equalized_images = list(map(local_histo_equalize, dataset))
    if plot==True: plot_images(title=("Showing samples of equalized images -"+text),dataset=equalized_images)
    # Normalization 
    normalized_images = list(map(image_normalize, equalized_images))
    if plot==True: plot_images(title="Showing samples of normalized images -"+text,
                               dataset=normalized_images)    
    return normalized_images

def local_histo_equalize(image):
    """
    Apply local histogram equalization to grayscale images.
        Parameters:
            image: A grayscale image.
    """    
    kernel = morp.disk(30)
    img_local = rank.equalize(image, selem=kernel)
    return img_local

def image_normalize(image):
    """
    Normalize images to [0, 1] scale.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    image = np.divide(image, 255)
    return image

def plot_images(title="Showing samples",dataset=10):
    if state==0:    y_label = y_train
    if state==1:    y_label = y_test
    if state==2:    y_label = y_validation
    sp = 1
    fig = plt.figure()
    fig.suptitle(title, fontsize=16)    
    for r in range(3):
        for c in range(6):                
            ax = plt.subplot(3, 6, sp)
            sample = dataset[sample_idx[sp - 1]]            
            ax.imshow(sample, cmap=cm.Greys_r)
            ax.axis('off')
            ax.set_title(str(y_label[sample_idx[sp - 1]]))
            sp += 1
    
    plt.show()


def dictionary(num):

	dict= ['speed limit 20 (prohibitory)','speed limit 30 (prohibitory)','speed limit 50 (prohibitory)','speed limit 60 (prohibitory)','speed limit 70 (prohibitory)','speed limit 80 (prohibitory)','restriction ends 80 (other)','speed limit 100 (prohibitory)','speed limit 120 (prohibitory)','no overtaking (prohibitory)','no overtaking (trucks) (prohibitory)',
	       'priority at next intersection (danger)','priority road (other)','give way (other)','stop (other)','no traffic both ways (prohibitory)','no trucks (prohibitory)','no entry (other)','danger (danger)','bend left (danger)','bend right (danger)','bend (danger)','uneven road (danger)','slippery road (danger)','road narrows (danger)',
	       'construction (danger)','traffic signal (danger)','pedestrian crossing (danger)','school crossing (danger)','cycles crossing (danger)','snow (danger)','animals (danger)','restriction ends (other)','go right (mandatory)','go left (mandatory)','go straight (mandatory)','go right or straight (mandatory)','go left or straight (mandatory)',
	       'keep right (mandatory)','keep left (mandatory)','roundabout (mandatory)','restriction ends (overtaking) (other)','restriction ends']

	return dict[num]



if __name__ == '__main__':
    cli()
