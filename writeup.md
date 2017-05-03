#**Behavioral Cloning** 

##Deep learning Behavioral Cloning Project Submission for Udacity Self Driving Car Nanodegree
##by. Reinaldo Maslim

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./centre.jpg 
[image2]: ./left.jpg
[image3]: ./right.jpg
[image4]: ./centre_flipped.jpg

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64 (model.py lines 84-97) 
Each of these convolutions are followed with RELU activation.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 105, 111). The model is trained and validated with separate datasets with ratio of 4:1 respectively. The input image has also been normalized about 0.

####3. Model parameter tuning

The model is trained with adam optimizer with mean square error loss. The model is trained in batches of 35 training data and 10 epochs. The loss graph shows that both training error and validation error decreases each epoch to lowest of around 0.03. However, beyond 10 epochs the validation error does not improve any further. 

####4. Appropriate training data

Each training data includes an image from the car labelled with the steering angle input by manual driver. 
For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use similar model by Nvidia. Their model is proven to work hence a good model to try with.

My first step was to use a few layers of convolution neural networks, followed with fully connected layers. I think this model is appropriate because the cnns removes position dependency in which it makes stronger assumptions by locality assumption. After that, the cnn depth's layer is flattened. Then fully connected layers in reducing size down to 1 will give steering value from regression. Each fully connected layers are also followed with relu activation except the last one because we are interested in the regression value instead of classification.

To combat the overfitting, I modified the model so that it has some dropout layers and normalize the training set. 

The final step was to run the simulator to see how well the car was driving around track one. Initially the car come out of the track especially on the bridges. To provide better training data, i use mouse or joystick to reduce steering input's resolution. 

After sufficient laps were recorded, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 79-118) consisted of a convolution neural network with the following layers and layer sizes:

input: 90x320x3 

1. 	Convolution 1: output depth=24, kernel size=5x5
   	Relu activation layer
2. 	Convolution 2: output depth=36, kernel size=5x5
	Relu activation layer
3. 	Convolution 3: output depth=48, kernel size=5x5
	Relu activation layer
4. 	Convolution 4: output depth=64, kernel size=3x3
	Relu activation layer
5. 	Convolution 5: output depth=128, kernel size=3x3
	Relu activation layer
6. 	flatten to 1D array
7. 	Fully connected layer 1: output 50
	Relu activation layer
	Dropout
9. 	Fully connected layer 2: output 10
	Relu activation layer
	Dropout
11. Fully connected layer 3: output 1 



####3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded more than five laps of centre driving. Here is an example image of center lane driving:

![alt text][image1]

First, I crop the input image to remove the sky section. Then, the image is normalized to range of -0.5 to 0.5. However, I kept all three color channels without grayscaling because 

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when it goes out of the lane.
To get more training data, I decided to use the left and right camera images. These images are labelled with a fixed offset from steering angle. These left, centre, and right images are shown below respectively:

![alt text][image1]
![alt text][image2]
![alt text][image3]

To augment the data set, I also flipped images and angles thinking that this would balance right and left steer bias. For example, here is an image that has then been flipped:

![alt text][image1]
![alt text][image4]


After the collection process, I have 6 data points per frame which gives a total of 240 000 data points. Using Nvidia GeForce GTX 1050 GPU, the training takes 45 minutes to complete. 
From the simulation as recorded in the video, the car can drive autonomously pretty well in the first circuit. It can pass the bridge and recover if it is off the track.  


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
