---
author: Euan Wielewski
date: 2019-01-24 18:31:13 +0000
excerpt: Use TensorFlow to build your own haggis-hunting app for Burns Night! The
  Scottish quest for the mythical wild haggis just got easier with deep learning.
feature_image: /assets/images/hero/haggis-not-haggis-hero.jpeg
layout: post
slug: haggis-not-haggis
tags: '[]'
title: 'Haggis, Not Haggis: How to build a haggis detection app with TensorFlow, Keras,
  and FloydHub for Burns Night'
---

## An address to a Haggis

Every year on the 25th of January, people around the world gather together to celebrate the life and poetry of Scotland’s national poet, Robert Burns. A typical Burns Night consists of friends and family, poetry readings, tartan clothing, and a lot of whisky. But the highlight of any reputable Burns Night is the [address to the haggis](https://en.wikipedia.org/wiki/Burns_supper#Address_to_a_Haggis), where that most-reviled of Scottish foods is brought into the dining room on a gleaming silver platter and sacrificed in the name of Robert Burns.

Accompanied by a proud bagpiper, the haggis receives a rousing toast (usually from a rather drunk uncle) before succumbing to its fate of feeding ten to twenty hungry Scots. There’s pomp, there’s circumstance, and it’s all a bit ridiculous — but I guarantee you will have a great time at a Burns Night!

Those of you who weren’t born and raised in Scotland probably have a question at this point. 

> What the heck is a haggis? 

This is a simple question with no simple answer. Haggis is different things to different people, but hopefully by the end of this article you will at least be able to recognize one.

To make your quest to find a haggis a little easier, I will take you through how you can build, train, and deploy a haggis detection app on [FloydHub](https://www.floydhub.com). We’ll train a state-of-the-art deep learning image recognition system to detect a haggis using Keras and TensorFlow and deploy it to a simple Flask web app.

Happy haggis hunting! 

## What does a haggis look like?

It is extremely difficult to spot a haggis in the wild. 

To give ourselves more of fighting chance, our haggis detection app will focus on the prepared haggis that is commonly found in a butchers or supermarket. Just for reference, I have included a picture of a rare wild haggis from the [Kelvingrove Art Gallery and Museum](https://www.glasgowlife.org.uk/museums/venues/kelvingrove-art-gallery-and-museum) in Glasgow.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_912F51EA41A425AC38654E6546E69EB61C2C095B73A5241EB6365BC768CCAEF1_1547984914377_haggis-Kelvingrove.jpg)Mythical wild haggis at the Kelvingrove Art Gallery and Museum in Glasgow

If we are going to build and train a deep learning haggis recognition model, then we will need a reasonable dataset of images. Luckily for you, I have spent the last two weekends taking images of haggis and random objects to give us a dataset to work with.

I’ve uploaded the images to FloydHub as a public dataset, which you can find [here](https://www.floydhub.com/euanwielewski/datasets/haggis-dataset).

The dataset consists of 100 images of haggis and 100 images of random objects (i.e. not haggis). The images are in jpeg format with an image resolution of 2160x2160 pixels. I have split the data into training (80%) and validation (20%) sets, following the standard Keras image classification directory structure outlined below:
    
    
    haggis-dataset
    |-- train
    |        |-- haggis
    |        |        |-- haggis_001.jpg
    |        |        |-- haggis_002.jpg
    |        |        |-- haggis_003.jpg
    |        |        |-- ...
    |        |-- not-haggis
    |                |-- not_haggis_001.jpg
    |                |-- not_haggis_002.jpg
    |                |-- not_haggis_003.jpg
    |                |-- ...
    |-- validation
            |-- haggis
            |        |-- haggis_001.jpg
            |        |-- haggis_002.jpg
            |        |-- haggis_003.jpg
            |        |-- ...
            |-- not-haggis
                    |-- not_haggis_001.jpg
                    |-- not_haggis_002.jpg
                    |-- not_haggis_003.jpg
                    |-- ...
    

To explore the dataset, you can either download it locally or spin up a [FloydHub Workspace](https://www.floydhub.com/product/build) and mount the dataset. Click this button if you’d like to open it up now:

[![Run on FloydHub](https://static.floydhub.com/button/button.svg)](https://floydhub.com/run?template=https://github.com/euanwielewski/floydhub-haggis-detector)

Here are some images from the dataset to give you a feel for the data and see what a real haggis looks like.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_912F51EA41A425AC38654E6546E69EB61C2C095B73A5241EB6365BC768CCAEF1_1548084652239_haggis-images.png)Example images of haggis from dataset![](https://d2mxuefqeaa7sj.cloudfront.net/s_912F51EA41A425AC38654E6546E69EB61C2C095B73A5241EB6365BC768CCAEF1_1548084686967_not-haggis-images.png)Example images of "not haggis" from dataset

## Training a haggis recognition model with Keras and TensorFlow on FloydHub

FloydHub is a really flexible platform that allows you to build and train deep learning models in a number of different ways. FloydHub Workspaces are a great tool for exploring data and iterating on your deep learning models. Since we know what our dataset looks like and don’t need to do any data exploration or preparation, we will be writing a training job script and using the Floyd CLI to submit our training jobs.

You can find the training job script on [GitHub](https://github.com/euanwielewski/floydhub-haggis-detector).

### Installing the Floyd CLI

The Floyd CLI (`floyd-cli`) is a Python-based command line tool that allows you to submit and monitor training jobs on the FloydHub servers from your terminal. It works across Windows, MacOS and Linux, and can be installed using `conda`:

`$ conda install -y -c conda-forge -c floydhub floyd-cli`

Or `pip`:

`$ pip install -U floyd-cli`

Using the `floyd-cli` makes it super easy to write scripts to conduct hyperparameter searches and rapidly test different model architectures.

### Initializing and uploading your own dataset

I’ve already uploaded and made public the dataset we will be using in this article, but for completeness, I thought I would show you how you can upload your own dataset to FloydHub using the `floyd-cli`.

The first thing to do is to create a `dataset` directory on your local machine that contains all the data you want to upload to FloydHub. To initialise a new dataset, navigate to your `dataset` directory in your terminal and type the following command:

`$ floyd data init <dataset_name>`

Replacing `<dataset_name>` with the name of of your own dataset. If this is the first time you’ve used the `floyd-cli` then you will be asked to login via the FloydHub website.

To upload your dataset to FloydHub, you use the `upload` command from your dataset directory:

`$ floyd data upload`

The `floyd-cli` compresses the data, uploads it and gives it a version number. This is super handy, since it allows you to track what version of your data you used to train a given model.

### Initializing a new project

Before we can submit jobs to the FloydHub servers, we first need to create a new project and initialize a directory for that project. In our terminal, we create a directory that we will use for our training scripts, navigate to it and then run the `init` command with the project name:
    
    
    $ mkdir haggis-model-training
    $ cd haggis-model-training
    $ floyd init haggis-training-project
    Project "haggis-training-project" initialized in current directory

If you haven’t already created the project, `floyd-cli` will automatically open the “Create Project” page in your web browser and give you the option to create the project.

### Building your model and writing a training job script

Now that we have the project setup, we can write our training job script and submit it for training. We create the training job file `training.py` and start writing the script by importing the libraries we will need to run the job:
    
    
    from tensorflow import test
    from keras.applications import vgg16
    from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
    from keras.optimizers import Adam, SGD, RMSprop
    from keras.models import Model, Input
    from keras.preprocessing.image import ImageDataGenerator

We then define the directories for our training and validation datasets, which is where we will mount our data when we use the `floyd-cli` tool to run the training job:
    
    
    # Data filepaths
    TRAIN_DATA_DIR = '/data/train/' # Point to directory with your data
    VALID_DATA_DIR = '/data/validation/'

Followed by some key model parameters:
    
    
    # Model parameters
    NUM_CLASSES = 2
    INPUT_SIZE = 224 # Width/height of image in pixels
    LEARNING_RATE = 0.0001

I find it really useful to add a simple check to see if a GPU is available, with appropriately defined batch sizes and number of epochs for if you are training on a CPU or GPU. I often submit small jobs to a CPU server (see flags in next section) to check that my script is working properly before submitting a big GPU job. Having this check saves a lot of time and avoids accidentally submitting a huge training job to a CPU server:
    
    
    # For GPU training - script checks if GPU is available
    BATCH_SIZE_GPU = 32 # Number of images used in each iteration
    EPOCHS_GPU = 50 # Number of passes through entire dataset
    
    # For CPU training
    BATCH_SIZE_CPU = 4
    EPOCHS_CPU = 1
    
    if test.is_gpu_available(): # Check if GPU is available
        BATCH_SIZE = BATCH_SIZE_GPU # GPU
        EPOCHS = EPOCHS_GPU
    
    else:
        BATCH_SIZE = BATCH_SIZE_CPU # CPU
        EPOCHS = EPOCHS_CPU

Next, we need to create `keras.ImageDataGenerator` objects for both the training and validation datasets. Since we have a small number of images, we need to augment them and generate more examples for the model to learn from. We only augment the training images, with a range of different transformations and distortions, then set the training and validation data generators to take images from the relevant directories:
    
    
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=45,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.25,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    
    valid_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
            TRAIN_DATA_DIR,
            target_size=(INPUT_SIZE, INPUT_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical')
    
    valid_generator = valid_datagen.flow_from_directory(
            VALID_DATA_DIR,
            target_size=(INPUT_SIZE, INPUT_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical')

Finally, we get to the meat. Building the deep convolutional neural network to classify our images as haggis or not haggis!

Again, since we have a limited amount of data, we will need to be clever about how we build and train the model. One way we can significantly speedup training and get great results quickly is to use transfer learning. In transfer learning, we take the weights from a network pre-trained on a large dataset and use that as the starting point for our model. In our case, we aren’t even going retrain those model weights, but instead use the pre-trained model to get bottleneck features from our images and then just train the final classification layers.

For the base of our model, we will use a [VGG16 architecture](https://arxiv.org/abs/1409.1556), that was pre-trained on the [ImageNet](https://en.wikipedia.org/wiki/ImageNet) dataset, and remove the final classification layer. We add a global average pooling layer, followed by a fully-connected layer with a ReLU activation function and then a logistic layer with a sigmoid activation function (because we have a binary classification problem).
    
    
    # Download pretrained VGG16 model and create model for transfer learning
    base_model = vgg16.VGG16(weights='imagenet', include_top=False)
    
    # Add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Add a fully-connected layer
    x = Dropout(0.2)(x)
    x = Dense(1024, activation='relu')(x)
    
    # Add a logistic layer
    x = BatchNormalization()(x)
    predictions = Dense(NUM_CLASSES, activation='sigmoid')(x)
    
    # Model for training
    model = Model(inputs=base_model.input, outputs=predictions)

Since the images we will be training our model with are very similar to those used to train the pre-trained model (i.e. ImageNet), we won’t need to retrain the base layers of the model and can freeze them. Freezing the layers means that the model weights for these layers won’t be updated during training.
    
    
    # Freeze all convolutional pretrained model layers - train only top layers
    for layer in base_model.layers:
        layer.trainable = False

We define an optimizer and provide it with the initial learning rate we defined earlier. In this case we will use the `Adam` optimizer because it uses an adaptive learning rate and is really efficient:

`optimizer = Adam(lr=LEARNING_RATE)`

We then compile the model, inputting the optimizer, the loss function and the metrics we want to record during training. Since we we have defined the problem as a binary classification task, we set the loss function as `binary_crossentropy`.
    
    
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=["accuracy"])
    model.summary()

Once the model is complied, we can then train it:
    
    
    STEP_SIZE_TRAIN=train_generator.n //train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n //valid_generator.batch_size
    model.fit_generator(train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        epochs=EPOCHS,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        verbose=2)

And once it has finished training, lets not forget to save it!

`model.save('model.h5')`

### Submitting a training job to FloydHub

Now that we have our dataset on FloydHub, initialized a project and written our training script, we are finally in a position to submit our training job. To do that, we simply call the run command with the `floyd-cli` in the initialized project directory in our terminal and add the relevant flags:
    
    
    $ floyd run --env tensorflow-1.12 --gpu --data euanwielewski/datasets/haggis-dataset/1:data "python training.py"

Since we are using Keras and TensorFlow, we use the `--env` flag followed by `tensorflow-1.12` to instruct FloydHub to spin up a server using a TensorFlow v1.12 image. We want to train our model on a GPU, so use the `--gpu` flag and we point FloydHub to the dataset we want to mount and where we want it mounted with `--data euanwielewski/datasets/haggis-dataset/1:data`.

We also provide the `floyd-cli` with the commands we want it to run once the server is spun up i.e. `"python training.py"`, which will run the training script we defined in the last section.

The `floyd-cli` uploads everything in the initialized project directory to the FloydHub servers, but if you have anything you don’t want uploaded, you can add it to the `.floydignore` file created during initialization.

And that’s it, we are training a deep convolutional neural network on a GPU in the cloud!

As a helpful aside, you can also create a `floyd.yml`[ config file ](https://docs.floydhub.com/floyd_config/)in your directory, and specify these fields in that config file. This will make it easier if you decide to run multiple jobs. Here’s what a `floyd.yml` config file looks like for our same job:
    
    
    env: tensorflow-1.12
    machine: gpu
    input:
      - source: euanwielewski/datasets/haggis-dataset/1
        destination: data
    command: python training.py

Now, to run this job, all you need to enter at the command line is:

`$ floyd run`

### Monitoring your training job

A really cool feature of FloydHub is that it easily allows you to monitor the system usage stats and the metrics of your training job. 

Just click on the URL link to your job (provided when you submit via the `floyd-cli`) and you will get an overview of useful things like model training and validation accuracy, and GPU memory usage.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_912F51EA41A425AC38654E6546E69EB61C2C095B73A5241EB6365BC768CCAEF1_1548087566236_training-accuracy-loss-25steps.png)Example training metrics from a FloydHub training job

After the designated number of epochs, our training job is complete. In case you want to go grab a coffee while you wait for your job to run, FloydHub handily sends you an email (or a [Slack notification](https://docs.floydhub.com/guides/notifications/)) to let you know when your training job is done. 

After 25 epochs, my haggis recognition model has a training accuracy of 94% and a validation accuracy of 100%, which is pretty amazing! 

Having our validation accuracy higher than our training accuracy is a little odd but this is due to the small size of the validation dataset (only 40 images). If we used a larger validation dataset the accuracy would likely drop to be closer to the training accuracy.

### Downloading your trained model

To download our model, we go to the job URL and click on the “Files” tab. FloydHub automatically saves everything in the `home` directory when the training job is finished and makes it available once the server has shutdown. Since that’s where we saved our trained model (`model.h5`), we should find it in the there. We download it and save it for later.

## Deploying a simple haggis detection app using Flask and FloydHub Serving

In addition to providing a platform to build and train deep learning models, FloydHub also allows you to host and serve machine learning models and simple web apps. We are going to use this feature to deploy our haggis detector app.

### Overview of the app

To build the haggis detection app, I used [this](https://github.com/mtobeiyf/keras-flask-deploy-webapp) excellent Keras web app as a starting point and updated it for our needs. You can find the FloydHub Haggis Detector web app repo [here](https://github.com/euanwielewski/floydhub-haggis-detector). To get started, just clone the repo locally.The app uses Bootstrap 4.0 and can easily be customised to create your own detector app!

### Serving a Keras model for inference

We don’t have space to go into the details of how the Flask app works (plus it isn’t really the thing we are interested in!) but in short, it’s basically a simple API server for Python that will allow us to serve our model with a nice user interface. 

However, it is worth spending some time going over how the model we trained is served. We do all the interesting stuff in `app.py`. The first thing we do is load the Keras modules we will need:
    
    
    import os
    import numpy as np
    
    # Keras
    from keras.applications.imagenet_utils import preprocess_input
    from keras.models import load_model
    from keras.preprocessing import image

And define a path to the model we trained in the previous section:
    
    
    # Path to Keras model
    model_file = 'model.h5'
    basepath = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(basepath, 'models', model_file)

In our simple Flask app, we store our Keras models in the `/models` directory and you can just copy the model we trained earlier into your local `/models` directory. 

We then load our trained model and create a prediction function:
    
    
    # Load your trained model and create prediction function
    model = load_model(model_path)
    model._make_predict_function()

If you are creating your own app that loads Keras models, be sure to keep model loading outside of the prediction route. Otherwise the model will be loaded every time a prediction is requested!

To serve a prediction we create a `/predict` route that loads an image, converts it into the right format and calls the `model.predict()` function:
    
    
    f = request.files['file']
    
    img_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
    f.save(img_path)
    
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = model.predict(x)

Finally, we return the prediction in a format to be displayed and that’s it, we have an app for serving a Keras model!

### Deploying the app

To deploy our app on FloydHub, we navigate to the `/haggis-detector-webapp` directory and initialize a new project:
    
    
    $ floyd init haggis-detector-serving
    Project "haggis-detector-serving" initialized in current directory

In the `/haggis-detector-webapp` directory, you will find a `floyd.yml` file that defines the environment and server that will spun up to serve our app:
    
    
    env: tensorflow-1.12
    machine: cpu

To deploy, we simply type the following `floyd-cli` command:

`$ floyd run --mode serve`

### The final result

Now our FloydHub Haggis Detector app is deployed, lets go to the serving URL on a mobile phone and test it out:

![](https://d2mxuefqeaa7sj.cloudfront.net/s_912F51EA41A425AC38654E6546E69EB61C2C095B73A5241EB6365BC768CCAEF1_1548154075755_haggis-detector-demo.gif)FloydHub Haggis Detector app in action!

## The hunt continues

I’ve taken you through how you can use FloydHub to train and deploy a haggis detection app that uses a state-of-the-art image recognition model. Hopefully this post gets you started with how tools like FloydHub can be used to easily train deep learning models on a GPU in the cloud by abstracting away the complexity of managing infrastructure for deep learning.

You now have a fully-functioning haggis detection app that you can use to hunt for a haggis. This is an amazing moment in the history of haggis-hunting — never before has a haggis-hunter been armed with such amazing deep learning technology.

I wish you good luck in your quest. Have a great Burns Night!