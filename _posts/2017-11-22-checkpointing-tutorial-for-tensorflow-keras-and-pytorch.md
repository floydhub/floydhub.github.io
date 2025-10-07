---
author: Alessio Gozzoli
date: 2017-11-22 05:48:00 +0000
excerpt: ' Ready to build, train, and deploy AI? Get started with FloydHub''s collaborative
  AI platform for free Try FloydHub for free [https://www.floydhub.com/?utm_source=blog&utm_medium=banner-checkpointing-s...'
feature_image: /assets/images/hero/checkpointing-tutorial-for-tensorflow-keras-and-pytorch-hero.jpg
layout: post
slug: checkpointing-tutorial-for-tensorflow-keras-and-pytorch
tags: [machine-learning]
title: Checkpointing Tutorial for TensorFlow, Keras, and PyTorch
---

### Ready to build, train, and deploy AI?

#### Get started with FloydHub's collaborative AI platform for free

###### [Try FloydHub for free ](https://www.floydhub.com/?utm_source=blog&utm_medium=banner-checkpointing-strategy&utm_campaign=try_floydhub_for_free)

This post will demonstrate how to checkpoint your training models on FloydHub so that you can resume your experiments from these saved states.

## Wait, but why?

![save](/assets/images/content/images/2018/04/save.png)

If you've ever played a video game, you might already understand why checkpoints are useful. For example, sometimes you'll want to save your game right before a big boss castle - just in case everything goes terribly wrong inside and you need to try again. Checkpoints in machine learning and deep learning experiments are essentially the same thing - a way to save the current state of your experiment so that you can pick up from where you left off.

Trust me, you're going to have a bad time if you lose one or more of your experiments due to a power outage, OS fault, job preemption, or any other type of unexpected error. Other times, even if you don't experience an unforeseen error, you might just want just to resume a particular state of the training for a new experiment - or try different things from a given state.

That's why you need checkpoints!

But, wait - there's one more reason, and it's a big one. If you don't checkpoint your training models at the end of a job, you'll have lost all of your results! Like, they're just **gone**. Simply put, if you'd like to make use of your trained models, you're going to need some checkpoints.

## So what is a checkpoint really?

The [Keras docs](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) provide a great explanation of checkpoints (that I'm going to gratuitously leverage here):

  * The architecture of the model, allowing you to re-create the model
  * The weights of the model
  * The training configuration (loss, optimizer, epochs, and other meta-information)
  * The state of the optimizer, allowing to resume training exactly where you left off.

Again, a checkpoint contains the information you need to save your current experiment state so that you can resume training from this point. Just like in that infernal _Zelda II: The Adventure of Link_ game from my childhood.

![zelda2](/assets/images/content/images/2018/04/zelda2.png)

# Checkpoint Strategies

At this point, I'll assume I've convinced you that checkpoints need to be a vital part of your deep learning workflow. So, let's talk strategy.

You can employ different checkpoint strategies according to the type of experiment training regime you're performing:

  * Short Training Regime (minutes to hours)
  * Normal Training Regime (hours to day)
  * Long Training Regime (days to weeks)

## Short Training Regime

The typical practice is to save a checkpoint only at the end of the training, or at the end of every epoch.

## Normal Training Regime

In this case, it's common to save multiple checkpoints every `n_epochs` and keep track of the _best_ one with respect to some validation metric that we care about. Usually, there's a fixed maximum number of checkpoints so as to not take up too much disk space (for example, restricting your maximum number of checkpoints to 10, where the new ones will replace the earliest ones).

## Long Training Regime

In this type of training regime, you'll likely want to employ a similar strategy to the _Normal_ regime - where you're saving multiple checkpoints every `n_epochs` and keeping track of the _best_ one with respect to the validation metric that you care about. In this case, since the training can be very long, it's common to save checkpoints less frequently but maintain a greater number of checkpoints.

### Which regime is right for me?

The tradeoff among these various strategies is between the **frequency** and the **number of checkpoint files** to keep. Let's take a look what's happening when we act over these two parameters:

Frequency | checkpoints | Cons | Pro  
---|---|---|---  
High | High | You need a lot of space!! | You can resume very quickly in almost all the interesting training states  
High | Low | You could have lost precious states | Minimize the storage space you need  
Low | High | It will take time to get to intermediate states | You can resume the experiments in a lot of interesting states  
Low | Low | You could have lost precious states | Minimize the storage space you need  
  
Hopefully, now you have a good intuition about what might be the best checkpoint strategy for your training regime. It should go without saying that you can obviously develop your own custom checkpoint strategy based on your experiment needs! These are just tips and best practices that I take into consideration for my own projects.

# Save and Resume on FloydHub

Now, let's dive into some code on FloydHub. I'll show you how to save checkpoints in three popular deep learning frameworks available on FloydHub: TensorFlow, Keras, and PyTorch.

Before you start, log into the FloydHub command-line-tool with the [floyd login](http://docs.floydhub.com/commands/login/) command, then fork and `init` the project:
    
    
    $ git clone https://github.com/floydhub/save-and-resume.git
    $ cd save-and-resume
    $ floyd init save-and-resume
    

For our checkpointing examples, we'll be using the `Hello, World` of deep learning: the [MNIST](http://yann.lecun.com/exdb/mnist/) classification task using a Convolutional Neural Network model.

Because it's always important to be clear about our checkpointing strategy up-front, I'll state the approach we're going to be taking:

  * Keep only one checkpoint
  * Trigger the strategy at the end of every epoch
  * Save the one with the best (maximum) validation accuracy

Considering this toy example, we can employ the Short Training Regime strategy. Feel free to adapt this for your own more complicated experiments!

### The commands

Before we dive into specific working examples, let's outline the basic commands you'll need. When starting a new job, your first command will look something like this:
    
    
    floyd run \
        [--gpu] \
        --env <env> \
        --data <your_dataset>:<mounting_point_dataset> \
        "python <script_and_parameters>"
    

**Important note** : within your python script, you'll want to make sure that the checkpoint is being saved to the `/output` folder. FloydHub will automatically save the contents of the `/output` directory as a job's `Output`, which is how you'll be able to leverage these checkpoints to resume jobs.

Once your job has been completed, you'll then be able to mount that's job's output as an input to your next job - allowing your script to leverage the checkpoint you created in the next run of this project.
    
    
    floyd run \
        [--gpu] \
        --env <env> \
        --data <your_dataset>:<mounting_point_dataset> \
        --data <output_of_previous_job>:<mounting_point_model> \
        "python <script_and_parameters>"
    

Okay, enough of that. Let's see how to make this tangible using three of the most popular frameworks on FloydHub.

# TensorFlow

![tf](/assets/images/content/images/2018/04/tf.png)

_[View full example on a FloydHub Jupyter Notebook](https://www.floydhub.com/redeipirati/projects/save-and-resume/53/code/tf_mnist_cnn_jupyter.ipynb)_

TensorFlow provides different ways to save and resume a checkpoint. In our example, we will use the [tf.Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator) API, which uses [tf.train.Saver](https://www.tensorflow.org/api_docs/python/tf/train/Saver), [tf.train.CheckpointSaverHook](https://www.tensorflow.org/api_docs/python/tf/train/CheckpointSaverHook) and [tf.saved_model.builder.SavedModelBuilder](https://www.tensorflow.org/api_docs/python/tf/saved_model/builder/SavedModelBuilder) behind the scenes.

To be more clear, the `tf.Estimator` API uses the first function to save the checkpoint, the second one to act according to the adopted checkpointing strategy, and the last one to export the model to be served with `export_savedmodel()` method.

Let's dig in.

## Saving a TensorFlow checkpoint

Before initializing an `Estimator`, we have to define the checkpoint strategy. To do so, we have to create a configuration for the Estimator using the [tf.estimator.RunConfig](https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig) API. Here's an example of how we might do this:
    
    
    # Save the checkpoint in the /output folder
    filepath = "/output/mnist_convnet_model"
    
    # Checkpoint Strategy configuration
    run_config = tf.contrib.learn.RunConfig(
        model_dir=filepath,
        keep_checkpoint_max=1)
    

In this way, we're telling the estimator which directory to save or resume a checkpoint from, and also how many checkpoints to keep.

Next, we have to provide this configuration at the initialization of the `Estimator`:
    
    
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
          model_fn=cnn_model_fn, config=run_config)
    

That's it. Seriously. We're now set up to save checkpoints in our TensorFlow code.

## Resuming a TensorFlow checkpoint

Guess what? We're also already set up to resume from checkpoints in our next experiment run. If the `Estimator` finds a checkpoint inside the given model folder, it will load from the last checkpoint.

## Okay, let me try

Don't take my word for it - try it out yourself. Here are the steps to run the TensorFlow checkpointing example on FloydHub.

### Via FloydHub's Command Mode

#### First time training command:
    
    
    floyd run \
        --gpu \
        --env tensorflow-1.3 \
        --data redeipirati/datasets/mnist/1:input \
        'python tf_mnist_cnn.py'
    

  * The `--env` flag specifies the environment that this project should run on (Tensorflow 1.3.0 + Keras 2.0.6 on Python3.6)
  * The `--data` flag specifies that the pytorch-mnist dataset should be available at the `/input` directory
  * The `--gpu` flag is actually optional here - unless you want to start right away with running the code on a GPU machine

#### Resuming from your checkpoint:
    
    
    floyd run \
        --gpu \
        --env tensorflow-1.3 \
        --data redeipirati/datasets/mnist/1:input \
        --data <your-username>/projects/save-and-resume/<jobs>/output:/model \
        'python tf_mnist_cnn.py'
    

  * The `--env` flag specifies the environment that this project should run on (Tensorflow 1.3.0 + Keras 2.0.6 on Python3.6)
  * The first `--data` flag specifies that the pytorch-mnist dataset should be available at the `/input` directory
  * The second `--data` flag specifies that the output of a previus Job should be available at the `/model` directory
  * The `--gpu` flag is actually optional here - unless you want to start right away with running the code on a GPU machine

### Via FloydHub's Jupyter Notebook Mode
    
    
    floyd run \
        --gpu \
        --env tensorflow-1.3 \
        --data redeipirati/datasets/mnist/1:input \
        --mode jupyter
    

  * The `--env` flag specifies the environment that this project should run on (Tensorflow 1.3.0 + Keras 2.0.6 on Python3.6)
  * The `--data` flag specifies that the pytorch-mnist dataset should be available at the `/input` directory
  * The `--gpu` flag is actually optional here - unless you want to start right away with running the code on a GPU machine
  * The `--mode` flag specifies that this job should provide a Jupyter notebook instance

#### Resuming from your checkpoint:

Just add `--data <your-username>/projects/save-and-resume/<jobs>/output:/model` to the previous command if you want to load a checkpoint from a previous Job in your Jupyter notebook.

# Keras

![keraslogo](/assets/images/content/images/2018/04/keraslogo.png)

_[View full example on a FloydHub Jupyter Notebook](https://www.floydhub.com/redeipirati/projects/save-and-resume/53/code/keras_mnist_cnn_jupyter.ipynb)_

Keras provides a great API for saving and loading checkpoints. Let's take a look:

## Saving a Keras checkpoint

Keras provides a set of functions called [callbacks](https://keras.io/callbacks/): you can think of callbacks as events that will be triggered at certain training states. The callback we need for checkpointing is the [ModelCheckpoint](https://keras.io/callbacks/#modelcheckpoint) which provides all the features we need according to the checkpointing strategy we adopted in our example.

**Note: this function will only save the model's weights** \- if you want to save the entire model or some of the components, you can take a look at [the Keras docs on saving a model](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model).

First up, we have to import the callback functions:
    
    
    from keras.callbacks import ModelCheckpoint
    

Next, just before the call to `model.fit(...)`, it's time to prepare the checkpoint strategy.
    
    
    # Save the checkpoint in the /output folder
    filepath = "/output/mnist-cnn-best.hdf5"
    
    # Keep only a single checkpoint, the best over test accuracy.
    checkpoint = ModelCheckpoint(filepath,
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=True,
                                mode='max')
    

  * `filepath="/output/mnist-cnn-best.hdf5"`: Remember, FloydHub will save the contents of `/output` folder! See [more on job output in the FloydHub docs](https://docs.floydhub.com/guides/data/storing_output/),
  * `monitor='val_acc'`: This is the metric we care about - validation accuracy,
  * `verbose=1`: It will print more information
  * `save_best_only=True`: Keep only the best checkpoint (in terms of maximum validation accurancy)
  * `mode='max'`: Save the checkpoint with max validation accuracy

By default, the period (or checkpointing frequency) is set to 1, which means at the end of every epoch.

For more information (such as filepath formatting options, checkpointing period, and more), you can explore the Keras [ModelCheckpoint](https://keras.io/callbacks/#modelcheckpoint) API.

Finally, we are ready to see this checkpointing strategy applied during model training. In order to do this, we need to pass the callback variable to the `model.fit(...)` call:
    
    
    # Train
    model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[checkpoint])  # <- Apply our checkpoint strategy
    

According to our chosen strategy, you will see:
    
    
    # This line when the training reach a new max
    Epoch < n_epoch >: val_acc improved from < previous val_acc > to < new max val_acc >, saving model to /output/mnist-cnn-best.hdf5
    
    # Or this line
    Epoch < n_epoch >: val_acc did not improve
    

That's it - you're now set up to save your Keras checkpoints.

## Resuming a Keras checkpoint

Keras models provide the [`load_weights()`](https://github.com/fchollet/keras/blob/master/keras/models.py#L718-L735) method, which loads the weights from a `hdf5` file.

To load the model's weights, you just need to add this line after the model definition:
    
    
    ... # Model Definition
    
    model.load_weights(resume_weights)
    

## Okay, let me try

Here's how you can do run this Keras example on FloydHub:

### Via FloydHub's Command Mode

#### First time training command:
    
    
    floyd run \
        --gpu \
        --env tensorflow-1.3 \
        'python keras_mnist_cnn.py'
    

  * The `--env` flag specifies the environment that this project should run on (Tensorflow 1.3.0 + Keras 2.0.6 on Python3.6)
  * The `--gpu` flag is actually optional here - unless you want to start right away with running the code on a GPU machine

[Keras provides an API to handle MNIST data](https://keras.io/datasets/#mnist-database-of-handwritten-digits), so we can skip the dataset mounting in this case.

#### Resuming from your checkpoint:
    
    
    floyd run \
        --gpu \
        --env tensorflow-1.3 \
        --data <your-username>/projects/save-and-resume/<jobs>/output:/model \
        'python keras_mnist_cnn.py'
    

  * The `--env` flag specifies the environment that this project should run on (Tensorflow 1.3.0 + Keras 2.0.6 on Python3.6)
  * The `--data` flag specifies that the output of a previus Job should be available at the `/model` directory
  * The `--gpu` flag is actually optional here - unless you want to start right away with running the code on a GPU machine

### Via FloydHub's Jupyter Notebook Mode
    
    
    floyd run \
        --gpu \
        --env tensorflow-1.3 \
        --mode jupyter
    

  * The `--env` flag specifies the environment that this project should run on (Tensorflow 1.3.0 + Keras 2.0.6 on Python3.6)
  * The `--gpu` flag is actually optional here - unless you want to start right away with running the code on a GPU machine
  * The `--mode` flag specifies that this job should provide us a Jupyter notebook.

#### Resuming from your checkpoint:

Just add `--data <your-username>/projects/save-and-resume/<jobs>/output:/model` if you want to load a checkpoint from a previous job.

# PyTorch

![py](/assets/images/content/images/2018/04/py.png)

_[View full example on a FloydHub Jupyter Notebook](https://www.floydhub.com/redeipirati/projects/save-and-resume/53/code/pytorch_mnist_cnn_jupyter.ipynb)_

Unfortunately, at the moment, PyTorch does not have as easy of an API as Keras for checkpointing. We'll need to write our own solution according to our chosen checkpointing strategy.

## Saving a PyTorch checkpoint

PyTorch does not provide an all-in-one API to defines a checkpointing strategy, but it does provide a simple way to save and resume a checkpoint. According the official docs about [semantic serialization](http://pytorch.org/docs/master/notes/serialization.html), the best practice is to save only the weights - due to a code refactoring issue.

Therefore, let's take a look at how to save the model weights in PyTorch.

First up, let's define a `save_checkpoint` function which handles all the instructions about the number of checkpoints to keep and the serialization on file:
    
    
    def save_checkpoint(state, is_best, filename='/output/checkpoint.pth.tar'):
        """Save checkpoint if a new best is achieved"""
        if is_best:
            print ("=> Saving a new best")
            torch.save(state, filename)  # save checkpoint
        else:
            print ("=> Validation Accuracy did not improve")
    

Then, inside the training (which is usually a for-loop of the number of epochs), we define the checkpoint frequency (in our case, at the end of every epoch) and the information we'd like to store (the epochs, model weights, and best accuracy achieved):
    
    
    ...
    
    # Training the Model
    for epoch in range(num_epochs):
        train(...)  # Train
        acc = eval(...)  # Evaluate after every epoch
    
        # Some stuff with acc(accuracy)
        ...
    
        # Get bool not ByteTensor
        is_best = bool(acc.numpy() > best_accuracy.numpy())
        # Get greater Tensor to keep track best acc
        best_accuracy = torch.FloatTensor(max(acc.numpy(), best_accuracy.numpy()))
        # Save checkpoint if is a new best
        save_checkpoint({
            'epoch': start_epoch + epoch + 1,
            'state_dict': model.state_dict(),
            'best_accuracy': best_accuracy
        }, is_best)
    

That's it! You can now save checkpoints in your PyTorch experiments.

## Resuming a PyTorch checkpoint

To resume a PyTorch checkpoint, we have to load the weights and the meta information we need before the training:
    
    
    # cuda = torch.cuda.is_available()
    if cuda:
        checkpoint = torch.load(resume_weights)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(resume_weights,
                                map_location=lambda storage,
                                loc: storage)
    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (trained for {} epochs)".format(resume_weights, checkpoint['epoch']))
    

For more information on loading GPU-trained weights on a CPU instance, you can check out this [PyTorch discussion](https://discuss.pytorch.org/t/loading-weights-for-cpu-model-while-trained-on-gpu/1032).

## Okay, let me try

Here's how you can do run this PyTorch example on FloydHub:

### Via FloydHub's Command Mode

#### First time training command:
    
    
    floyd run \
        --gpu \
        --env pytorch-0.2 \
        --data redeipirati/datasets/pytorch-mnist/1:input \
        'python pytorch_mnist_cnn.py'
    

  * The `--env` flag specifies the environment that this project should run on (PyTorch 0.2.0 on Python 3)
  * The `--data` flag specifies that the pytorch-mnist dataset should be available at the `/input` directory
  * The `--gpu` flag is actually optional here - unless you want to start right away with running the code on a GPU machine

#### Resuming from your checkpoint:
    
    
    floyd run \
        --gpu \
        --env pytorch-0.2 \
        --data redeipirati/datasets/pytorch-mnist/1:input \
        --data <your-username>/projects/save-and-resume/<jobs>/output:/model \
        'python pytorch_mnist_cnn.py'
    

  * The `--env` flag specifies the environment that this project should run on (PyTorch 0.2.0 on Python 3)
  * The first `--data` flag specifies that the pytorch-mnist dataset should be available at the `/input` directory
  * The second `--data` flag specifies that the output of a previus Job should be available at the `/model` directory
  * The `--gpu` flag is actually optional here - unless you want to start right away with running the code on a GPU machine

### Via FloydHub's Jupyter Notebook Mode
    
    
    floyd run \
        --gpu \
        --env pytorch-0.2 \
        --data redeipirati/datasets/pytorch-mnist/1:input \
        --mode jupyter
    

  * The `--env` flag specifies the environment that this project should run on (PyTorch 0.2.0 on Python 3)
  * The `--data` flag specifies that the pytorch-mnist dataset should be available at the `/input` directory
  * The `--gpu` flag is actually optional here - unless you want to start right away with running the code on a GPU machine
  * The `--mode` flag specifies that this job should provide us a Jupyter notebook.

#### Resuming from your checkpoint:

Just add `--data <your-username>/projects/save-and-resume/<jobs>/output:/model` if you want to load a checkpoint from a previous Job.

# Making progress together

We covered a lot of ground today, so feel free to reach out if you have questions about checkpointing your models on FloydHub. We're working to build a seamless workflow for your deep learning training, and checkpointing is an important part of that experience! Thanks, and happy training :)

* * *

#### FloydHub Call for AI writers

Want to write amazing articles like Alessio and play your role in the long road to Artificial General Intelligence? [We are looking for passionate writers](https://floydhub.github.io/write-for-floydhub/?utm_source=floydhub&utm_medium=banner&utm_campaign=call_for_writers_2019), to build the world's best blog for practical applications of groundbreaking A.I. techniques. FloydHub has a large reach within the AI community and with your help, we can inspire the next wave of AI. [Apply now](https://goo.gl/forms/PbOw0VmUnOfO1Lxp1) and join the crew!