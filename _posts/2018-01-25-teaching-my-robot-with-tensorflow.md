---
author: Charlie Harrington
date: 2018-01-25 04:49:00 +0000
excerpt: 'If you''re like me, then you''d do pretty much anything to have your own
  R2-D2 or BB-8 robotic buddy. Just imagine the adorable adventures you''d have together!

  I''m delighted to report that the Anki Coz...'
feature_image: /assets/images/hero/teaching-my-robot-with-tensorflow-hero.jpg
layout: post
slug: teaching-my-robot-with-tensorflow
tags: [machine-learning]
title: Teaching My Robot With TensorFlow
---

If you're like me, then you'd do pretty much anything to have your own R2-D2 or BB-8 robotic buddy. Just imagine the adorable adventures you'd have together!

I'm delighted to report that the [Anki Cozmo](https://www.anki.com/en-us/cozmo) is the droid you've been looking for.

Cozmo is big personality packed into a itty-bitty living space. You don't need to know how to code to play with Cozmo - _but if you do_ \- then Cozmo has even more phenominal cosmic power.

In this post, I'm going to show you how you can teach your own Cozmo to recognize everyday objects using transfer learning with TensorFlow on FloydHub.

## The setup

Install the [Cozmo Python SDK](http://cozmosdk.anki.com/docs/), create a new virtualenv, and clone the [cozmo-tensorflow](https://www.github.com/whatrocks/cozmo-tensorflow) project to your local machine.
    
    
    virtualenv ~/.env/cozmo -p python3
    source ~/.env/cozmo/bin/activate
    git clone https://www.github.com/whatrocks/cozmo-tensorflow
    cd cozmo-tensorflow
    pip install -r requirements.txt
    

  

Next up - login to the FloydHub CLI (sign up for a [free account here](https://www.floydhub.com/plans)). If you need to install the FloydHub CLI, just [check out this guide](https://docs.floydhub.com/guides/basics/install/) in our [documentation](https://docs.floydhub.com/).
    
    
    floyd login
    

## 1\. Use Cozmo to generate training data

Getting enough training data for a deep learning project can be a pain. But thankfully we have a robot who loves to run around and take photos with his camera, so let's just ask Cozmo to take pictures of things we want our robot to learn.

Let's start with a can of delicious overpriced seltzer. Place Cozmo directly in front of a can of seltzer. Make sure that your robot has enough space to rotate around the can while it is taking pictures. Be sure to enter the name of the object that Cozmo is photographing when you run the `cozmo-paparazzi` script.
    
    
    python3 cozmo-paparazzi.py seltzer
    

  

![CozmoPaparazzi](/assets/images/content/images/2018/04/cozmo-paparazzi.gif)

Repeat this step for as many objects (labels) as you want Cozmo to learn! You should now see all your image labels as subdirectories within the `/data` folder of your local directory.

### Uploading dataset to FloydHub

Next up - let's upload our images to [FloydHub](https://www.floydhub.com/whatrocks/datasets/cozmo-images) as a [FloydHub Dataset](https://docs.floydhub.com/guides/create_and_upload_dataset/). This will allow us to mount these images during our upcoming model training and model serving jobs on FloydHub. Datasets on FloydHub are an easy way for your training jobs to reference a version-controlled dataset.
    
    
    cd data
    floyd data init cozmo-images
    floyd data upload
    

In our case, I've named this image dataset `cozmo-images`. I've made it a [public dataset](https://www.floydhub.com/whatrocks/datasets/cozmo-images), so feel free to use it in your own Cozmo projects!

## 2\. Training our model on FloydHub

And now the fun begins. First, Make sure you are in the project's root directory, and then initialize a FloydHub project so that we can train our model on one of FloydHub's fully-configured TensorFlow cloud GPU machines.

Side note - if that last sentence sounded like a handful, then just know that FloydHub takes care of configuring and optimizing everything on your cloud machine so that it's ready for your GPU-powered deep learning experiments. You can specify the exact deep learning framework you'd like to use - whether that's TensorFlow 1.4 or PyTorch 0.3 or [more](https://docs.floydhub.com/guides/environments/) \- and FloydHub will make sure your machine has everything you need to start training immediately.

Okay, back to business, let's initialize our project:
    
    
    floyd init cozmo-tensorflow
    

Now we're ready to kick off a deep learning training job on FloydHub.

A few things to note:

  * We'll be doing some simple transfer learning with the [Inception v3 model](https://github.com/tensorflow/models/tree/master/research/inception) provided by Google. Instead of training a model from scratch, we can start with this pre-trained model, and then just swap out its final layer so that we can teach it to recognize the objects we want Cozmo to learn. Transfer learning is a very useful technique, and you can read more about it on [TensorFlow's website](https://www.tensorflow.org/tutorials/image_retraining).
  * We're going to be mounting the images dataset that Cozmo created with the `--data` flag at the `/data` directory on our FloydHub machine.
  * I'm enabling Tensorboard for this job with the `--tensorboard` flag so that I can visually monitor my job's training process
  * I've edited the `retrain.py` script ([initially provided by the TensorFlow team](https://github.com/googlecodelabs/tensorflow-for-poets-2)) to write its output to the `/output` directory. This is super important when you're using FloydHub, because FloydHub jobs always store their outputs in the `/output` directory). In our case, we'll be saving our retrained ImageNet model and its associated training labels to the job's `/output` folder.

    
    
    floyd run \
      --gpu \
      --data whatrocks/datasets/cozmo-images:data \
      --tensorboard \
      'python retrain.py --image_dir /data'
    

  

That's it! There's no need to configure anything on AWS or install TensorFlow or deal with GPU drivers or anything like that. (If you're paying close attention, I didn't include the `--env` flag in my job command - that's because FloydHub's default environment includes TensorFlow 1.1.0 and Keras 2.0.6, and that's all I need for my training 😎).

Once your job is complete, you'll be able to see your newly retrained model in [your job's output directory](https://www.floydhub.com/whatrocks/projects/cozmo-tensorflow/8/output).

I recommend converting your job's output into a standalone FloydHub Dataset to make it easier for you to mount our retrained model in future jobs (which we're going to be doing in the next step). You can do this by clicking the 'Create Dataset' button on the job's output page. Check out the Dataset called [cozmo-imagenet](https://www.floydhub.com/whatrocks/datasets/cozmo-imagenet) to see my retrained model and labels.

## 3\. Connecting Cozmo to our retrained model

We can test our newly retrained model by running another job on FloydHub that:

  * Mounts our [retrained model and labels](https://www.floydhub.com/whatrocks/datasets/cozmo-imagenet)
  * Sets up a public REST endpoint for model serving

[Model-serving](https://docs.floydhub.com/guides/run_a_job/#-mode-serve) is an experimental feature on FloydHub - we'd love to hear your [feedback on Twitter](https://www.twitter.com/floydhub_)! In order for this feature to work, you'll need to include a simple Flask app called `app.py` in your project's code.

For our current project, I've created a simple Flask app that will receive an image from Cozmo in a POST request, evaluate it using the model we trained in our last step, and then respond with the model's results. Cozmo can then use the results to determine whether or not it's looking at a specific object.
    
    
    floyd run \
      --data whatrocks/datasets/cozmo-imagenet:model \
      --mode serve
    

  

Finally, let's run our `cozmo-detective.py` script to ask Cozmo to move around the office to find a specific object.
    
    
    python3 cozmo-detective.py toothpaste
    

  

Every time that Cozmo moves, the robot will send an black and white image of whatever it's seeing to the model endpoint on FloydHub - and FloydHub will run the model against this image, returning the following payload with "Cozmo's guesses" and how long it took to compute the guesses.
    
    
    {
      'answer': 
        {
          'plant': 0.022327899932861328, 
          'seltzer': 0.9057837128639221, 
          'toothpaste': 0.07188836485147476
        }, 
      'seconds': 0.947
    }
    

  

If Cozmo is at least 80% confident that it is looking at the object in question, then the robot will run towards it victoriously!

![finder](/assets/images/content/images/2018/04/cozmo-detective.gif)

Once Cozmo's found all your missing objects, don't forget to shut down your serving job on FloydHub.

## A new hope

  

_It's a magical world, Cozmo, ol' buddy... let's go exploring!_

I'm eager to see what you and your Cozmo can find together, along with a little help from your friends at FloydHub. Share your discoveries with us on [Twitter](https://twitter.com/floydhub_)!

### References

This project is an extension of @nheidloff's [Cozmo visual recognition project](https://github.com/nheidloff/visual-recognition-for-cozmo-with-tensorflow) and the [Google Code Labs TensorFlow for Poets project](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0). I also wrote about this project on my [personal site](https://www.charlieharrington.com/teaching-my-robot-with-tensorflow) \- except with a lot more references to Short Circuit and The Legend of Zelda.

### Ready to build, train, and deploy AI?

#### Get started with FloydHub's collaborative AI platform for free

###### [Try FloydHub for free ](https://www.floydhub.com/?utm_source=blog&utm_medium=banner&utm_campaign=try_floydhub_for_free)