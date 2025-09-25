---
author: Sai Soundararaj
date: 2018-06-05 20:51:34 +0000
excerpt: Meet Workspaces 👋- a new cloud IDE for deep learning, powered by FloydHub
  GPUs
feature_image: /assets/images/hero/workspaces-hero.png
layout: post
slug: workspaces
tags: '[]'
title: Workspaces on FloydHub
---

Meet **Workspaces** 👋 - a new cloud IDE for deep learning, powered by FloydHub.

![](/assets/images/content/images/2018/06/workspace-4.gif)

Workspaces is an interactive environment for developing and training deep learning models on FloydHub. You can run Jupyter notebooks, Python scripts, access the terminal, and much more. Even better, you can seamlessly toggle between CPU and GPU machines while you're working, right when you're ready for that extra computing power. All the files and data in a Workspace are preserved across sessions, just like your local computer. 

In fact, we like to think of Workspaces as your persistent, on-demand deep learning computer in the cloud.

## Rethinking our Jupyter Notebook experience

Workspaces is the latest evolution of our popular Jupyter Notebook development experience. 

We've traditionally offered three modes for running jobs on FloydHub:

  * **Command mode** : Running pre-defined model training scripts

    
    
    floyd run "python train.py"

  * **Serving mode** : Spinning up a REST-endpoint to integrate your trained models with your production apps and services

    
    
    floyd run --mode serve

  * **Jupyter Notebook mode** : Opening up an interactive environment for model development 

    
    
    floyd run --mode jupyter

Far and away, Jupyter Notebook mode has been our most popular mode for jobs, so we dug in with our customers to explore ways to make the experience even better. 

Here's what we came up with:

### Embracing JupyterLab

We're huge fans of [Project Jupyter](http://jupyter.org/), and we're going all-in on JupyterLab – their [next evolution](https://blog.jupyter.org/jupyterlab-is-ready-for-users-5a6f039b8906) of the Jupyter Notebook. JupyterLab is a fully extensible interactive computing environment, with all sorts of powerful features:

  * Multiple notebook kernels in a single customizable UI
  * File browser with rich outputs for images, CSVs, TSVs, and more
  * Full terminal access

We think you'll love the new look and feel of the interactive Jupyter experience on FloydHub – we're already in ❤️.

### Ready to build, train, and deploy AI?

#### Get started with FloydHub's collaborative AI platform for free

###### [Try FloydHub for free ](https://www.floydhub.com/?utm_source=blog&utm_medium=banner&utm_campaign=try_floydhub_for_free)

### On-demand GPU machine, configured with your framework of choice

It can be hard for an individual or company to keep up with the ongoing and relentless march forward by the popular deep learning frameworks - it seems like there's a new TensorFlow version every few weeks!

FloydHub already provides dozens of pre-configured environments with the latest versions of most popular frameworks like TensorFlow and PyTorch, so that you don't have to worry about keeping your configuration and drivers updated. With Workspaces, it's gotten easier – you can now easily toggle between our deep learning environments in a single click.

###### Toggle between GPU and CPU machines

Even better, we wanted to make it as easy as possible to use GPU-compute in your projects. Previously, you could run a Jupyter job on FloydHub with either a CPU machine or a GPU machine - but this was something you had to decide up front. 

Now, with Workspaces, you can start your development process using one of our cheaper CPU machines, and then [seamlessly toggle](https://docs.floydhub.com/guides/workspace/#switching-between-cpu-and-gpu) your Workspace to a GPU machine – without losing your work, directly in your active Workspace, saving you both money and time. And that sounds 🙌 to us.

### Better insights into your machine and training metrics

You're not alone – deep learning can still feel like a black box. But we're working to fix that with Workspaces.

###### TensorBoard

Workspaces come with [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) pre-enabled for all environments. Just click the TensorBoard button in the metrics bar to start visualizing your training process.

###### System Metrics

We're also highlighting your machine's current vitals through our System Metrics. At a glance, you'll be able to see the latest and historical stats for your CPU or GPU utilization, disk space, RAM usage, and more. 

###### Insights

We'll also monitor your machine for any [Insights](https://docs.floydhub.com/guides/insights/) \- such as under-utilizing your GPU or running out of disk space – and alert you discretely when these situations come up. We want to make your model development process as seamless and painless as possible – and automatic machine Insights are another step in that direction.

![](/assets/images/content/images/2018/06/insight_alert.gif)Insights into your deep learning machine's vitals

### Attaching datasets to active environments

Deep learning is nothing without data, and FloydHub is no stranger to managing your company's datasets for training jobs. We're also hosting thousands of popular public datasets, readily available for your next project.

Previously, you needed to attach any datasets before starting your job. Needless to say, but this was annoying. Especially when you forgot to attach something.

With Workspaces, you can now attach (and remove) datasets before, during, and after any Workspace session. This is a game-changer 💥. 

Just open up the Data panel in your Workspace to search for your team's datasets or any public dataset, click attach, and we'll asynchronously add the dataset to your active Workspace.

## How to get started with Workspaces today

We're excited for you to check out Workspaces today. There are two ways to create a Workspace on FloydHub:

  1. **Click the "Create Workspace" button on any of your existing or new FloydHub projects.** You'll be able to either create a blank Workspace or bootstrap your Workspace with code from any public git URL - like GitHub, GitLab, or Bitbucket.
  2. **Use a starter Workspace "template" designed for a specific deep learning task**. We've prepared [a few templates](https://www.floydhub.com/explore/templates) for some of the most common machine learning and deep learning tasks, including:

  * [Sentiment Analysis](https://www.floydhub.com/explore/templates/natural-language/sentiment-analysis)
  * [Named Entity Recognition](https://www.floydhub.com/explore/templates/natural-language/named-entity-recognition)
  * [Object Classification](https://www.floydhub.com/explore/templates/computer-vision/object-classification)
  * [Price Prediction](https://www.floydhub.com/explore/templates/regression/price-prediction)

Templates are a great jumping off point for your next deep learning project. You can used a template to help you get something working as a baseline, and then you can adapt the model to your specific needs by swapping in your own dataset or making other adjustments.

![](/assets/images/content/images/2018/06/Screen-Shot-2018-06-05-at-12.05.57-PM.png)Workspace templates help you get started with a deep learning task

We're thrilled about Workspaces, and can't wait to keep improving the experience based on your feedback!