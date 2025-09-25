---
author: Naren Thiagarajan
date: 2017-09-06 05:06:00 +0000
excerpt: Want to run Tensorboard on FloydHub?! No problem! We make it easy.
feature_image: /assets/images/hero/tensorboard-on-floydhub-hero.png
layout: post
slug: tensorboard-on-floydhub
tags: '[]'
title: Tensorboard on FloydHub
---

### Tensorboard

[Tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) is a visualization tool that is packaged with Tensorflow. It is very useful for understanding and debugging Tensorflow projects.You can visualize your Tensorflow graph, plot quantitative metrics about the execution of your graph, and show additional data like images that pass through it.

### Enabling Tensorboard for your project

To enable Tensorboard for your job run, you just need to add `--tensorboard` flag to the floyd run command. For example:
    
    
    floyd run --env tensorflow-1.1 --tensorboard --mode jupyter
    

This should start a jupyter notebook using Tensorflow 1.1 environment and enable Tensorboard. You can open Tensorboard directly from the link on the job page in the dashboard.

![Dashboard](/assets/images/content/images/2018/06/dashboard.png)

### Demo

You can view a full demo of this feature in this video:

As always, let us know if you have any feedback on this and other features on FloydHub.