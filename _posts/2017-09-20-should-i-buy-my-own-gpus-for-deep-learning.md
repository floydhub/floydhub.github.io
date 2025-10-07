---
author: Naren Thiagarajan
date: 2017-09-20 04:48:00 +0000
excerpt: Deep learning algorithms involve huge amounts of matrix multiplications and
  other operations which can be massively parallelized. GPUs usually consist of thousands
  of cores which can speed up these op...
feature_image: /assets/images/hero/should-i-buy-my-own-gpus-for-deep-learning-hero.jpg
layout: post
slug: should-i-buy-my-own-gpus-for-deep-learning
tags: [machine-learning]
title: Should I buy my own GPUs for Deep Learning?
---

Deep learning algorithms involve huge amounts of matrix multiplications and other operations which can be massively parallelized. GPUs usually consist of thousands of cores which can speed up these operations by a huge factor and reduce training time drastically. This makes GPUs essential to doing effective deep learning. Most data scientists are now faced with a question of whether to buy GPUs off the shelf and set up their own rigs or instead spin up cloud instances with GPUs.

There are some obvious similarities here to the dilemma faced by engineers a decade ago on where to host their web applications. They had to choose between setting up their own servers and using cloud services. The choice may seem much more straight forward now - around [90% of all enterprises](https://www.rightscale.com/blog/cloud-industry-insights/cloud-computing-trends-2017-state-cloud-survey) now use cloud services in their infrastructure.

This post walks through some of the major considerations to keep in mind when scaling your deep learning efforts.

### Choice of GPUs

When you are getting your own GPU, you have a much wider choice to choose among. The newer consumer grade GPUs like GTX 1080 are about 4 times faster than any cloud GPU. The consumer grade GPUs are also relatively cheaper. Tim Dettmers wrote a great  
[article](http://timdettmers.com/2017/04/09/which-gpu-for-deep-learning/) recently comparing the various GPUs for deep learning.

### Cost

Another consideration to keep in mind when purchasing your own GPUs is that all costs, except electricity, are paid upfront. Depending on your setup this can get pretty high. You'll also need to build a server rig to support your GPU (you cannot plug-in your new GPU into your laptop). You can do this for [under $1,000](https://www.oreilly.com/learning/build-a-super-fast-deep-learning-machine-for-under-1000) \- but a reasonably future-proof machine will cost around _$2,000_. [Andrej Karpathy's deep learning rig](https://twitter.com/karpathy/status/648256662554341377) costs around _$5,000_.

### New GPUs Specifically for Deep Learning

Of course, one of the major downsides to buying your own GPUs is that you are stuck with them for a while. The field of deep learning has exploded in the recent months, and the hardware manufacturers are starting to catch up with the demand now. There will soon be newer GPUs coming out specifically for deep learning - for instance [Tesla P100](http://www.nvidia.com/object/tesla-p100.html) and [V100s](http://www.anandtech.com/show/11559/nvidia-formally-announces-pcie-tesla-v100-available-later-this-year). The cloud providers will make then available in the next 12 - 18 months. When you buy your own hardware, you run the risk of the hardware becoming obsolete sooner than you might expect.

There are also some cases where you will not be able to buy certain hardware - like  
[TPUs from Google](https://cloud.google.com/blog/big-data/2017/05/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu) which are optimized for running Tensorflow code, and will only be available on Google Cloud Platform.

### Parallelism

Deep learning requires running multiple separate experiments in parallel to do parameter sweeping (or [hyper-parameter optimization](https://en.wikipedia.org/wiki/Hyperparameter_\(machine_learning\))). Since deep learning is a black box when it comes to visibility into its models, parallel training is important - it helps to traverse the parameter space efficiently and get to the final model faster.

Deep learning frameworks are starting to support [distributed training](https://www.tensorflow.org/deploy/distributed) to work with really large datasets. Cloud GPUs are easy to scale for such cases as a single user will need 10s of GPU instances for training, which could become prohibitively expensive if you needed to purchase all that GPU hardware yourself.

### Bursty workloads

Some users might not require a GPU 24x7. You could use a CPU instance (such as your laptop) to build your implementation and test on a small dataset. You can then use one or more GPU instances only when you need the computation power when training on a large dataset. With cloud services you have the flexibility to rent a GPU instance only when you need it.

### Reproducibility

It is important that your deep learning training is reproducible by others. Reproducibility enables collaboration which leads to improved models. It is very difficult to share your a local setup completely with your collaborators. There are tools like [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) that help facilitate collaboration, but it is still essential to version the entire tool chain - including your CUDA drivers, specific versions of cuDNN, deep learning framework (e.g. Tensorflow) and their dependencies. Using cloud instances with [standard machine images](https://aws.amazon.com/amazon-ai/amis/) will help with reproducibility.

### The road ahead

Currently, there is only a small set of use cases where buying your own GPUs would make sense for most people. With the landscape of deep learning changing rapidly both in software and hardware capabilities, it is a safe bet to rely on cloud services for all your deep learning needs.

* * *

#### FloydHub

Take full advantage of cloud GPUs with [FloydHub](https://www.floydhub.com/) \- a fully managed deep learning platform to develop, train and deploy your deep learning models. You can develop interactively using [Jupyter notebooks](http://docs.floydhub.com/getstarted/quick_start_jupyter/) and use GPU instances when required. All your  
experiments are versioned end-to-end for complete reproducibility.