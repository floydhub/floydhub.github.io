---
layout: "post"
title: "Counting the Crowd at Shake Shack with Deep Learning"
date: "2018-08-29 19:47:01 +0000"
slug: "humans-of-ml-dimitri-roche"
author: "Charlie Harrington"
excerpt: "The FloydHub #humansofml interview with Dimitri Roche - a software engineer who built a Shake Shack crowd-counting CoreML app using deep learning and FloydHub"
feature_image: "__GHOST_URL__/content/images/2018/08/Screen-Shot-2018-08-29-at-10.13.19-AM-1.png"
tags: "[]"
---

Dimitri Roche is a software engineer based in New York City, where our paths first aligned based on a mutual love of Shake Shack.

If you’re not yet familiar, Shake Shack is a tasty burger chain founded by hospitality-guru Danny Meyer. Praised for juicy burgers, crispy fries, and malty shakes, Shake Shake is perhaps best known for the infamously long line at its flagship Madison Square Park location (you can check out their [24/7 live webcam](https://www.shakeshack.com/location/madison-square-park) if you don’t believe me). Forgetting to check **The** **Shack Cam** has [devastated many a lunch hour](https://nypost.com/2010/04/14/new-yorks-longest-restaurant-lines/slide-1/) for NYC’s Flatiron office dwellers.

![](/assets/images/content/images/2018/08/Screen-Shot-2018-08-29-at-10.22.17-AM.png)A grim scene from the "Shack Cam"

Last year, Dimitri decided to make things a little better for his fellow New Yorkers with deep learning. He released [**Count**](http://count.dimroc.com/) \- a dynamic crowd-counting app based on a [multi-scale convolutional neural network](https://blog.dimroc.com/2017/11/19/counting-crowds-and-lines/), trained on [FloydHub’s cloud GPU machines](https://www.floydhub.com/?utm_source=blog&utm_medium=link&utm_campaign=humans_of_ml_dimitri_roche) using live-stream data from the Shack Cam.

Recently, Dimitri’s been busy [porting Count to run natively on iOS devices](https://blog.dimroc.com/2018/08/12/counting-crowds-with-coreml/) for more real-time crowd-detection using Apple’s [CoreML](https://developer.apple.com/documentation/coreml) and MLKit frameworks. 

!["Count" on the iPhone X using CoreML](/assets/images/content/images/2019/08/CrowdCountiOS.gif)"Count" on the iPhone X using CoreML

We sat down with Dimitri to chat about Shake Shack, the challenges and opportunities of “Edge AI”, and how to get started with deep learning.

**I love your crowd counting project because it’s the perfect blend of cool new technology with something practical in the real world. Tell me, what’s the longest you’ve ever waiting in line at Shake Shack?**

At least thirty minutes at Madison Square Park. It was worth it.

****What’s your go-to** Shake Shack **order?****

It used to be the double cheeseburger and a side of cheese fries, but now I’m a bad vegan, so I get a single cheeseburger.

****What led you to think that a deep learning approach could help solve this problem?****

Deep learning excels at taking fuzzy input to answer a specific question. I’ve been collaborating with Steve Berry, of [Thought Merchants](http://thoughtmerchants.com/), for a while now and we wanted to create a fun proof-of-concept product for this emerging technology. Given that the Shake Cam is a fixed camera with some consistency in the backdrop, I thought it would translate well to the ML domain and Steve thought it could be an interesting brand. I then set out to answer the question: how many people are in line?

****So, how exactly does your crowd-counting model work?****

After quickly realizing that off the shelf object detection wouldn’t work due to how dense crowds get, I started reading white papers about crowd counting and settled on [multi-scale convolutional neural network](https://blog.dimroc.com/2017/11/19/counting-crowds-and-lines/). It feeds an image into a multi-column convolutional neural network (CNN) that maps heads to a density or heat map. The sum of all these pixels are the final size of the crowd.

![The sum of the pixel values is the size of the crowd](/assets/images/content/images/2019/08/predictionBreakdown.jpg)The sum of the pixel values is the size of the crowd

The next subtle problem that needs to be solved is separating the size of the line from the size of the whole crowd. 

**What do you mean by that?**

The people sitting and eating are no longer in line and need to be excluded. Another fuzzy input scenario. This is solved with a second neural network of two dense layers, trained against the heatmap, to exclude people not in the contiguous series of dots from the front of the line. 

![](/assets/images/content/images/2019/08/lineNotHot.jpg)![](/assets/images/content/images/2019/08/lineNotHot2.jpg)

****Now, let’s try same question, but as a Reddit-style ELI5 “Explain Like I’m Five.”** How does your crowd-counting app work?**

My app Count creates a heatmap of heads where 1.0 is present and 0.0 is absent, then adds up all the pixels in the image to get the total size of the crowd. From this crowd, it figures out how many people are actually waiting by seeing which dots make up a solid line, and only counts those pixels.

****What were the biggest challenges you faced in developing your original crowd counting model?****

  1. Getting the models to converge and not run away and predict gibberish. I couldn’t trust my mean squared error alone, because it would seemingly get lower but in an undesirable way that would leave blank images, and eventually stall. I mitigated this by generating a picture of the prediction every epoch to see if progress was actually being made.
  2. If we include the Core ML version of Count, handling such wildly different input and crowd densities stretched the input past fuzzy and made prediction a challenge for a single model. I opted to create an ML pipeline that classified the crowds as either singles, tens, hundreds, and used a model tailored for those scenarios.

![CoreML Pipeline: Crowd image classification determines which counting model to use](/assets/images/content/images/2019/08/CrowdCountStrategies.jpg)CoreML Pipeline: Crowd image classification determines which counting model to use

**Before we dig into the CoreML version, I have a follow-up question.**I’m surprised you didn’t mention data annotation, data processing, or data version control as a** n overall**challenge. Can you let us know how you handled your data pipeline for this project?****

Well, I solved the annotation problem by throwing some money at it. I duct-taped together a [head annotator](https://github.com/dimroc/head-annotator) and crowd sourced the work through Amazon Mechanical Turk.

![](/assets/images/content/images/2018/08/Screen-Shot-2018-08-29-at-10.23.46-AM.png)Head annotation built on top of Amazon Mechanical Turk

It was expensive annotating thousands of these images, but I certainly didn’t want to do it myself. Definitely worth it to start, but now I’m reluctant to spend more money on annotations now that [Count](http://count.dimroc.com/) is up and running as a proof of concept.

As for data versioning, I used [git lfs](https://git-lfs.github.com/) to store the data, and appreciated FloydHub’s automatic data versioning to keep track of which model used which data. Proper ML shops will have a relatively fixed model and churn on the data to squeak out better accuracy. Since this was a proof of concept from a one-man shop, I experimented more with the layers of the model and less with the training data. If I were to take this to the next level, data versioning would be critical for me because I would be training the same model architecture in parallel with different data.

****Have you shared your model with Shake Shack? Are they thinking about adding it to the official Sha** ck**Cam site? Please say yes.****

Haha, no. There are weeks when Count's amazingly accurate, and some weeks, like snowy dark ones, where the model coughs up some weird numbers.

![](/assets/images/content/images/2019/08/snownotpeople.jpg)

Knowing that Count is still susceptible to these edge cases has made me reserved about publicizing it. My main motivation was to learn and have fun on the way, everything else is a welcome bonus.

****Let’s change gears to your recent work. You’ve been porting Crowd to run on mobile devices, where models are local to the device. You mentioned that this part of a trend called “Edge AI.” Can you explain Edge AI a bit more?****

When dealing with networks, the term “edge” is used to describe the machine closest to the request. For example, in a CDN, an edge server is the server closest to the client. Borrowing that term, Edge AI describes a scenario where a neural network, or some other from of AI, is running on a local device, like your phone, as opposed to a server in the cloud.

****What are the pros and cons of running models on devices versus on the cloud?****

Lots of pros:

  * Improved privacy is a big advantage of running models locally. You no longer need to upload potentially sensitive information to have an intelligent answer.
  * In scenarios involving real time interaction, we reduce latency and improve responsiveness by not needing to connect to a server.
  * Offloading all that processing to phones will reduce server infrastructure and costs for developers.

Cons:

  * Unforgiving with battery consumption
  * Limited GPUs can limit model complexity and performance.

****You’re using Apple’s CoreML and MLKit libraries. How’s that going?****

Core ML has been pretty pleasant. If you are using Keras, the coremltools will have you covered for most scenarios. There can be some nuances when dealing with variable sized inputs, as I discuss in my [blog](https://blog.dimroc.com/2018/08/12/counting-crowds-with-coreml/), but for most people, you can convert from Keras to Core ML with:
    
    
    coreml_model = coremltools.converters.Keras.convert(
        path,
        input_names=['input_1'],
        image_input_names=['input_1'],
        output_names=['density_map'])
    
    coreml_model.save("CrowdPredictor.mlmodel")

And then drag and drop the new file into Xcode.

My next project will involve sequences on recurrent neural networks (RNNs) and both frameworks have limited capabilities in that field. We’ll see what happens.

**Are you thinking about porting to Android as well?**

I wanted to cover Android, and thought about using [Flutter](https://flutter.io/), but I need to stay native for ML and could only take on so much at a time. This endeavor focused on Edge AI and I’ve learned a lot from it.

****What are some real-world use-cases for your mobile device model?****

  * Stadiums could count the number of people in attendance.
  * Businesses could tie this into their POS to figure out times of peak demand and hire accordingly.

****I looked for your app on the iOS App Store, but didn’t see it there yet. Do you have any plans to release Crowd on the App Store?****

This all uses iOS 12 Beta for Core ML 2, which won’t be available until around September 2018. Perhaps then I’ll release it.

**Is Count publicly available for people to reproduce?**

Yes! It’s available here: <https://github.com/dimroc/count>. 

****What are you hoping to learn or tackle next?****

I’m hopping on the recurrent neural network (RNN) train and looking into speech recognition. Specifically [Lip Reading Sentences in the Wild](https://arxiv.org/pdf/1611.05358.pdf).

****Any advice for people who want to get started with deep learning or programming?****

Take Andrew Ng’s Coursera course! Or if you’re already knee deep in some data science, step out of the Jupyter notebook for a while and make something that works end to end.

****Let’s pretend that Danny Meyer’s here with us -- is there anything you want to say to him? Any special requests for a new burger or shake option?****

I’m pretty happy with the classics really. Treat those cows well so I don’t feel so bad eating them.

****Where can people go to learn more about you and your work?****

Feel free to follow me on [Twitter](https://twitter.com/dimroc), [GitHub](https://github.com/dimroc), or my [experiments](https://experiments.dimroc.com/#/) site.