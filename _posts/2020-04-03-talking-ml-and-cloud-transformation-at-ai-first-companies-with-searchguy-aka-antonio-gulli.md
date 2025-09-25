---
author: Alessio Gozzoli
date: 2020-04-03 22:46:13 +0000
excerpt: 'This Humans of Machine Learning interview has us sitting down with Searchguy,
  aka Antonio Gulli, who’s been a pioneer in the world of data science for 20+ years
  now, to talk transformation, opportunity, and mentorship, among other topics. '
feature_image: /assets/images/hero/talking-ml-and-cloud-transformation-at-ai-first-companies-with-searchguy-aka-antonio-gulli-hero.png
layout: post
slug: talking-ml-and-cloud-transformation-at-ai-first-companies-with-searchguy-aka-antonio-gulli
tags: '[]'
title: Talking ML and Cloud Transformation at AI-First Companies with @searchguy,
  aka Antonio Gulli
---

For this [Humans of Machine Learning](https://floydhub.github.io/tag/humans-of-ml/) ([#humansofml](https://twitter.com/hashtag/humansofml)) interview, I’m super excited to share my conversation with Antonio Gulli. Antonio is a part of computer lore - _**he created one of the first public search engines**_ back in 1996 when search was not even a thing. Not to mention that he did this in Italy!

Antonio is currently at the Office of CTO at Google Cloud, where he leads innovation at the intersection of infrastructure and machine learning. He is also the author of _Deep Learning with TensorFlow and Keras 2nd edition_.

In this conversation, we’ll cover Antonio’s journey, his takes on TensorFlow evolution, and his advice for spotting opportunities for ML and Cloud transformation. Antonio has a rich background in ML technologies and a real pleasure to talk to. I hope you enjoy our chat.

## TensorFlow Evolution: From Framework to Platform

** _**[Alessio]:**_****_**You started your journey in data science many years before it was recognized as the “[sexiest job](https://hbr.org/2012/10/data-scientist-the-sexiest-job-of-the-21st-century)” in the market. You also had an interesting startup career, many years before the dot com bubble. Could you give our readers a brief tour down memory lane? Tell us how all of this started for you.**_**

[Antonio]: During the late 90s, I was attending university in Pisa and got interested in neural networks, optimization, and parallel computation. At that time, it was probably not considered the smartest choice because neural networks were kind of dying and going through the "AI winter," a period of reduced funding and interest in artificial intelligence research. Parallel computation was considered very hard because it needed to run on very expensive computers.   
  
In 1995, I had a chance to play with the open-source code of Lycos, which was at that time a research project and later a web search engine. That was truly fascinating. So, I decided to work in that direction. Since the very early days, web search had a lot of data science and machine learning. In 1996, I formed a company in Italy and we worked on one of the first public search engines ever, [Arianna](https://it.wikipedia.org/wiki/Arianna_\(motore_di_ricerca\)). Google was founded in 1998. Today, Arianna is a piece of software mentioned in the [Computer History Museum](http://computerhistory.org) . My passion for data science is rooted there and has never stopped.

_**When did you start your journey at Google?**_

It started with a decline. In 2005, I declined an offer to join Google in Zurich when there were less than 10 people; today there are more than 4000 employees from 85 nationalities. Then, I applied a few years later but did not make the bar. Later, I joined as Engineering Director and Site Leader for Warsaw, Poland. Today I am in Google Cloud, London. 

> If you want to join Google, then simply do not give up. I know people who tried three times and made a huge contribution when they joined.

**_**In 2016 you published your first Deep Learning book,**_**Deep Learning with Keras** _**. It’s now out in its second edition,**_**Deep Learning with TensorFlow 2 and Keras, which I just read** _**. It’s definitely a great introduction to the topic. Given the speed at which the field is evolving, what has your experience been with the content update?**_**

> [#deeplearning](https://twitter.com/hashtag/deeplearning?src=hash&ref_src=twsrc%5Etfw) with [#tensorflow](https://twitter.com/hashtag/tensorflow?src=hash&ref_src=twsrc%5Etfw) and [#keras](https://twitter.com/hashtag/keras?src=hash&ref_src=twsrc%5Etfw) , 2nd ed is finally here! 620 pages providing extensive coverage of the progress in [#machinelearning](https://twitter.com/hashtag/machinelearning?src=hash&ref_src=twsrc%5Etfw) , [#deeplearning](https://twitter.com/hashtag/deeplearning?src=hash&ref_src=twsrc%5Etfw) , and [#neuralnetworks](https://twitter.com/hashtag/neuralnetworks?src=hash&ref_src=twsrc%5Etfw) during past 3 years. Thanks to [#google](https://twitter.com/hashtag/google?src=hash&ref_src=twsrc%5Etfw) and [#octo](https://twitter.com/hashtag/octo?src=hash&ref_src=twsrc%5Etfw) \- with [@palsujit](https://twitter.com/palsujit?ref_src=twsrc%5Etfw) and Amita<https://t.co/8ORL8eSoJV> [pic.twitter.com/qn3eFSiWeo](https://t.co/qn3eFSiWeo)
> 
> -- Antonio Gulli (@antoniogulli) [January 10, 2020](https://twitter.com/antoniogulli/status/1215526296496877568?ref_src=twsrc%5Etfw)

Book release!

In 2014, I read a few scientific papers on neural networks and started to think, "Hey, this thing seems to be having a comeback." The problem was finding a good software framework in which to develop machine learning models. [Theano](http://deeplearning.net/software/theano/) was a good one , but maybe a bit too low-level. I remember that [Keras](https://keras.io/) was impressive because it had the right level of abstraction for describing complex neural network models in a very simple and effective way. That was a perfect combination: neural networks were back and there was a good framework to play with. 

To my knowledge, no one had written a book about the progress made in deep learning with Keras during the previous 3-4 years. That's the reason why in 2016, my friend [Sujit](https://www.linkedin.com/in/sujitpal/) and I decided to write _[Deep Learning with Keras](https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-keras)_. I was in a coffee shop in Amsterdam and Sujit was in California. 

In 2017, TensorFlow 1.x was released by Google, and that inspired me to write a second book, _[TensorFlow Cookbook](https://www.packtpub.com/big-data-and-business-intelligence/tensorflow-1x-deep-learning-cookbook)_ with my friend [Amita](https://www.linkedin.com/in/amitakapoor/). This time, I was in a coffee shop in Warsaw and she was in India.

In 2019, Google released TensorFlow 2.X, adopting Keras as high level APIs. Amita, Sujit, and I joined our efforts and I wrote a third book, _[Deep Learning with TensorFlow 2 and Keras](https://www.packtpub.com/data/deep-learning-with-tensorflow-2-0-and-keras-second-edition)_. It was published on 27 Dec 2019, and it is considered the second edition of the first book. I was in many coffee shops in multiple cities—Warsaw, Pisa, and London—while Sujit in California and Amita in India. 

The first book was about 200 pages and was translated in Chinese and Japanese. The third book is more than 650 pages because a lot of things have happened in the intervening three years. Writing books is good because you learn new things and then need to explain them with simple words.

_**What do you think about the TensorFlow 2.0 release? What are your favorite bits?**_

Keras is a beautiful API with the right level of abstraction; it is really well done. TensorFlow is an ecosystem with many different components. I got excited to see things like [TPU](https://www.tensorflow.org/guide/tpu) support, [federated learning](https://www.tensorflow.org/federated/federated_learning), [TensorFlow Graphics](https://www.tensorflow.org/graphics), and [TensorFlow Agents](https://www.tensorflow.org/agents). I also like the fact that you have many pre-trained models that can be used for things like transfer learning. Then there is [eager execution](https://www.tensorflow.org/guide/eager), which has opened the door to interesting discussions in the community.

_**Do you think it closed the gap with PyTorch?**_  
  
I don't think there is a gap. Both the frameworks have done excellent work. TensorFlow is a large open-source machine learning ecosystem delivered by a team of AI experts and used in several thousand AI projects.

**_**In the last year, we’ve been seeing the evolution of TensorFlow from a framework to a more comprehensive ecosystem. What do you think is the driving force behind that?**_**

> TensorFlow has a more holistic vision where the end-to-end workflow is integrated. 

Model building and training is only a small fraction of a ML workflow. ML projects also need a lot of software engineering, from data ingestion to model training pipelines to test and deployment of models to production; that’s surely a driving force. 

> Additionally, we are moving toward an ecosystem of products that interoperate with each other. 

For instance, [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx) helps with building end-to-end machine learning pipelines. Then there is tooling to help model builders and model deployers understand their performance. A lot of effort has been put into integrations with beloved open-source tools like Numpy, SciPy, and Spacy.

The driving force behind this is TensorFlow's community. We are continuously impressed with the projects that they create, and the open-source tools that they share with the world that are powered by TF. [TF special interest groups](https://www.tensorflow.org/community/forums#special_interest_groups) have been doing fantastic work.

_**What are the next stages in this evolution?**_

I look forward to more and more integration of `tf.keras` and TF2.x support into TensorFlow Ecosystem. I’d recommend that anyone who wants to stay tuned in [join our community](http://tensorflow.org/community). All changes and prominent new features to our API are discussed as RFCs (Requests For Comments), [here](https://floydhub.github.io/p/e7c6fe82-2844-4960-b719-305586cbd63a/github.com/tensorflow/community).

**_**Recently Francois tweeted that “Tensorflow 2.0 changes everything.” Tell us more—what would you say is the vision that is driving this claim?**_**

> TensorFlow 2.0 is a machine learning platform from the future. It changes everything.
> 
> -- François Chollet (@fchollet) [September 29, 2019](https://twitter.com/fchollet/status/1178360311130144768?ref_src=twsrc%5Etfw)

I asked Francois about his tweet. He has a more detailed write up on his TensorFlow 2.0 + Keras crash course colab [notebook](https://colab.research.google.com/drive/1UCJt8EYjlzCs1H1d1X0iDGYJsHKwu-NO#scrollTo=zoDjozMFREDU). The notebook itself is a great resource and I recommend enthusiasts giving it a look. 

In essence, we’ve used the learning and feedback from the last four years of seeing TensorFlow 1.0 and Keras in use to take a big step with the redesign. We’ve fixed a bunch of issues in the previous paradigm and made the framework fast, production-ready and very scalable. This is a pretty important evolution for the deep learning community.

The evolution from TF1.x to TF2.x is covered in the second chapter of Deep Learning with Tensorflow 2.X with Keras. In particular, we provide many examples of TF1.x computation graph style and TF2.0 eager computation with a comparison of pros and cons and lots of attention to performance.

**_**I recently watched a[talk you gave in Milan](https://youtu.be/1UFzNl0d3BE). There was something you said that I found really interesting but at the same time extremely provocative: that a strong component of every Data Scientist workflow is based on data intuition, rather than science. Can you please elaborate?**_**

Data scientists build intuitions over time, and that’s a good thing. How can I clean my data? What features should I use? What model should I pick? How to tune hyper-parameters? So many options, so little time. 

Intuition is what drives a large part of our activities. However, you are right—I was purposely very provocative.

There are two key points. First, we all want to live in a world where rather mundane tasks are avoided. For instance, what about a tool for finding a good combination of hyper-parameters on your behalf? And what about a tool that would automatically select the best deep learning model for my problem? 

Second, it would be great to have more data, metrics, and scientific results driving our intuition, so having tools driving our intuition is super important.  
Google is investing in Cloud AutoML. This is based on Google’s state-of-the-art transfer learning and neural architecture search technology. Cloud AutoML enables developers with limited machine learning expertise to train high-quality models specific to their business needs. For instance, the paper “[Automated deep learning design for medical image classification by health-care professionals with no coding experience: a feasibility study,” __ by Livia Faes, et al.](https://www.sciencedirect.com/science/article/pii/S2589750019301086) and published in September, 2019, was written by doctors with no specific deep learning expertise, and Cloud AutoML helped them to obtain great results. We cover Cloud AutoML in Chapter 14 of our recent book.

> Cloud AutoML might be a cornerstone for the democratization of sophisticated algorithmic modelling in health care and in other domains.

**_**You have a bird’s eye view of the cloud space. What are the biggest challenges you’ve seen companies face while moving their AI projects to the cloud?**_**

The three major areas of challenges include infrastructure, data, and people.

First, as models move from prototypes to evaluation and production, the ML Ops engineers have to keep up with the changes to the tooling, environments, and underlying dependencies as specific as the hardware type. 

Second, the ability to utilize data at scale to optimize and improve the models is critical for deployments. Companies can struggle to achieve agility and scale by consolidating their view of data as they move to the cloud. 

Third, data scientists are not the only people your model needs. You need data engineers, ML Ops engineers, app developers, product managers, and data scientists to be working in unison throughout the ML lifecycle from data ingestion to model understanding to model deployment and management.

**_**What are some specific things Google Cloud is doing to address these challenges?**_**

Cloud offers a significant level of flexibility to address these challenges and it adapts to your data science needs. With open source offerings like [Kubeflow](https://www.kubeflow.org/), data scientists and ML Ops engineers are able to abstract the underlying infrastructure dependencies as they move their models from prototype to production. 

With an ETL service like [Data Fusion](https://cloud.google.com/data-fusion), you can easily ingest data from hundreds of different sources, thereby enabling you to get a scalable view of your data from multi-cloud environments. [BigQuery](https://cloud.google.com/bigquery) further amplifies the effect by providing you with unmatched scale and speed to analyze and extract value from this data. 

Lastly, we view data science and AI as a team sport. As a result, we build products and services that help every user who has a role in the ML lifecycle to be effective in their job.

Here’s an example: If you want to build and train your machine learning model, you can use GPUs and TPUs on-demand, scaling the cost of training up and down dynamically. TPUs are covered in Chapter 16 of our book. If you want, you can use predefined models and fine-tune them with transfer learning. For serving, there is an efficient infrastructure, including frameworks such as [TF-Serve](https://www.tensorflow.org/tfx/guide/serving). Plus, cloud offers a set tools to manage the ingestion-training-serving workloads in an efficient way. 

Personally, I love notebooks, especially [colab](https://colab.sandbox.google.com/), which gives a democratic access to resources. A few months ago, I had the pleasure to mentor two kids in Singapore who [used colab and cloud for smarter recycling](https://www.blog.google/around-the-globe/google-asia/singapore-students-using-cloud-smarter-recycling/).

Also, I want to mention that cloud offers also a set of API that are ready to use if you want to add high-quality machine learning functionalities to your projects (see [Google Cloud APIs](https://cloud.google.com/apis)).

One final trend is the idea of offering AI solutions and not only AI products or API. Google, for instance, invested a lot in [Google Cloud’s Contact Center AI](https://cloud.google.com/solutions/contact-center), an artificial intelligence solution that converses naturally with customers . This solution was inspired by [Google Duplex](https://ai.googleblog.com/2018/05/duplex-ai-system-for-natural-conversation.html).

**_**Recently we saw that all the major Cloud players have rebranded part of their offerings as AI platforms. How can companies take full advantage of the new offerings but at the same time avoid the vendor lock-in?**_**

I think there are two aspects here. First, I would suggest using established open source frameworks such as TensorFlow and Keras. Secondly, I would leverage [Kubernetes](https://kubernetes.io/) for orchestration because, again, it's open source. We've already discussed the benefits of cloud in the previous question but, in this context, I would add that having cloud-managed solutions has many cost benefits because you can focus on core data science work. For instance, if you are looking for something that works across multiple clouds and premises, you can have a look at [Anthos](https://cloud.google.com/anthos), an open hybrid and multi-cloud application platform.  
  
And I’ll mention [Kubeflow](https://www.kubeflow.org/) one more time, which makes deployments of machine learning workflows on Kubernetes simple, portable, and scalable.

Personally, I believe that cloud and machine learning are offering companies the opportunity to provide innovative solutions for a very large number of customers. This transformation is about becoming a platform with unprecedented user bases and offering functionality as a service in a way that better satisfies the customer.

## AI-first company management

** _**You are at the helm of providing cloud solutions to the EMEA region for Google Cloud. Tell us more about the industries where AI is catching on and where it isn’t.**_**

AI is catching up in any place where automation at scale with human interaction is required. Human interactions are non-deterministic and complex; humans hate to be locked into strict rules and love to challenge and understand any system. AI can provide more flexibility and can help with coping with human versatility and temper. This is critical for bank automation, contact centers, retailers, health, telecommunication, and more.  
  
In Europe, I see strong interest both from the enterprise side and from digital natives.

**_**Machine learning projects tend to be extremely different from pure software projects. What advice do you have for companies in this regard? What should they be doing differently?**_**

> Let me be provocative one more time. I disagree. I think that machine learning projects are just another type of software project.

As discussed, ML projects need a lot of software engineering, from data ingestion to model training pipelines to test and deployment of models to production. Besides that, a lot of machine learning’s principled approaches and best practices come from software engineering: make sure that you test your models; make sure that what you see in your test environment is consistent with production; make sure that you have reasonable metrics in place which describe your north star. So things are not extremely different.

Of course, it’s not all the same. For instance, in machine learning it’s good to have offline experiments—a lot of them—as you need to generate offline the best candidates, then cherry-pick among them the ones performing better in online experiments with A/B testing. That’s something that you probably weren’t doing with traditional software projects.

**_**One of the challenges we’ve seen companies run into is identifying the right problems to apply ML to. What is the best way for companies to identify opportunities to integrate ML models into their business?**_**

I think that there are two indicators. **First, metrics are a principled way to describe data problems.** Think about a situation where you have to optimize the use of electricity in a domestic environment. Think about recognizing certain types of cancer from x-rays with a desired level of accuracy. Think about a call center based on artificial intelligence that needs to recognize a human voice with a given level of accuracy. Think about translating documents across multiple languages with some expected level of quality. These are problems which can be described in terms of actionable metrics. The presence of actionable metrics is a first indicator.

**Second, it’s ideal to have data that can be used to train your models.** If you can describe your problem in terms of time series, then you probably need historical data. For example, that's what you need if your goal is to optimize electricity consumption. If you want to develop a system to recognize cancer from x-rays, then you need examples to train your models. If you need to build a system to recognize voice, then you need examples to train.   
  
If you have these two conditions, then you are in a good spot.

## Personal-philosophical

 _**Before we wrap up, I’d like to get more into fun, hypothetical territory. You once tweeted that we shouldn’t insist on defining AGI as one thing. Tell us more. How far would you say we are from AGI?**_

> Not sure why we insist in defining [#agi](https://twitter.com/hashtag/agi?src=hash&ref_src=twsrc%5Etfw) as one thing. In nature there are clearly multiple types of [#intelligence](https://twitter.com/hashtag/intelligence?src=hash&ref_src=twsrc%5Etfw). [#sleeping](https://twitter.com/hashtag/sleeping?src=hash&ref_src=twsrc%5Etfw) strategy of [#birds](https://twitter.com/hashtag/birds?src=hash&ref_src=twsrc%5Etfw)' [#flocks](https://twitter.com/hashtag/flocks?src=hash&ref_src=twsrc%5Etfw) are mesmerizing and definitely different from humans. [pic.twitter.com/CKgSujfTPA](https://t.co/CKgSujfTPA)
> 
> -- Antonio Gulli (@antoniogulli) [January 11, 2020](https://twitter.com/antoniogulli/status/1215982578450214912?ref_src=twsrc%5Etfw)

I think that we still don't have a full understanding of what intelligence is. Last month I was in Brighton, UK and I saw a very large bird flock of probably ten thousand birds dancing in the sky with perfect coordination. I was mesmerized and started to think that this is clearly a form of intelligence that we humans seem not able to achieve. The same enchanting experience happens anytime I see bees or ants and their collective efforts.

I think that we should have a better intuition of what intelligence is or, better, how many types of different intelligences there are out there. Fortunately, there are many brilliant minds in DeepMind who are thinking hard about these questions.I look with a lot of curiosity to the effort put in place by Francois Chollet with Arc. “[On the Measure of Intelligence, François Chollet, 2019](https://arxiv.org/abs/1911.01547)”. This effort is in the direction of defining and evaluating intelligence in a way that enables comparisons between two systems, as well as comparisons with humans. So François is trying to define metrics to measure and compare. Plus, Arc dataset has few examples, so it’s less likely that people can use traditional techniques like curve fitting. Francois recently tweeted “ _To be clear, if each ARC task came with 100,000 examples, you could most likely train a DL model on each task and shove ARC that way. But they don't -- you get 2-4 examples per task. So now you have to exhibit some intelligence, not just fit a curve_ ”. That’s an interesting aspect.

**_**What is your opinion about the debate between Symbolic & Connectionist AI?***_**

> It seems to me that pretty much every big AI project is a hybrid and has a combination of symbolic AI and machine learning. 

For instance, AlphaGo and the derivatives can definitely be seen as a hybrid system in this sense, a "symbolic" tree search guided by neural networks.

_**Can AI be conscious?**_

In my opinion, I’m not sure that humans have a clear sense of what consciousness is. I think that we should first understand this aspect better first. However, it’s good to see that companies like Google have shared [AI principles ](https://www.blog.google/technology/ai/ai-principles/) and are investing a lot in things like [Explainable AI](https://cloud.google.com/explainable-ai), a set of tools for deploying interpretable and inclusive machine learning models.

_**What are you hoping to learn or tackle next in your career?**_

I now work for an organization called Office of the CTO in Google Cloud and my role is at the intersection between infrastructure and machine learning. I love to co-innovate with cloud partners and, together, solve challenging problems.

In my life, I've been lucky. I’ve met inspirational people who gave me trust (still not sure why they gave me that trust), from my science professor at high school who taught me how to love science, to my colleagues in search startups back in the late 90s, to my managers and colleagues at different companies. I feel extremely thankful to all of them, and the best way to say thanks is to help others and give trust especially to people who are starting out. 

Today, I devote time to mentoring people, both inside and outside of Google, across three countries. In the future, I would love to do more in this direction.

_**Antonio, thanks so much for taking the time to chat with me today. Where can people follow or reach you out for mentoring opportunities?**_

You can connect with me on [Twitter](https://twitter.com/antoniogulli) and on [LinkedIn](https://www.linkedin.com/in/searchguy/).

* * *

_*According to what we know from the Google blog and other people involved, Google Search is actually an hybrid approach of BERT + Knowledge Graph, which is also what Gary Marcus used to refer in the AI debate as Hybrid AI_