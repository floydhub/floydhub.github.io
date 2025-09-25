---
author: Charlie Harrington
date: 2018-06-19 17:04:34 +0000
excerpt: Generate your own commencement speeches using Markov chains in Python on
  FloydHub.
feature_image: /assets/images/hero/markov-chains-hero.jpg
layout: post
slug: markov-chains
tags: '[]'
title: Generating Commencement Speeches With Markov Chains
---

Imagine this. You're the founder _slash_ CEO _slash_ product-visionary of a three month-old electric scooter startup. In between transforming the global transportation market and dabbling in your first angel investments (you know, just to get your feet wet), you've been asked by your beloved alma mater to deliver this year's commencement address to the graduating class.

That's right, you unfortunately _didn't_ drop out of college, so you've already lost that convenient narrative thread. You agree to give the speech and assure yourself you'll be fine. You've got no problem waxing on about local city scooter politics or the unit-economics of the secondary charging market. That should give you a good five minutes or so. Maybe you can even ride a scooter on stage towards the podium for a laugh? Write that down, you mutter to no one in particular.

But as the graduation date approaches, the pressure's rising. Your plucky Chief-of-Staff _slash_ travel agent _slash_ only real friend anymore asks you how the speech draft's going, and you just smile and nod, "Great, Sam. It's about connecting the dots. In reverse."

"You mean like that Steve Jobs one? Stanford, 2005. I've watched the YouTube video like a million times."

"Oh, no, not like that. It's more about the benefits of failure and, you know, the importance of imagination."

"J.K. Rowling, Harvard, 2008. C'mon, you're not going to make that _world's largest Gryffindor reunion_ joke, too. Are you? Say no, please. Say no right now."

"No, course not. Anyway, isn't it about time for my daily transcendental gratitude journaling? You almost made me miss it again. Give me one of your pens. Not that one, the other one."

You sit down at the communal lunch table, a glass of ginger green-tea kombucha and a terrifying blank piece of paper in front of you.

Right as – you swear – you were about to start writing the greatest commencement address of all time, one of those newfangled data scientists walks over and sits down, directly across from you. Can't they see that you're _in-the-zone_? And why are you paying them so much if they're just sitting around all the time?

"I think I can help you."

You glance up at them, barely.

"I overheard your conversation with Sam. Don't give me that look, it's an open office layout – probably your idea, too. Anyway, I think I can help you out. You need a speech. A good one. And quickly."

You drop your (Sam's) pen on the table, cross your arms, and lean backwards.

"I'm listening."

"Have you heard of Markov chains?"

## A little bit of Markov in your life

In this post, I'll show you how you can easily generate your own overwrought and highly-sentimental commencement address clichés using a simple [Markov chain library](https://github.com/jsvine/markovify) in Python and an open dataset of commencement speeches on FloydHub.

In just a few minutes, you'll be spouting gems like:

> The secret to success in start-ups, or any other collaboration, is to stick an old head on a motorcycle weaving his way down the hall, he passed a door – it was empty.

Or perhaps this wisdom nugget:

> Think of your ancestors: Among them, for everybody here, among your ancestors and I even thought about running away from the old ones.

### Try it now

[ ![Run](https://static.floydhub.com/button/button.svg) ](https://floydhub.com/run?template=https://github.com/whatrocks/markov-commencement-speech)

Click this button to open a [Workspace](https://docs.floydhub.com/guides/workspace/) on FloydHub where you can train a Markov chain model to generate "commencement speech style" sentences in a live JupyterLab environment. The [commencement address dataset](https://floydhub.com/whatrocks/datasets/commencement) of ~300 famous speeches will be automatically attached and available in the Workspace. Just follow along with the `speech_maker` Jupyter notebook. It's that easy, folks.

### Ready to build, train, and deploy AI?

#### Get started with FloydHub's collaborative AI platform for free

###### [Try FloydHub for free ](https://www.floydhub.com/?utm_source=blog&utm_medium=banner&utm_campaign=try_floydhub_for_free)

#### But what's a Markov chain?

A Markov chain, named after [this bearded devil](https://en.wikipedia.org/wiki/Andrey_Markov), is a model describing a sequence of states (these states which could be some situation, or set of values, or, in our case, a word in a sentence) along with the probability of moving from one state to any other state.

The best explainer I've found for Markov chains is the [visual explainer produced by Victor Powell and Lewis Lehe](http://setosa.io/blog/2014/07/26/markov-chains/). Stop what you're doing right now and read their post.

Good, you're back. 

As you just learned, Markov chains are popular modeling tools in a variety of industries where people need to model impossibly-large real world systems – finance, environmental science, computer science. Powell and Lehe point out that Google's big bad PageRank algorithm is a form of Markov chain. So they're, like, a big deal.

But Markov chains have also found a nice sweet spot in the text generation realm of natural language processing (NLP). Or, in other words, they're perfect for creating Captain Picard Twitter bots.

> Number One, I've just been paid a visit from Q. He wants to look me in the eye.
> 
> -- MarkovPicard (@MarkovPicard) [July 7, 2017](https://twitter.com/MarkovPicard/status/883430429373730821?ref_src=twsrc%5Etfw)

## Generating the best speech ever

In our case, we want to use a Markov chain to generate random sentences based on a corpus of famous commencement speeches. 

Luckily, there's a simple Python library for that first part. It's called [markovify](https://github.com/jsvine/markovify). I'll show you how to use it in just a second.

First, we need to get some speech transcripts. Ah, data — the cause of and solution to all of deep learning's problems.

NPR's [The Best Commencement Speeches, Ever](https://apps.npr.org/commencement/), a site I frequent often, was a great starting point. From there, some [BeautifulSoup scraping](https://www.crummy.com/software/BeautifulSoup/) – along with a shocking amount of manual spam removal from the transcripts found on popular commencement speech sites (ugh, don't ask) – led to the assemblage of a dataset containing ~300 plaintext commencement speech transcripts. 

The [commencement speech dataset](https://www.floydhub.com/whatrocks/datasets/commencement) is publicly available on FloydHub so that you can use it your own projects. I've also put together a simple Gatsby.js static site if you'd like to [casually read the speeches](https://whatrocks.github.io/commencement-db/) at your leisure.

Okay, back to business. As mentioned, the markovify library is insanely easy to use. Let me remind you again to just click the "Run on FloydHub" button above to follow along in the `speech_maker` notebook. Actually, here it is again:

[ ![Run](https://static.floydhub.com/button/button.svg) ](https://floydhub.com/run?template=https://github.com/whatrocks/markov-commencement-speech)

To generate our sentences, we're going to iterate through all the speeches in our dataset (available at the `/floyd/input/speeches` path) and create a Markov model for each speech. 
    
    
    import os
    import markovify
    
    SPEECH_PATH = '/floyd/input/speeches/'
    
    speech_dict = {}
    for speech_file in os.listdir(SPEECH_PATH):
        with open(f'{SPEECH_PATH}{speech_file}') as speech:
            contents = speech.read()
    
            # Create a Markov model for each speech in our dataset
            model = markovify.Text(contents)
            speech_dict[speech_file] = model
    

Then, we'll use a markovify's `combine` method to combine them into one large Markov chain.
    
    
    models = list(speech_dict.values())
    
    # Combine the Markov models
    model_combination = markovify.combine(models)
    

Finally, we'll generate our random sentence:
    
    
    print(model_combination.make_sentence())
    

> Certainly I could do the most elegant and extraordinary products in federally funded highway projects.

You may have noticed that I've organized the individual speech models into a dictionary with the speech's filename as keys. Why did you do that, dummy? Well, if you keep following along with the `speech_maker` notebook, you'll see that this helps you more easily filter the speeches as you keep experimenting.

For example, maybe you only want to generate sentences from speeches delivered at Stanford University? Or at MIT? Or only from speeches delivered in the 1980s?

The workspace contains a CSV with metadata for each speech (speaker `name`, `school`, and `year`). This is your chance to make a dent in the commencement speech universe. Oh, here's an idea: why don't you brush off your Simpson's TV script Recurrent Neural Network (RNN) code and give it a whirl on this dataset?

![](/assets/images/content/images/2018/06/12.jpg)May I see it? No.

## Add 'Run on FloydHub' to your own projects

It can be a real pain in the you-know-what organizing your own machine learning experiments, let alone trying to share your work with others. Many folks wiser than I (or is it me?) have acknowledged the reproducibility crisis in data science.

The "Run on FloydHub" button is here to help. With this button, we're making it just a little bit easier to reproduce and share your data science projects. 

Now, you can simply add this button to your repos on GitHub and anyone will be able to spin up a Workspace on FloydHub along with your code, datasets, deep learning framework, and any other environment configs.

Here's what you need to do:

###### 1\. Create a floyd.yml config file in your repo
    
    
    machine: gpu
    env: pytorch-1.4

###### 2\. Add this snippet to your README
    
    
    <a href="https://floydhub.com/run">
        <img src="https://static.floydhub.com/button/button.svg" alt="Run">
    </a>

###### 3\. You're done! 🎉

Seriously. Try it out right now on with our [Object Classification repo](https://github.com/floydhub/image-classification-template). Or our [Sentiment Analysis repo](https://github.com/floydhub/sentiment-analysis-template). Or even [this post's project](https://github.com/whatrocks/markov-commencement-speech).

**The real magic is when you also include the required datasets in your config file**. For example, the `floyd.yml` config file for the Sentiment Analysis project looks like this: 
    
    
    env: tensorflow-1.7
    machine: cpu
    data:
      - source: floydhub/datasets/imdb-preprocessed/1
        destination: imdb

This will spin up your code in a CPU-powered Workspace using Tensorflow 1.7 as well as attach [the IMDB Dataset](https://www.floydhub.com/floydhub/datasets/imdb-preprocessed/1). This makes it insanely easy for anyone to reproduce your experiments, right from your GitHub README.

You can [read more](https://docs.floydhub.com/guides/run_on_floydhub_button/) about the Run on FloydHub button in our docs. We're looking forward to seeing what you share with the world! Good luck, graduates!