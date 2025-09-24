---
layout: "post"
title: "Creating Datasets from Public URLs"
date: "2017-11-04 04:38:00 +0000"
slug: "creating-datasets-from-public-urls"
author: "Charlie Harrington"
excerpt: "We're trying to make it easier to discover and use interesting datasets on
FloydHub. To that end, we just released an improved file-viewer to help you dig
in and explore data directly on FloydHub.

Fo..."
feature_image: "https://images.unsplash.com/photo-1451187580459-43490279c0fa?ixlib=rb-0.3.5&q=80&fm=jpg&crop=entropy&cs=tinysrgb&w=1080&fit=max&ixid=eyJhcHBfaWQiOjExNzczfQ&s=163e3ea37d2c0fda3f586d5552752f59"
tags: "[]"
---

We're trying to make it easier to discover and use interesting datasets on FloydHub. To that end, we just released an improved file-viewer to help you dig in and explore data directly on FloydHub.

For example, here's one of my [favorite pups](https://www.floydhub.com/fastai/projects/lesson1_dogs_cats/13/data/data/train/dogs/dog.10701.jpg) from the [Kaggle Cats vs. Dogs dataset](https://www.floydhub.com/fastai/datasets/cats-vs-dogs).

![](/assets/images/content/images/2018/04/fh-dog-1.png)

Lots of work still to be done here - like CSV viewers and more - but we're hoping this already makes your deep learning life a little bit more fun.

We also spruced up our [Explore](https://www.floydhub.com/explore/trending) page last week. We're now featuring collections of the projects and datasets you'll need to survive the [Udacity Deep Learning Nanodegree](https://www.floydhub.com/explore/courses/udacity-deep-learning) and [Fast.ai Part 1](https://www.floydhub.com/explore/courses/fast-ai-part-1) courses. And more collections are coming soon (hello, [deeplearning.ai](https://www.deeplearning.ai/) \- we're on to you!)

But - if there's **one thing** you've told us **loud and clear** through our [forum](https://forum.floydhub.com) \- it's that you'd like an faster way to create FloydHub datasets directly from public URLs.

And guess what? We agree. Yes, yes - one might argue (and we've tried) that it's technically possible to do this right now, but, really, the process is sort-of (most definitely) annoying. You need to first download the dataset directly to your machine - which might take forever if it's large enough - and then upload to back to FloydHub - which might also take forever.

There's got to be another way.

**Today, we're releasing the first step towards a brighter dataset future - creating datasets directly from your FloydHub job outputs.**

What the - how does this help?

### Let me explain. Work with me here.

Let's say you've found a great dataset - like a list of the [current members of the United States Congress](https://github.com/unitedstates/congress-legislators) \- and you'd like to turn this CSV into a FloydHub dataset.

Now, before we continue, it's probably worth rehashing why we would even want to create a dataset on FloydHub. Seems like a hassle. Well, creating a dataset on FloydHub has [many benefits](https://docs.floydhub.com/getstarted/core_concepts/#why-keep-datasets-separate-from-code) \- but, primarily, it boils down being able to easily use the data in future GPU-powered training jobs on FloydHub. That's why we're here in the first place!

Okay, next up - let's create a new project on FloydHub called `washington` and fire up a fresh Jupyter Notebook session from the command line using the `floyd-cli` tool.
    
    
    $ floyd init washington
    $ floyd run --mode jupyter
    

Our new Jupyter notebook session should open up automatically in our browser. Once we're inside, let's first head over to the Jupyter terminal to grab the CSV data.

![](/assets/images/content/images/2018/04/initial-1.png)

Let's take a look at where we are.
    
    
    $ pwd
    /output
    $ ls
    command.sh
    

When running in Jupyter mode, FloydHub automatically places us in the `/output` directory of our Jupyter notebook instance. Let's create a new folder called `/congress` to hold our data, and we'll use `wget` to fetch the CSV to our Jupyter instance.
    
    
    $ mkdir congress
    $ cd congress 
    $ wget https://theunitedstates.io/congress-legislators/legislators-current.csv
    

Great! Now that we've grabbed the CSV, we can either stop the entire job right now and create a dataset right away - or we can open up a notebook to play around with the data, make sure it's what we want, and even clean it up a bit before saving it as a dataset. Let's do the latter.

Open up a new Python notebook from the main Jupyter session window.

![](/assets/images/content/images/2018/04/notebook-1.png)

Next, let's explore our data a bit:
    
    
    import pandas as pd
    import matplotlib.pyplot as plt
    %matplotlib inline
    congress = pd.read_csv("congress/legislators-current.csv")
    

How about we take a quick peek at our data:
    
    
    congress.head(5)
    

![](/assets/images/content/images/2018/04/head-1.png)

I don't like the looks of that `district` column - there's a lot of missing values. We could try replacing or transforming these missing values, but - for now - let's just drop it entirely from our dataset:
    
    
    congress.drop(["district"], axis=1, inplace=True)
    

Before we go, let's try a chart to make sure we're looking at whole picture.
    
    
    parties = congress.groupby('party')
    parties.size().plot(kind='bar')
    

![](/assets/images/content/images/2018/04/parties-1.png)

Ah, 2017. Finally, let's save our file so that we don't lose our changes to our data (remember, we dropped the `district` column):
    
    
    congress.to_csv("congress/legislators-current.csv")
    

Now, you can `Save and Checkpoint` your notebook, and then stop the running CPU job from FloydHub by clicking Cancel. Then you can head over to the `Output` tab of your job, and click the new `Create Dataset` button.

![](/assets/images/content/images/2018/04/output-1.png)

A real friendly-looking modal will pop up and ask which dataset you'd like to use - you can either add to one of your existing datasets or create a nice, new, fresh one right here.

![](/assets/images/content/images/2018/04/modal-1.png)

Once that's done, you'll be whisked away to your newly created dataset on FloydHub, now populated with that public URL CSV data you wanted so badly.

![](/assets/images/content/images/2018/04/dataset-1.png)

We've made it. You've done it. Now you can reference this new dataset the next time you're running a job!

## Whole lotta steps

Yes, this is still cumbersome - but this should make things much **faster** since you no longer need to download datasets locally to your machine. Let us know what you think! There's a feedback button on the output modal with a short survey, or just hit us up on [Twitter](https://www.twitter.com/FloydHub_) or in the forum.