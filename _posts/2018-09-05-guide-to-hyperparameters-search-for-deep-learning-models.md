---
layout: "post"
title: "Practical Guide to Hyperparameters Optimization for Deep Learning Models"
date: "2018-09-05 18:24:41 +0000"
slug: "guide-to-hyperparameters-search-for-deep-learning-models"
author: "Alessio Gozzoli"
excerpt: "Learn techniques for identifying the best hyperparameters for your deep learning projects, including code samples that you can use to get started on FloydHub."
feature_image: "https://images.unsplash.com/photo-1509581376349-c9994910b6c0?ixlib=rb-0.3.5&q=80&fm=jpg&crop=entropy&cs=tinysrgb&w=1080&fit=max&ixid=eyJhcHBfaWQiOjExNzczfQ&s=ee30f25b817d82c89ee060cda00f900f"
tags: "[]"
---

> Are you tired of babysitting your DL models? If so, you're in the right place. In this post, we discuss motivations and strategies behind effectively searching for the best set of hyperparameters for any deep learning model. We'll demonstrate how this can be done on FloydHub, as well as which direction the research is moving. When you're done reading this post, you'll have added some powerful new tools to your data science tool-belt – making the process of finding the best configuration for your deep learning task as automatic as possible. 

Unlike machine learning models, deep learning models are literally full of hyperparameters. Would you like some evidence? Just take a look at the [Transformer base v1 hyperparameters definition](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py#L1467-L1525).

I rest my case. 

Of course, not all of these variables contribute in the same way to the model's learning process, but, given this additional complexity, **it's clear that finding the best configuration for these variables in such a high dimensional space is not a trivial challenge**. 

Luckily, we have different strategies and tools for tackling the searching problem. Let's dive in!

### Ready to build, train, and deploy AI?

#### Get started with FloydHub's collaborative AI platform for free

###### [Try FloydHub for free ](https://www.floydhub.com/?utm_source=blog&utm_medium=banner&utm_campaign=try_floydhub_for_free)

# Our Goal

**How?**

We want to find the _best configuration_ of hyperparameters which will give us the _best score_ on the metric we care about on the validation / test set. 

**Why?**

Every scientist and researcher wants the best model for the task given the available resources: 💻, 💰 and ⏳ (aka _compute_ ,_money_ , and _time_). Effective hyperparameter search is the missing piece of the puzzle that will help us move towards this goal. 

**When?**

  * It's quite common among researchers and hobbyists to try one of these searching strategies during the _last steps of development_. This helps provide possible improvements from the best model obtained already after several hours of work.
  * Hyperparameter search is also common as a stage or component in a _semi/fully automatic deep learning pipeline_. This is, obviously, more common among data science teams at companies. 

# Wait, but [what exactly are hyperparameters](https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/)?

Let's start with the simplest possible definition,

> _**Hyperparameters** are the knobs that you can turn when building your machine / deep learning model. _

![](/assets/images/content/images/2018/08/Screen-Shot-2018-08-22-at-17.59.25.png)Hyperparameters - the "knobs" or "dials" metaphor

Or, alternatively:

> **Hyperparameters** are __ all the training variables set manually with a pre-determined value before starting the training.

We can likely agree that the Learning Rate and the Dropout Rate are considered hyperparameters, but what about the model design variables? These include embeddings, number of layers, activation function, and so on. Should we consider these variables as hyperparameters?

![](/assets/images/content/images/2018/08/Screen-Shot-2018-08-22-at-18.32.53.png)Model Design Variables + Hyperparameters → Model Parameters

For simplicity's sake, yes – we can also consider the **model design components** as part of the hyperparameters set.

Finally, how about the parameters obtained from the training process – the variables learned from the data? These weights are known as **model parameters**. We'll exclude them from our hyperparameter set.

Okay, let's try a real-world example. Take a look at the picture below for an example illustrating the different classifications of variables in a deep learning model.

![](/assets/images/content/images/2018/08/Screen-Shot-2018-08-22-at-18.42.51.png)Variables classification example

# Our next problem: searching is expensive

Now that we know we want to search for the best configuration of hyperparameters, we're faced with the challenge that _searching for hyperparameters is an_ _iterative process_ constrained by **💻** ,**💰** and**⏳**.

![](/assets/images/content/images/2018/08/Screen-Shot-2018-08-22-at-19.06.53.png)The hyperparameters search cycle

Everything starts with a guess (_step 1_) of a promising configuration, then we will need to wait until a full training (_step 2_) to get the actual evaluation on the metric of interest (_step 3_). We'll track the progress of the searching process (_step 4_), and then according to our searching strategy, we'll select a new guess (_step 1_). 

We'll keep going like this until we reach a terminating condition (such as running out of **⏳** or **💰**). 

# Let's talk strategies

We have four main strategies available for searching for the best configuration.

  * **Babysitting (aka Trial & Error) **
  * **Grid Search**
  * **Random Search**
  * **Bayesian Optimization**

# Babysitting

Babysitting is also known as _Trial & Error_ or _Grad Student Descent_ in the academic field. This approach is **100% manual** and the most widely adopted by researchers, students, and hobbyists. 

The end-to-end workflow is really quite simple: a student devises a new experiment that she follows through all the steps of the learning process (from data collection to feature map visualization), then will she iterates sequentially on the hyperparameters until she runs out time (usually due to a deadline) or motivation. 

![](/assets/images/content/images/2018/08/Screen-Shot-2018-08-23-at-14.59.35.png)Babysitting

If you've enrolled in the deeplearning.ai course, then you're familiar with this approach - it is [the Panda workflow described by Professor Andrew Ng](https://www.coursera.org/lecture/deep-neural-network/hyperparameters-tuning-in-practice-pandas-vs-caviar-DHNcc).

This approach is very educational, but it doesn't scale inside a team or a company where the time of the data scientist is really valuable. 

Thus, we arrive at the question:

> **_“Is there a better way to invest my_ time?_”_**

Surely, yes! We can optimize your time by defining an automatic strategy for hyperparameter searching!

# Grid Search

Taken from the imperative command "Just try everything!" comes [Grid Search](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search) – a naive approach of simply trying every possible configuration.

Here's the workflow:

  * Define a grid on _n_ dimensions, where each of these maps for an hyperparameter. e.g. _n_ = (learning_rate, dropout_rate, batch_size)
  * For each dimension, define the range of possible values: e.g. batch_size = [4, 8, 16, 32, 64, 128, 256]
  * Search for all the possible configurations and wait for the results to establish the best one: e.g. _C1_ = (0.1, 0.3, 4) -> acc = 92%, _C2_ = (0.1, 0.35, 4) -> acc = 92.3%, etc...

The image below illustrates a simple grid search on two dimensions for the Dropout and Learning rate. 

![](/assets/images/content/images/2018/08/Screen-Shot-2018-08-23-at-15.33.56.png)Grid Search on two variables in a parallel concurrent execution

This strategy is embarrassingly parallel because it doesn't take into account the computation history (we will expand this soon). But what it does mean is that the more computational resources **💻** you have available, then the more guesses you can try at the same time!

The real pain point of this approach is known as the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality). This means that more dimensions we add, the more the search will explode in time complexity (usually by an exponential factor), ultimately making this strategy unfeasible!

It's common to use this approach when the dimensions are less than or equal to 4. But, in practice, even if it guarantees to find the best configuration _at the end, it's still not preferable._ Instead, it's better to use [Random Search](https://en.wikipedia.org/wiki/Random_search) — which we'll discuss next _._

### Try grid search now!

Click this button to open a [Workspace](https://floydhub.github.io/workspaces/) on [FloydHub](https://www.floydhub.com/?utm_medium=readme&utm_source=hyperparameters_search_examples&utm_campaign=sept_2018). You can use the workspace to run the code below (Grid Search using Scikit-learn and Keras) on a fully configured cloud machine. 

[ ![Run](https://static.floydhub.com/button/button.svg) ](https://floydhub.com/run?template=https://github.com/floydhub/hyperparameters-search-examples)
    
    
    # Load the dataset
    x, y = load_dataset()
    
    # Create model for KerasClassifier
    def create_model(hparams1=dvalue,
                     hparams2=dvalue,
                     ...
                     hparamsn=dvalue):
        # Model definition
        ...
    
    model = KerasClassifier(build_fn=create_model) 
    
    # Define the range
    hparams1 = [2, 4, ...]
    hparams2 = ['elu', 'relu', ...]
    ...
    hparamsn = [1, 2, 3, 4, ...]
    
    # Prepare the Grid
    param_grid = dict(hparams1=hparams1, 
                      hparams2=hparams2, 
                      ...
                      hparamsn=hparamsn)
    
    # GridSearch in action
    grid = GridSearchCV(estimator=model, 
                        param_grid=param_grid, 
                        n_jobs=, 
                        cv=,
                        verbose=)
    grid_result = grid.fit(x, y)
    
    # Show the results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    

# Random Search

A few years ago, Bergstra and Bengio published [an amazing paper](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) where they demonstrated the inefficiency of Grid Search.

The only real difference between Grid Search and Random Search is on the _step 1_ of the strategy cycle – Random Search picks the point randomly from the configuration space. 

Let's use the image below (provided in the paper) to show the claims reported by the researchers.

![](/assets/images/content/images/2018/08/Screen-Shot-2018-08-24-at-17.09.14.png)Grid Search vs Random Search

The image compares the two approaches by searching the best configuration on two hyperparameters space. It also assumes that one parameter is more important that the other one. This is a safe assumption because Deep Learning models, as mentioned at the beginning, are really full of hyperparameters, and usually the researcher / scientist / student knows which ones affect the training most significantly.

In the Grid Layout, it's easy to notice that, even if we have trained 9 models, we have used only 3 values per variable! Whereas, with the Random Layout, it's extremely unlikely that we will select the same variables more than once. It ends up that, with the second approach, we will have trained 9 models using 9 different values for each variable. 

As you can tell from the space exploration at the top of each layout in the image, we have explored the hyperparameters space more widely with Random Search (especially for the more important variables). This will help us to find the best configuration in fewer iterations.

In summary: Don't use Grid Search if your searching space contains more than 3 to 4 dimensions. Instead, use Random Search, which provides a really good baseline for each searching task.

![](/assets/images/content/images/2018/08/Screen-Shot-2018-08-24-at-18.13.43.png)Pros and cons of Grid Search and Random Search

### Try Random Search now!

[ ![Run](https://static.floydhub.com/button/button.svg) ](https://floydhub.com/run?template=https://github.com/floydhub/hyperparameters-search-examples)

Click this button to open a [Workspace](https://floydhub.github.io/workspaces/) on [FloydHub](https://www.floydhub.com/?utm_medium=readme&utm_source=hyperparameters_search_examples&utm_campaign=sept_2018). You can use the workspace to run the code below (Random Search using Scikit-learn and Keras.) on a fully configured cloud machine. 
    
    
    # Load the dataset
    X, Y = load_dataset()
    
    # Create model for KerasClassifier
    def create_model(hparams1=dvalue,
                     hparams2=dvalue,
                     ...
                     hparamsn=dvalue):
        # Model definition
        ...
    
    model = KerasClassifier(build_fn=create_model) 
    
    # Specify parameters and distributions to sample from
    hparams1 = randint(1, 100)
    hparams2 = ['elu', 'relu', ...]
    ...
    hparamsn = uniform(0, 1)
    
    # Prepare the Dict for the Search
    param_dist = dict(hparams1=hparams1, 
                      hparams2=hparams2, 
                      ...
                      hparamsn=hparamsn)
    
    # Search in action!
    n_iter_search = 16 # Number of parameter settings that are sampled.
    random_search = RandomizedSearchCV(estimator=model, 
                                       param_distributions=param_dist,
                                       n_iter=n_iter_search,
                                       n_jobs=, 
    								   cv=, 
    								   verbose=)
    random_search.fit(X, Y)
    
    # Show the results
    print("Best: %f using %s" % (random_search.best_score_, random_search.best_params_))
    means = random_search.cv_results_['mean_test_score']
    stds = random_search.cv_results_['std_test_score']
    params = random_search.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    

## One step back, two steps forward

As an aside, when you need to set the space for each dimension, _it's very important to use the right scale per each variables._

![](/assets/images/content/images/2018/08/Screen-Shot-2018-08-24-at-17.44.08.png) Common scale space for batch size and learning rate 

For example, it's common to use values of [batch size as a power of 2](https://datascience.stackexchange.com/questions/20179/what-is-the-advantage-of-keeping-batch-size-a-power-of-2) and sample the learning rate in the log scale.

![](/assets/images/content/images/2018/08/Screen-Shot-2018-08-24-at-18.03.49.png)Zoom In!

It's also very common to start with one of the layouts above for a certain number of iterations, and then _zoom into_ a promising subspace by sampling more densely in each variables range, and even starting a new search with the same or a different searching strategy.

## Yet another problem: independent guesses!

Unfortunately, both Grid and Random Search share the common downside:

> **_“Each new guess is independent from the previous run!”_**

It can sound strange and surprising, but what makes Babysitting effective – despite the amount of time required – is the ability of the scientist to drive the search and experimentation effectively by using the past as a resource to improve the next runs. 

Wait a minute, this sounds familiar... what if we try to model the hyperparameter search as a machine learning task?! 

Allow me to introduce [Bayesian Optimization](https://en.wikipedia.org/wiki/Bayesian_optimization).

# Bayesian Optimization

This search strategy builds a surrogate model that tries to predict the metrics we care about from the hyperparameters configuration. 

At each new iteration, the surrogate we will become more and more confident about which new guess can lead to improvements. Just like the other search strategies, it shares the same termination condition. 

![](/assets/images/content/images/2018/08/Screen-Shot-2018-08-24-at-18.21.48.png)Bayesian Opt Workflow

If this sounds confusing right now, don't worry – it's time for another visual example.

## The Gaussian Process in action

We can define the [Gaussian Process](https://en.wikipedia.org/wiki/Gaussian_process) as the surrogate that will learn the mapping from hyperparameters configuration to the metric of interest. It will not only produce the prediction as a value, but it will also give us the range of uncertainty (mean and variance). 

Let's dive into the [example provided by this great tutorial.](https://www.iro.umontreal.ca/~bengioy/cifar/NCAP2014-summerschool/slides/Ryan_adams_140814_bayesopt_ncap.pdf)

![](/assets/images/content/images/2018/08/step1.png)Gaussian Process in action with 2 Points

In the above image, we are following the first steps of a Gaussian Process optimization on a single variable (on the horizontal axes). In our imaginary example, this can represent the learning rate or dropout rate.

On the vertical axes, we are plotting the metrics of interest as a function of the single hyperparameter. Since we're looking for the lowest possible value, we can think of it as the loss function. 

The black dots represent the model trained so far. The red line is the ground truth, or, in other words, the function that we are trying to learn. The black line represents the mean of the actual hypothesis we have for the ground truth function and the grey area shows the related uncertainty, or variance, in the space. 

As we can notice, the uncertainty diminishes around the dots because we are quite confident about the results we can get around these points (since we've already trained the model here). The uncertainty, then, increases in the areas where we have less information.

Now that we've defined the starting point, we're ready to choose the next promising variables on which train a model. For doing this, we need to define an _acquisition function_ which will tell us where to sample the next configuration. 

In this example, we are using the _Expected Improvement_ : a function that is aiming to find the lowest possible value if we will use the proposed configuration from the uncertainty area. The blue dot in the Expected Improvement chart above shows the point selected for the next training. 

![](/assets/images/content/images/2018/08/step2.png)Gaussian Process in action with 3 Points

The more models we train, the more confident the surrogate will become about the next promising points to sample. Here's the chart after 8 trained models:

![](/assets/images/content/images/2018/08/step8.png)Gaussian Process in action with 8 Points

The Gaussian Process falls under the class of algorithms called _Sequential Model Based Optimization (SMBO)_. As we've just seen, these algorithms provide a really good baseline to start the search for the best hyperparameter configuration. But, just like every tool, they come with their downsides:

  * By definition, the process is sequential
  * It can only handle numeric parameters
  * It doesn't provide any mechanism to stop the training if it's performing poorly

Please note that we've really just scratched the surface about this fascinating topic, and if you're interested in a more detailed reading and how to extend SMBO, then take a look at [this paper](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf).

### Try Bayesian Optimization now!

[ ![Run](https://static.floydhub.com/button/button.svg) ](https://floydhub.com/run?template=https://github.com/floydhub/hyperparameters-search-examples)

Click this button to open a [Workspace](https://floydhub.github.io/workspaces/) on [FloydHub](https://www.floydhub.com/?utm_medium=readme&utm_source=hyperparameters_search_examples&utm_campaign=sept_2018). You can use the workspace to run the code below (Bayesian Optimization (SMBO - TPE) using [Hyperas](https://github.com/maxpumperla/hyperas)) on a fully configured cloud machine.
    
    
    def data():
        """
        Data providing function:
        This function is separated from model() so that hyperopt
        won't reload data for each evaluation run.
        """
        # Load / Cleaning / Preprocessing
        ...
        return x_train, y_train, x_test, y_test
        
    def model(x_train, y_train, x_test, y_test):
        """
        Model providing function:
        Create Keras model with double curly brackets dropped-in as needed.
        Return value has to be a valid python dictionary with two customary keys:
            - loss: Specify a numeric evaluation metric to be minimized
            - status: Just use STATUS_OK and see hyperopt documentation if not feasible
        The last one is optional, though recommended, namely:
            - model: specify the model just created so that we can later use it again.
        """
        # Model definition / hyperparameters space definition / fit / eval
        return {'loss': <metrics_to_minimize>, 'status': STATUS_OK, 'model': model}
        
    # SMBO - TPE in action
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=,
                                          trials=Trials())
    
    # Show the results
    x_train, y_train, x_test, y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(x_test, y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    

# Search strategy comparison

It's finally time to summarize what we've covered so far to understand the strengths and weakness of each proposal.

![](/assets/images/content/images/2018/08/Screen-Shot-2018-08-30-at-14.50.17.png)Summary

Bayes SMBO is probably the best candidate as long as resources are not a constraint for you or your team, but you should also consider establishing a baseline with Random Search.

On the other hand, if you're still learning or in the development phase, then babysitting – even if unpractical in term of space exploration – is the way to go.

Just like I mentioned in the SMBO section, none of these strategies provide a mechanism to save our resources if a training is performing poorly or even worse diverging – we'll have to wait until the end of the computation. 

Thus, we arrive at the last question of our fantastic quest:

> **_“Can we optimize the training time?”_**

Let's find out.

# The power of stopping earlier

![](/assets/images/content/images/2018/08/Screen-Shot-2018-08-24-at-19.05.21.png)If I could only help him to stop!

[Early Stopping](https://en.wikipedia.org/wiki/Early_stopping) is not only a famous [regularization](https://en.wikipedia.org/wiki/Regularization_\(mathematics\)) technique, but it also provides a great mechanism for preventing a waste of resources when the training is not going in the right direction.

Here's a diagram of the most adopted stopping criteria:

![](/assets/images/content/images/2018/08/Screen-Shot-2018-08-24-at-19.09.10.png)Stopping criteria

The first three criteria are self-explanatory, so let's focus our attention to the last one.

It's common to cap the training time according to the class of experiment inside the research lab. This policy acts as a funnel for the experiments and optimizes for the resources inside the team. In this way, we will be able to allocate more resources only to the most promising experiments. 

The `floyd-cli` (the software used by our users to communicate with FloydHub and that we've [open-sourced on Github](https://github.com/floydhub/floyd-cli)) provides a flag with this purpose: our power users are using it massively to regulate their experiments.

These criteria can be applied manually when babysitting the learning process, or you can do even better by integrated these rules in your experiment through the hooks/callbacks provided in the most common frameworks:

  * [Keras](https://github.com/keras-team/keras) provides a great [EarlyStopping](https://keras.io/callbacks/#earlystopping) function and even better a suite of super useful [callbacks](https://keras.io/callbacks/). Since Keras has been recently integrated inside TensorFlow, you will be able to use the [callbacks inside your TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback) code.
  * [TensorFlow](https://www.tensorflow.org/) provides the [Training Hooks](https://www.tensorflow.org/api_guides/python/train#Training_Hooks), these are probably not intuitive as Keras callbacks (or the tf.keras API), but they provides you more control over the state of the execution. TensorFlow 2.0 (currently in beta) introduces a new API for managing hyperparameters optimization, you can find more info in the [official TensorFlow docs](https://www.tensorflow.org/tensorboard/r2/hyperparameter_tuning_with_hparams).
  * There is even more in the TensorFlow/Keras realm! The Keras team has just released an [hyperparameter tuner for Keras](https://github.com/keras-team/keras-tuner), specifically for `tf.keras` with TensorFlow 2.0.

> Keras Tuner is now out of beta! v1 is out on PyPI.<https://t.co/riqnIr4auA>  
>   
> Fully-featured, scalable, easy-to-use hyperparameter tuning for Keras & beyond. [pic.twitter.com/zUDISXPdBw](https://t.co/zUDISXPdBw)
> 
> -- François Chollet (@fchollet) [October 31, 2019](https://twitter.com/fchollet/status/1189992078991708160?ref_src=twsrc%5Etfw)

🎉

  * At this time, [PyTorch](https://pytorch.org/) hasn't yet provided a hooks or callbacks component, but you can check the [TorchSample](https://github.com/ncullen93/torchsample) repo and in the amazing [Forum](https://discuss.pytorch.org/).
  * The [fast.ai](http://www.fast.ai/) library provides [callbacks](https://github.com/fastai/fastai/blob/649e5f72981e5ef714d7709598c8b3b64f6d905f/fastai/sgdr.py) too, you can find more info in the official [fastai callbacks doc page](https://docs.fast.ai/callbacks.html). If you are lost or need some help, I strongly recommend you to reach the [amazing fast.ai community.](http://forums.fast.ai/)
  * [Ignite](https://github.com/pytorch/ignite) (high-level library of PyTorch) provides callbacks similarly to Keras. The library is actually under active development but it certainly seems a really interesting option. 

I will stop the list here to limit the discussion to the most used / trending frameworks (I hope to not have hurt the sensibility of the other frameworks' authors. If so, you can direct your complaints to me and I'll be happy to update the content!)

### This is not the end. 

There is a subfield of machine learning called “AutoML” (Automatic Machine Learning) which aims to automate methods for model selection, features extraction and / or hyperparameters optimization. 

This tool is the answer to the last question (I promise!):

> **_“Can we**learn** the whole process?”_**

You can think of AutoML as Machine Learning task which is solving another Machine Learning task, similar to what we've done with the Baeysian Optimiziation. Essentially, this is Meta-Machine Learning.

# Research: AutoML and PBT

You have most likely heard of [Google's AutoML](https://ai.googleblog.com/2018/03/using-evolutionary-automl-to-discover.html) which is their re-branding for [Neural Architecture Search](https://en.wikipedia.org/wiki/Neural_architecture_search). Remember, all the way at the beginning of the article, we decided to merge the _model design component_ into the hyperparameters variables? Well, Neural Architecture Search is the subfield of AutoML which aims to find the best models for a given task. A full discussion on this topic would require a series of articles. Luckily, Dr. Rachel Thomas at fast.ai did an amazing job that [we are happy to link!](http://www.fast.ai/2018/07/12/auto-ml-1/)

I would like to share with you another interesting [research effort from DeepMind](https://deepmind.com/blog/population-based-training-neural-networks/) where they used a variant of Evolution Strategy algorithm to perform hyperparameters search called Population Based Training (PTB is also at the foundation of another [amazing research from DeepMind](https://deepmind.com/blog/capture-the-flag/) which wasn't quite covered from the press but that I strongly encourage you to check out on your own). Quoting DeepMind:

> PBT - like random search - starts by training many neural networks in parallel with random hyperparameters. But instead of the networks training independently, it uses information from the rest of the population to refine the hyperparameters and direct computational resources to models which show promise. This takes its inspiration from genetic algorithms where each member of the population, known as a worker, can exploit information from the remainder of the population. For example, a worker might copy the model parameters from a better performing worker. It can also explore new hyperparameters by changing the current values randomly.

Of course, there are probably tons of other super interesting researches in this area. I've just shared with you the ones who gained some recent prominence in the news. 

# Managing your experiments on FloydHub

One of the biggest features of FloydHub is the ability to compare different model you're training when using a different set of hyperparameters. 

The picture below shows a list of jobs in a FloydHub project. You can see that this user is using the job's `message` field (e.g. `floyd run --message "SGD, lr=1e-3, l1_drop=0.3" ...` ) to highlight the hyperparameters used on each of these jobs.

Additionally, you can also see the [training metrics](https://floydhub.github.io/metrics-on-floydhub/) for each job. These offer a quick glance to help you understand which of these jobs performed best, as well as the type of machine used and the total training time.

![](/assets/images/content/images/2018/08/Screen-Shot-2018-08-24-at-19.10.18.png)Project Page

The FloydHub dashboard gives you an easy way to compare all the training you've done in your hyperparameter optimization – and it updates in real-time.

Our advice is to create a different FloydHub project for each of the tasks/problems you have to solve. In this way, it's easier for you to organize your work and collaborate with your team.

### Training metrics

As mentioned above, you can easily emit [training metrics](https://floydhub.github.io/metrics-on-floydhub/) with your jobs on FloydHub. When you view your job on the FloydHub dashboard, you'll find real-time charts for each of the metrics you've defined. 

This feature is not intended to substitute [Tensorboard](https://floydhub.github.io/tensorboard-on-floydhub/) (we provides this feature as well), but instead aims to highlight the behavior of your training given the configuration of hyperparameters you've selected. 

For example, if you're babysitting the training process, then the training metrics will certainly help you to determine and apply the stopping criteria.

![](/assets/images/content/images/2018/08/Screen-Shot-2018-08-24-at-19.13.18.png)Training metrics

# FloydHub HyperSearch (Coming soon!)

We are currently planning to release some examples of how to wrap the `floyd-cli` command line tool with these proposed strategies to effectively run hyperparameters search on FloydHub. **So, stay tuned!**

#### One last thing! 

Some of the FloydHub users have asked for a simplified hyperparameter search solution (similar to the solution proposed in the [Google Vizier](https://ai.google/research/pubs/pub46180) paper) inside FloydHub. If you think this would be useful to you, please let us know by contacting our [support](mailto:support@floydhub.com) or by posting on the [Forum](https://forum.floydhub.com/).

We're really excited to improve FloydHub to meet all your training needs!

* * *

### Do you model for living? 👩‍💻 🤖 Be part of a ML/DL user research study and get a cool AI t-shirt every month 💥

  
We are looking for _full-time data scientists_ for a ML/DL user study. You'll be participating in a calibrated user research experiment for 45 minutes. The study will be done over a video call. We've got plenty of funny tees that you can show-off to your teammates. We'll ship you a different one every month for a year!

Click [here](https://typings.typeform.com/to/zpYrlW?utm_source=blog&utm_medium=bottom_text_hypersearch&utm_campaign=full_time_ds_user_study) to learn more.

* * *

#### **FloydHub Call for AI writers**

Want to write amazing articles like Alessio and play your role in the long road to Artificial General Intelligence? [We are looking for passionate writers](https://floydhub.github.io/write-for-floydhub/?utm_source=floydhub&utm_medium=banner&utm_campaign=call_for_writers_2019), to build the world's best blog for practical applications of groundbreaking A.I. techniques. FloydHub has a large reach within the AI community and with your help, we can inspire the next wave of AI. [Apply now](https://goo.gl/forms/PbOw0VmUnOfO1Lxp1) and join the crew!