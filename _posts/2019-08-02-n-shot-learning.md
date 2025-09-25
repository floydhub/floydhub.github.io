---
author: Heet Sankesara
date: 2019-08-02 10:03:27 +0000
excerpt: 'Is it possible to use machine learning with small data? Yes, it is! Here''s
  N-Shot Learning. '
feature_image: /assets/images/hero/n-shot-learning-hero.jpeg
layout: post
slug: n-shot-learning
tags: '[]'
title: 'N-Shot Learning: Learning More with Less Data'
---

> _Artificial Intelligence is the new electricity - Andrew NG_

If AI is the new electricity, then data is the new coal.  
Unfortunately, just as we’ve seen a hazardous depletion in the amount of available coal, many AI applications have little or no data accessible to them.  
New technology has made up for a lack of physical resources; likewise, new techniques are needed to allow applications with little data to perform satisfactorily. This is the issue at the heart of what is becoming a very popular field: N-shot Learning.

### Ready to build, train, and deploy AI?

#### Get started with FloydHub's collaborative AI platform for free

###### [Try FloydHub for free ](https://www.floydhub.com/?utm_source=blog&utm_medium=banner-n-shot-learning&utm_campaign=try_floydhub_for_free)

# N-Shot Learning

![Eminem image surrounded by the intro of Lose Yourself](https://lh6.googleusercontent.com/yDlyrtJPN2eJOht4I8qZmL3yf4En623m8WIL_1McChAhbKomSMKbLNZw2O-7hqNavEJhwGPBwLuL1RtfSH8sod4ItCqGz2eS2iI8RO-vkNKyBkeSmvgvQkNkY-tvuQum4olu1KKw)

You may be asking, what the heck is a shot, anyway? Fair question.A shot is nothing more than a single example available for training, so in N-shot learning, we have N examples for training. With the term “few-shot learning”, the “few” usually lies between zero and five, meaning that training a model with zero examples is known as zero-shot learning, one example is one-shot learning, and so on. All of these variants are trying to solve the same problem with differing levels of training material.  

### Why N-Shot?

Why do we need this when we are already getting less than a 4% error in ImageNet?

To start, ImageNet’s dataset contains a multitude of examples for machine learning, which is not always the case in fields like medical imaging, drug discovery and many others where AI could be crucially important. Typical deep learning architecture relies on substantial data for sufficient outcomes- ImageNet, for example, would need to train on hundreds of hotdog images before accurately assessing new images as hotdogs. And some datasets, much like a fridge after a 4th of July celebration, are greatly lacking in hotdogs.  

There are many use cases for machine learning where data is scarce, and that is where this technology comes in. We need to train a deep learning model which has millions or even billions of parameters, all randomly initialized, to learn to classify an unseen image using no more than 5 images. To put it succinctly, our model has to train using a very limited number of hotdog images.

To approach an issue as complex as this one, we need to first define it clearly.  
In the N-shot learning field, we have $n$ labeled examples of each $K$ classes, i.e. $N * K$ total examples which we call support set $S$ . We also have to classify Query Set $Q$, where each example lies in one of the $K$ classes. N-shot learning has three major sub-fields: zero-shot learning, one-shot learning, and few-shot learning, which each deserve individual attention.

### Zero-Shot Learning

To me, this is the most interesting sub-field. With zero-shot learning, the target is to classify unseen classes without a single training example.

How does a machine “learn” without having any data to utilize?

Think about it this way. Can you classify an object without ever seeing it?

![Constellation of Cassiopeia in the night sky](/assets/images/content/images/2019/08/cassiopea.jpg)Constellation of Cassiopeia in the night sky. [Source](https://www.star-registration.com/constellation/cassiopeia)

Yes, you can if you have adequate information about its appearance, properties, and functionality. Think back to how you came to understand the world as a kid. You could spot Mars in the night sky after reading about its color and where it would be that night, or identify the constellation Cassiopeia from only being told “it’s basically a malformed ‘W’”.

According to this year trend in NLP, [Zero shot learning will become more effective](https://floydhub.github.io/ten-trends-in-deep-learning-nlp/#9-zero-shot-learning-will-become-more-effective).

A machine utilizes the metadata of the images to perform the same task. The metadata is nothing but the features associated with the image. Here is a list of a few papers in this field which gave excellent results.

  * [Learning to Compare: Relation Network for Few-Shot Learning](https://arxiv.org/pdf/1711.06025v2.pdf)
  * [Learning Deep Representations of Fine-Grained Visual Descriptions](https://arxiv.org/pdf/1605.05395v1.pdf)
  * [Improving zero-shot learning by mitigating the hubness problem](https://arxiv.org/abs/1412.6568v3)

### One-Shot Learning

In one-shot learning, we only have a single example of each class. Now the task is to classify any test image to a class using that constraint. There are many different architectures developed to achieve this goal, such as [Siamese Neural Networks](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf), which brought about major progress and led to exceptional results, and then [matching networks](https://arxiv.org/pdf/1606.04080.pdf), which also helped us make great leaps in this field.

Now there are many excellent papers for understanding one-shot learning, as below.

  * [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/pdf/1703.03400v3.pdf)
  * [One-shot Learning with Memory-Augmented Neural Networks](https://arxiv.org/pdf/1605.06065v1.pdf)
  * [Prototypical Networks for Few-shot Learning](https://arxiv.org/pdf/1703.05175v2.pdf)

### Few-Shot Learning

Few-shot learning is just a flexible version of one-shot learning, where we have more than one training example (usually two to five images, though most of the above-mentioned models can be used for few-shot learning as well).

During the 2019 Conference on Computer Vision and Pattern Recognition, [Meta-Transfer Learning for Few-Shot Learning](https://arxiv.org/pdf/1812.02391v3.pdf) was presented. This model set the precedent for future research; it gave state-of-the-art results and paved the path for more sophisticated meta-transfer learning methods.

Many of these meta-learning and reinforcement-learning algorithms are combined with typical deep learning algorithms to produce remarkable results. Prototypical networks are one of the most popular deep learning algorithms, and are frequently used for this task.

In this article, we’ll accomplish this task using [Prototypical Networks](https://arxiv.org/pdf/1703.05175v2.pdf) and understand how it works and why it works.

## The Idea Behind Prototypical Networks

![A diagram of the function of the prototypical network. An encoder maps an image into a vector in the embedding space \(dark circles\). Support images are used to define the prototype \(stars\). Distances between prototypes and encoded query images are used to classify them. Source](https://lh5.googleusercontent.com/dM2dhO5xN_JAAtPZy4Ns5x1rBuKU-bGZl8Hj6bO71qIP-F48nsCgmaqKVtotqEmunEoyLJIUZWQ2P7l1YqglZ3_XArvZ1yyOmicJdMJ48Bzw9k9jAvRTKL4cHDpHREEM97CwDkES)A diagram of the function of the prototypical network. An encoder maps an image into a vector in the embedding space (dark circles). Support images are used to define the prototype (stars). Distances between prototypes and encoded query images are used to classify them. [Source](https://www.semanticscholar.org/paper/Gaussian-Prototypical-Networks-for-Few-Shot-on-Fort/feaecb5f7a8d29636650db7c0b480f55d098a6a7/figure/1)

Unlike typical deep learning architecture, prototypical networks do not classify the image directly, and instead learn the mapping of an image in [metric space](https://en.wikipedia.org/wiki/Metric_space).

For anyone needing a mathematics refresher, metric space deals with the notion of “distance”. It does not have a distinguished "origin" point; instead, in metric space we only compute the distance of one point to another. You therefore lack the operations of addition and scalar multiplication that you have in a vector space (because, unlike with vectors, a point only represents a coordinate, and adding two coordinates or scaling a coordinate makes no sense!). Check out [this](https://math.stackexchange.com/questions/114940/what-is-the-difference-between-metric-spaces-and-vector-spaces) link to learn more about the difference between vector space and metric space.

Now that we have that background, we can begin to understand how prototypical networks do not classify the image directly, but instead learn the mapping of an image in metric space. As can be seen in the above diagram, the encoder maps the images of the same class within tight proximity to each other, while different classes are spaced at a considerable distance. This means that whenever a new example is given, the network just checks the nearest cluster and classifies the example to its corresponding class. The underlying model in the prototypical net that maps images into metric space can be called an “Image2Vector” model, which is a Convolutional Neural Network (CNN) based architecture.

Now for those who don’t know a lot about CNNs, you can read more here:

  * Check out the list of best deep learning courses [here](https://floydhub.github.io/best-deep-learning-courses-updated-for-2019/).
  * Check out the list of best deep learning book [here](https://floydhub.github.io/best-deep-learning-books-updated-for-2019/).
  * To learn and apply it quickly refer to [Building Your First ConvNet](https://floydhub.github.io/building-your-first-convnet/)

### A brief Introduction to Prototypical Networks

Simply put, their aim is to train a classifier. This classifier can then make generalizations regarding new classes that are unavailable during training, and only needs a small number of examples of each new class. Hence, the training set contains images of a set of classes, while our test set contains images of another set of classes which is entirely disjointed from the former one. In this model, the examples are divided randomly into the support set and query set.

### Overview of Prototypical Network

![Few-shot prototypes $C_k$ are computed as the mean of embedded support examples for each class. The encoder maps new image\($X$\) and classifies it to the closest class like $C_2$ in the above image.](https://lh3.googleusercontent.com/D1r0cQ9QlrF3b-v4PlM1T_8kmdo7adxrTak5JcDZbhPxucxcdME9nHZsvC1qOtjIpj5SqcYVvw8NRrjBj9ryl6deOPJWPlOJqNnwMHM24hSOUIPgh1TkA4ZhGZTosr_PVNPk_lOj)Few-shot prototypes $C_k$ are computed as the mean of embedded support examples for each class. The encoder maps new image($X$) and classifies it to the closest class like $C_2$ in the above image. [Source](https://arxiv.org/pdf/1703.05175.pdf )

In the context of few-shot learning, a training iteration is known as an episode. An episode is nothing but a step in which we train the network once, calculate loss and backpropagate the error. In each episode, we select $N_c$ classes at random from the training set. For each class, we randomly sample $N_s$ images. These images belong to the support set and the learning model is known as $N_s$-shot model. Another randomly sampled Nq images are obtained which belongs to the query set. Here $N_c$, $N_s$ & $N_q$ are just hyperparameters in the model where $N_c$ is the number of classes per iteration, $N_s$ is the number of support examples per class and $N_q$ is the number of query examples per class.   

After that, we retrieve D-dimensional points from the support set images by passing them through “Image2Vector” model. This model encodes an image with its corresponding point in the metric space. For each class we now have multiple points, but we need to represent them as one point for each class. Hence, we compute geometric center, i.e. mean of the points, for each class. After that, we also need to classify the query images.  

To do that, we first need to encode every image in the query set into a point. After that, the distance from each centroid to each query point is calculated. At last, each query image is predicted to lie in the class which is nearest to it. That’s how the model works in general.

But the question now is, what is the architecture of this “Image2Vector” model?

### Image2Vector function

![Image2vector CNN architecture used in the paper.](https://lh4.googleusercontent.com/dupE1e1OLDG1tTrwP11axUduhgYs8jTa81rRahH9MVEIDCp67lNqSOC_q3AG1c3-fi89-fZUkH5-yfz8vYB-NNlFeET5dqCZBie__KhBGFFi52JCsw2TnEAGU8l0k1UhY1NAuJ0s)Image2vector CNN architecture used in the paper.

For all practical purposes, 4–5 CNN blocks are used. As shown in the above image, each block consists of a CNN layer followed by batch normalization, then by a ReLu activation function which leads into a max pool layer. After all the blocks, the remaining output is flattened and returned as a result. This is the architecture used in the [paper](https://arxiv.org/pdf/1703.05175v2.pdf) and you can use any architecture you like. It is necessary to know that though we call it “Image2Vector” model, it actually converts an image into a 64-dimensional point in the metric space. To understand the difference more, check out these [math stack exchange](https://math.stackexchange.com/questions/645672/what-is-the-difference-between-a-point-and-a-vector) answers.

###   
Loss function

![The working of negative log-likelihood. ](https://lh6.googleusercontent.com/yES2Ka08Hwqe59qPnmOWSyq0wdXT3s09a-g2y4RR-isjgpOCK53Wcimsqt6Leo2N8pEcKc_eblStlAyAJ9mCtNEYR1wbH_yXveCrqjdvOptjRF9qFG2Zep1iPxMKbHpXKT1zjs7Y)The working of negative log-likelihood. [Source](https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/#nll).

Now that we know how the model is working, you might be wondering how we’re going to calculate loss function. We need a loss function which is robust enough for our model to learn representation quickly and efficiently. Prototypical Nets use log-softmax loss, which is nothing but log over softmax loss. The log-softmax has the effect of heavily penalizing the model when it fails to predict the correct class, which is what we need. To know more about the loss function go[ here](https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/). [Here](https://discuss.pytorch.org/t/logsoftmax-vs-softmax/21386) is a very good discussion about softmax and log-softmax.

### Dataset overview

![A few classes of images in Omniglot dataset](https://lh5.googleusercontent.com/tA94V2yK45UQpL6JNiWqDEVqmvpzj9dajcmFrLzO1ng6OE8hNa6A1z8vaA5e9vjhZ76Cztcvy0WOuVoEhju-eMmkd0ZB47H6Be7CJ3uFCXQ_MVTTNGcm4Qs-gM3RaCD7ex-8Ipe4)A few classes of images in Omniglot dataset. [Source](https://github.com/brendenlake/omniglot).

The network was trained on the [Omniglot dataset](https://github.com/brendenlake/omniglot). The Omniglot data set is designed for developing more human-like learning algorithms. It contains 1,623 different handwritten characters from 50 different alphabets. Then, to increase the number of classes, all the images are rotated by 90, 180 and 270 degrees, with each rotation resulting in an additional class. Hence the total count of classes reached to 6,492(1,623 * 4) classes. We split images of 4,200 classes to training data while the rest went to the test set. For each episode, we trained the model on 5 examples from each of the 64 randomly selected classes. We trained our model for 1 hour and got about 88% accuracy. The official paper claimed to achieve the accuracy of 99.7% after training for a few hours and tuning a few parameters.

**Time to get your hands dirty!**

You can easily run [the code](https://github.com/Hsankesara/Prototypical-Networks) by clicking on the button below. 

[![Run on FloydHub](https://static.floydhub.com/button/button.svg)](https://floydhub.com/run?template=https://github.com/Hsankesara/Prototypical-Networks)

Let's dive into the code!
    
    
    class Net(nn.Module):
        """
        Image2Vector CNN which takes the image of dimension (28x28x3) and return column vector length 64
        """
        def sub_block(self, in_channels, out_channels=64, kernel_size=3):
            block = torch.nn.Sequential(
                torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU()
                torch.nn.MaxPool2d(kernel_size=2))
            return block
        
        def __init__(self):
            super(Net, self).__init__()
            self.convnet1 = self.sub_block(3)
            self.convnet2 = self.sub_block(64)
            self.convnet3 = self.sub_block(64)
            self.convnet4 = self.sub_block(64)
    
        def forward(self, x):
            x = self.convnet1(x)
            x = self.convnet2(x)
            x = self.convnet3(x)
            x = self.convnet4(x)
            x = torch.flatten(x, start_dim=1)
            return x
    

The above snippet is an implementation of image2vector CNN architecture. It takes images of dimensions 28x28x3 and returns a vector of length 64.
    
    
    class PrototypicalNet(nn.Module):
        def __init__(self, use_gpu=False):
            super(PrototypicalNet, self).__init__()
            self.f = Net()
            self.gpu = use_gpu
            if self.gpu:
                self.f = self.f.cuda()
        
        def forward(self, datax, datay, Ns,Nc, Nq, total_classes):
            """
            Implementation of one episode in Prototypical Net
            datax: Training images
            datay: Corresponding labels of datax
            Nc: Number  of classes per episode
            Ns: Number of support data per class
            Nq:  Number of query data per class
            total_classes: Total classes in training set
            """
            k = total_classes.shape[0]
            K = np.random.choice(total_classes, Nc, replace=False)
            Query_x = torch.Tensor()
            if(self.gpu):
                Query_x = Query_x.cuda()
            Query_y = []
            Query_y_count = []
            centroid_per_class  = {}
            class_label = {}
            label_encoding = 0
            for cls in K:
                S_cls, Q_cls = self.random_sample_cls(datax, datay, Ns, Nq, cls)
                centroid_per_class[cls] = self.get_centroid(S_cls, Nc)
                class_label[cls] = label_encoding
                label_encoding += 1
                Query_x = torch.cat((Query_x, Q_cls), 0) # Joining all the query set together
                Query_y += [cls]
                Query_y_count += [Q_cls.shape[0]]
            Query_y, Query_y_labels = self.get_query_y(Query_y, Query_y_count, class_label)
            Query_x = self.get_query_x(Query_x, centroid_per_class, Query_y_labels)
            return Query_x, Query_y
        
        def random_sample_cls(self, datax, datay, Ns, Nq, cls):
            """
            Randomly samples Ns examples as support set and Nq as Query set
            """
            data = datax[(datay == cls).nonzero()]
            perm = torch.randperm(data.shape[0])
            idx = perm[:Ns]
            S_cls = data[idx]
            idx = perm[Ns : Ns+Nq]
            Q_cls = data[idx]
            if self.gpu:
                S_cls = S_cls.cuda()
                Q_cls = Q_cls.cuda()
            return S_cls, Q_cls
        
        def get_centroid(self, S_cls, Nc):
            """
            Returns a centroid vector of support set for a class
            """
            return torch.sum(self.f(S_cls), 0).unsqueeze(1).transpose(0,1) / Nc
        
        def get_query_y(self, Qy, Qyc, class_label):
            """
            Returns labeled representation of classes of Query set and a list of labels.
            """
            labels = []
            m = len(Qy)
            for i in range(m):
                labels += [Qy[i]] * Qyc[i]
            labels = np.array(labels).reshape(len(labels), 1)
            label_encoder = LabelEncoder()
            Query_y = torch.Tensor(label_encoder.fit_transform(labels).astype(int)).long()
            if self.gpu:
                Query_y = Query_y.cuda()
            Query_y_labels = np.unique(labels)
            return Query_y, Query_y_labels
        
        def get_centroid_matrix(self, centroid_per_class, Query_y_labels):
            """
            Returns the centroid matrix where each column is a centroid of a class.
            """
            centroid_matrix = torch.Tensor()
            if(self.gpu):
                centroid_matrix = centroid_matrix.cuda()
            for label in Query_y_labels:
                centroid_matrix = torch.cat((centroid_matrix, centroid_per_class[label]))
            if self.gpu:
                centroid_matrix = centroid_matrix.cuda()
            return centroid_matrix
        
        def get_query_x(self, Query_x, centroid_per_class, Query_y_labels):
            """
            Returns distance matrix from each Query image to each centroid.
            """
            centroid_matrix = self.get_centroid_matrix(centroid_per_class, Query_y_labels)
            Query_x = self.f(Query_x)
            m = Query_x.size(0)
            n = centroid_matrix.size(0)
            # The below expressions expand both the matrices such that they become compatible with each other in order to calculate L2 distance.
            centroid_matrix = centroid_matrix.expand(m, centroid_matrix.size(0), centroid_matrix.size(1)) # Expanding centroid matrix to "m".
            Query_matrix = Query_x.expand(n, Query_x.size(0), Query_x.size(1)).transpose(0,1) # Expanding Query matrix "n" times
            Qx = torch.pairwise_distance(centroid_matrix.transpose(1,2), Query_matrix.transpose(1,2))
            return Qx
    

  
The above snippet is an implementation of a single episode in Prototypical Net. It is well commented, but if you have any doubts just ask in the comments or create an issue [here](https://github.com/Hsankesara/DeepResearch/).  

![Overview of the  Network](https://lh3.googleusercontent.com/pKUM6kIafLtbYhEd5ByNeHWsQ6YzucSqnuhuGa6uad6XZn_jj1Bv73EmxTtGXGHRZshQw5prYNTyMPjxQPBIMvWnJ9BIQLk__rKB57d4l8r9K8sypt3snt4bMhBQKRdmuK3n9YDM)Overview of the Network. [Source](https://youtu.be/wcKL05DomBU).

  
The code is structured in the same format in which the algorithm is explained. We give the prototypical network function the following inputs: input image data, input labels, number of classes per iteration i.e $N_c$ , number of support examples per class i.e $N_s$ and number of query examples per class i.e. $N_q$. The function returns $Query_x$, which is a distance matrix from each Query point to each mean point and $Query_y$ which is a vector containing labels corresponding to $Query_x$. $Query_y$ stores the class in which images of $Query_x$ actually belong. In the above image, we can see that 3 classes are used, i.e. $N_c$ =3, and that for each class, a total of 5 examples are used for training, i.e. $N_s$=5. Above $S$ represents the support set that contains those 15 ($N_s * N_c$ ) images and $X$ represents the query set. Notice that both support set and query set passes through $f$, which is nothing but our “Image2Vector” function. It mapped all the images in metric space. Let’s break the whole process down step by step.

First of all, we choose $N_c$ classes randomly from the input data. For each class, we randomly select a support set and a query set from the images using the `random_sample_cls` function. In the above image, $S$ is the support set and $X$ is the query set. Now that we chose the classes ($C_1$, $C_2$, and $C_3$), we pass all the support set examples through the “Image2vector” model and compute the centroid for each class using the `get_centroid` __ function. The same can be observed in the nearby image where $C_1$ and $C_2$ are the center, computed using the neighboring points. Each centroid represents a class and will be used for classifying queries.

![Centroid calculation in the Network](https://lh4.googleusercontent.com/34Km3uovz5Khb_On7AtmmUu1QTXQZ9sO9ekEzmMpcmD_t72RgBkkEFF3MBFXzx0Sd147s8jJLWEOBIAGjRiyJgmQ6Mff8pnNS4ZSQdGLoITuuVuAmJX3Xzj9NYgytZiAcHIIIYSJ)Centroid calculation in the Network. [Source](https://youtu.be/wcKL05DomBU).

After computing centroid for each class, we now have to predict the query image to one of the classes. For that, we need actual labels corresponding to each query, which we get by using the `get_query_y` function. The $Query_y$ is categorical data and the function converts this categorical text data into a one-hot vector, which will only be “1” in the row label where the image corresponding to the column point actually belongs, and will be “0” else in the column.

After that, we need points corresponding to each $Query_x$ image in order to classify it. We get the points using “Image2Vector” model and now we need to classify them. For that purpose, we calculate the distance between each point in $Query_x$ to each class center. This gives us a matrix where index $ij$ represents the distance of the point corresponding to ith query image from the center of jth class. We used the `get_query_x` __ function to construct the matrix and save the matrix in the $Query_x$ variable. The same can be seen in the nearby image. For each example in the query set, The distance it has from $C_1$, $C_2$ and $C_3$ is being calculated. In this case, $x$ is closest to $C_2$ and we can therefore say that $x$ is predicted to belong to class $C_2$.

Programmatically, we can use a simple argmin function to do the same, i.e. to find out the class where the image was predicted to lie. Then we use the predicted class and actual class to calculate loss and backpropagate the error.

If you want to use the trained model or just have to retrain again for yourself, [here](https://github.com/Hsankesara/DeepResearch/tree/master/Prototypical_Nets) is my implementation. You can use it as an API and train the model using a couple of lines of code. You can find this network in action [here](https://www.kaggle.com/hsankesara/prototypical-net/).

### Resources

Here are a few resources that might help you learn this topic thoroughly:

  * [One Shot Learning with Siamese Networks using Keras](https://sorenbouma.github.io/blog/oneshot/)
  * [One-Shot Learning: Face Recognition using Siamese Neural Network](https://towardsdatascience.com/one-shot-learning-face-recognition-using-siamese-neural-network-a13dcf739e)
  * [Matching network official implementation](https://github.com/AntreasAntoniou/MatchingNetworks)
  * [Prototypical Network official implementation.](https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch)
  * [Meta-Learning for Semi-Supervised Few-Shot Classification](https://arxiv.org/abs/1803.00676)

### Limitations

Though prototypical networks produce great results, they still have limitations. The first one is the lack of generalization. It works on the Omniglot dataset well because all the images in there are images of a character, and hence share a few similar characteristics. However, if we were to try using the model to classify different breeds of cats, it wouldn’t give us accurate results. Cats and character images share few characteristics, and the number of common features which can be exploited to map the image on the corresponding metric space is negligible.

Another limitation to prototypical networks is that they only use mean to decide center, and ignore the variance in support set. This hinders the classifying ability of the model when the images have noise. This limitation is overcome by using [Gaussian Prototypical Networks](https://arxiv.org/abs/1708.02735) which utilizes the variance in the class by modeling the embedded points using Gaussian formulations.

###   
Conclusion

Few-Shot learning has been a topic of active research for a while. There are many novel approaches which use prototypical networks, like this [meta-learning](https://arxiv.org/abs/1803.00676) one, and which show great results. Researchers are also exploring it with reinforcement-learning, which also has great potential. The best thing about this model is that it is simple and easy to understand, and it gives incredible results.

* * *

## ******FloydHub Call for AI writers******

Want to write amazing articles like Heet and play your role in the long road to Artificial General Intelligence? [We are looking for passionate writers](https://floydhub.github.io/write-for-floydhub/?utm_source=floydhub&utm_medium=banner&utm_campaign=call_for_writers_2019), to build the world's best blog for practical applications of groundbreaking A.I. techniques. FloydHub has a large reach within the AI community and with your help, we can inspire the next wave of AI. [Apply now](https://goo.gl/forms/PbOw0VmUnOfO1Lxp1) and join the crew!

* * *

**About Heet Sankesara**

Heet is a passionate and diligent machine learning researcher who loves working on messy datasets to solve intricate problems. He is also a dedicated blogger who reads up on new concepts and works to introduce them to readers in the most engaging way possible. Heet is always up for discussing new ideas or even old ideas and their impacts and implications. This tutorial is the third article in his [DeepResearch](https://github.com/Hsankesara/DeepResearch) series.  
If you like this tutorial please let him know in comments, and if you don’t then please let him know in comments more briefly. You can find the rest of his blogs [here](https://medium.com/@heetsankesara3). Heet is also a [FloydHub AI Writer](https://floydhub.github.io/write-for-floydhub/). You can connect with him on [GitHub](https://github.com/hsankesara), [Twitter](https://twitter.com/thesankesara), [Linkedin](https://www.linkedin.com/in/heet-sankesara-72383a152/), and [Kaggle](https://www.kaggle.com/hsankesara/).