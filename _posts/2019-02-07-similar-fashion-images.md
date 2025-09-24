---
layout: "post"
title: "Recommending Similar Fashion Images with Deep Learning"
date: "2019-02-07 14:37:11 +0000"
slug: "similar-fashion-images"
author: "James Le"
excerpt: "Explore how deep learning is changing the fashion industry by training your own visual recommendation model for similar fashion images using TensorFlow and FloydHub"
feature_image: "https://images.unsplash.com/photo-1416339698674-4f118dd3388b?ixlib=rb-1.2.1&q=80&fm=jpg&crop=entropy&cs=tinysrgb&w=1080&fit=max&ixid=eyJhcHBfaWQiOjExNzczfQ"
tags: "[]"
---

Within a few years, machine learning will completely change the fashion industry. Fashion brands from small to big are already using machine learning techniques to predict and design what you’ll be wearing next year, next week, even tomorrow.

**Stitch Fix** is already at the forefront of AI-driven fashion with its hybrid design garments, which are created by [algorithms](https://algorithms-tour.stitchfix.com/) that identify trends and styles missing from the Stitch Fix inventory and suggest new designs for human designers’ approval.

![](/assets/images/content/images/2019/02/fashio-1.jpeg)Stitch Fix Algorithm

**Rent the Runway** is another company that [utilizes AI in the fashion realm](http://dresscode.renttherunway.com/who-we-are/). The company provides rentals of designer dresses to those who might otherwise not be able to afford them or may not need them long-term. They use extensive recommendation systems and machine learning algorithms to suggest dresses for their customers based on their profiles and optimize the delivery time.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_26D8191B6F13CAB8D7A633E957B00DECC09CB28AECA93C7EDFEB07C504C27EFB_1547048222791_rent-the-runway.jpg)Rent the Runway

**Pinterest** has also done some amazing computer vision work in visual search, where fashion is a popular use case. [The Pinterest Lens product](https://about.pinterest.com/en/lens) allows users to discover ideas without having to find the right words to describe them first. For example, you can point the Lens at a pair of shoes, then tap to see related styles or even ideas for what else to wear them with.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_26D8191B6F13CAB8D7A633E957B00DECC09CB28AECA93C7EDFEB07C504C27EFB_1547047412629_Pinterest-Lens.jpg)Pinterest Lens

Within the larger artificial intelligence realm, computer vision is an important area of focus for fashion products, because a user’s buying decision is primarily influenced by the product’s visual appearance. Deep learning added a huge boost to this already rapidly developing field. With deep learning, a lot of new applications of computer vision techniques have been introduced and are now becoming parts of everyday lives ([facial recognition](https://floydhub.github.io/build-image-classification-app-with-fastai/), photo stylization, [autonomous vehicles](https://floydhub.github.io/toy-self-driving-car-part-one/)).

In this post, we will build a model that is capable of doing large-scale visual recommendation. If you’re not familiar with that term, a **visual recommendation** model is one that can incorporate visual signals directly into the recommendation objective. Essentially, a user interested in buying a particular item from the screen may want to explore visually similar items before finishing her purchase. These could be items with similar colors, patterns, and shapes.

More specifically, we will design a model that takes a fashion image as input (the image on the left below), and outputs a few most similar pictures of clothes in a given dataset of fashion images (the images on the right side). 

![](https://d2mxuefqeaa7sj.cloudfront.net/s_26D8191B6F13CAB8D7A633E957B00DECC09CB28AECA93C7EDFEB07C504C27EFB_1548006497421_Romper-Examples.png)An example top-5 result on the romper category![](https://d2mxuefqeaa7sj.cloudfront.net/s_26D8191B6F13CAB8D7A633E957B00DECC09CB28AECA93C7EDFEB07C504C27EFB_1548006511799_Hoodies-Example.png)An example top-5 result on the hoodies category

All the code is prepared on [GitHub](https://github.com/khanhnamle1994/fashion-recommendation) and FloydHub. You can follow along with the code in this post by clicking this button to open the code and datasets in a Jupyter Workspace on FloydHub:

[![Run on FloydHub](https://static.floydhub.com/button/button.svg)](https://floydhub.com/run?template=https://github.com/khanhnamle1994/fashion-recommendation)

The architecture is based on the “[Identity Mapping in Deep Residual Networks](https://arxiv.org/abs/1603.05027)” paper by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. The code is written in Python and uses TensorFlow, a popular deep learning framework.

If you’re new to deep learning, computer vision and TensorFlow, I’d recommend getting a feel for them by checking out these tutorials I have written elsewhere [[1](https://medium.com/cracking-the-data-science-interview/the-10-deep-learning-methods-ai-practitioners-need-to-apply-885259f402c1)] [[2](https://heartbeat.fritz.ai/the-5-computer-vision-techniques-that-will-change-how-you-see-the-world-1ee19334354b)] [[3](https://heartbeat.fritz.ai/the-5-deep-learning-frameworks-every-serious-machine-learner-should-be-familiar-with-93f4d469d24c)].

## Datasets

In this project, I work with the [DeepFashion dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html), which is collected by researchers in the Chinese Hong Kong University. It has over 800,000 diverse fashion images and rich annotations with additional information about landmarks, categories, pairs etc. The dataset consists of 5 different kinds of predicting subsets that are tailored towards their specific tasks.

One subset, called [Attribute Prediction](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html), can be used for clothing category and attribute prediction. With close to 290,000 images of 50 clothing categories and 1,000 clothing attributes, this subset is ideal for our experiment.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_26D8191B6F13CAB8D7A633E957B00DECC09CB28AECA93C7EDFEB07C504C27EFB_1548007391133_deep-fashion-attribute-prediction.jpg)DeepFashion Attribute Prediction Subset

We will only use the upper body clothes images due to the limitation of computation resources and time.

We changed the old labels of 6 categories and randomly picked 3,000 images from each category to have evenly distributed labels, as shown in the table below. After this step, we have 18,000 images in total. A random 1,500 of them are used as validation set and the others are used as train set.

Category | Old Label | New Label | Total  
---|---|---|---  
Blouses Shirts | 10 | 0 | 3,000  
Dresses | 13 | 1 | 3,000  
Hoodies | 8 | 2 | 3,000  
Jackets Vests | 2 | 3 | 3,000  
Rompers Jumpsuits | 18 | 4 | 3,000  
Sweaters | 7 | 5 | 3,000  
  
## Model Architecture

Deep neural networks are perfect tools to map an image to a vector that ignores the irrelevant details.

We will train such neural networks to classify the clothing images into 6 categorical labels and use the feature layer as the deep features of the images. The feature layer will be able to capture features of the clothes, like the categories, fabrics, and patterns. We do that by searching for nearest neighbors based on the feature layer.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_26D8191B6F13CAB8D7A633E957B00DECC09CB28AECA93C7EDFEB07C504C27EFB_1548363494057_architecture.png)Model Workflow

## ResNet Model

### ResNet Architecture

To classify the images, we use a model based on deep residual networks ([ResNet](https://arxiv.org/abs/1603.05027)). ResNet is characterized by the **residual block** structure. This incorporates identity shortcut connections**** which essentially skip the training of one or more layers.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_26D8191B6F13CAB8D7A633E957B00DECC09CB28AECA93C7EDFEB07C504C27EFB_1548017592160_residual-block.png)Residual Block

The residual block is further optimized by a **pre-activation module**. This allows the gradients to propagate through the shortcut connections to any of the earlier layers without hindrance. Instead of starting with a convolution (_weight_), we start with a series of `(BN=> RELU => CONV) * N` layers (assuming bottleneck is being used). Then, the residual module outputs the _addition_ operation __ that’s fed into the next residual module in the network (since residual modules are stacked on top of each other).

![](https://d2mxuefqeaa7sj.cloudfront.net/s_26D8191B6F13CAB8D7A633E957B00DECC09CB28AECA93C7EDFEB07C504C27EFB_1548017773002_pre-activation-module.png)(a) original bottleneck residual module.(e) full pre-activation residual module. Called pre-activation because BN and ReLU layers occur before the convolutions.

The overall network architecture looked like below, and our model will be similar to it. We represent each image with the values of the feature layer (the global pool layer), as this layer captures the most detailed information of the images.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_26D8191B6F13CAB8D7A633E957B00DECC09CB28AECA93C7EDFEB07C504C27EFB_1548017821255_simple-resnet.gif)Full ResNet Architecture

### ResNet Implementation

Let’s look at the implementation of the ResNet architecture in TensorFlow.

The `create_variables` function takes in `name` (the name of the variable), `shape` (a list of dimensions), `initializer` (Xavier is the default option), and `is_fc_layer` (set to be False). It returns the created variable new_variables.
    
    
    def create_variables(name,
                         shape,
                         initializer=tf.contrib.layers.xavier_initializer(),     
                         is_fc_layer=False):
        if is_fc_layer is True:
            regularizer =  tf.contrib.layers.l2_regularizer(scale=FLAGS.fc_weight_decay)
        else:
            regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)
        
    
    	new_variables = tf.get_variable(name,
    					shape=shape,
    					initializer=initializer,
    					regularizer=regularizer)
    	return new_variables

The `output_layer` function takes in `input_layer` (a 2D tensor) and `num_labels` (the number of output labels). It returns the output layers `fc_h` and `fc_h2`.
    
    
    def output_layer(input_layer, num_labels):
        input_dim = input_layer.get_shape().as_list()[-1]
        
        fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], 
        is_fc_layer=True, initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        
        fc_b = create_variables(name='fc_bias', shape=[num_labels], 
        initializer=tf.zeros_initializer)
        
        fc_w2 = create_variables(name='fc_weights2', shape=[input_dim, 4], 
        is_fc_layer=True, initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        
        fc_b2 = create_variables(name='fc_bias2', shape=[4], 
        initializer=tf.zeros_initializer)
        
        fc_h = tf.matmul(input_layer, fc_w) + fc_b
        fc_h2 = tf.matmul(input_layer, fc_w2) + fc_b2
        return fc_h, fc_h2

The `conv_bn_relu_layer` function applies convolution, batch normalization and ReLU to the input tensor sequentially. It takes in `input_layer` (a 4D tensor), `filter_shape` (a list that contains [filter_height, filter_width, filter_depth, filter_number]), and `stride` (the stride size for our convolution). It returns a 4D tensor `output`.
    
    
    def conv_bn_relu_layer(input_layer, filter_shape, stride, 
                            second_conv_residual=False, relu=True):
        out_channel = filter_shape[-1]
        if second_conv_residual is False:
            filter = create_variables(name='conv', shape=filter_shape)
        else: filter = create_variables(name='conv2', shape=filter_shape)
    
        conv_layer = tf.nn.conv2d(input_layer, filter, 
                                  strides=[1, stride, stride, 1], padding='SAME')
        mean, variance = tf.nn.moments(conv_layer, axes=[0, 1, 2])
    
        if second_conv_residual is False:
            beta = tf.get_variable('beta', out_channel, tf.float32,
                                   initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable('gamma', out_channel, tf.float32,
                                    initializer=tf.constant_initializer(1.0, tf.float32))
        else:
            beta = tf.get_variable('beta_second_conv', out_channel, tf.float32,
                                   initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable('gamma_second_conv', out_channel, tf.float32,
                                    initializer=tf.constant_initializer(1.0, tf.float32))
    
        bn_layer = tf.nn.batch_normalization(conv_layer, mean, variance, 
                                              beta, gamma, BN_EPSILON)
        if relu:
            output = tf.nn.relu(bn_layer)
        else:
            output = bn_layer
        return output

The `bn_relu_conv_layer` function applies batch normalization, ReLU and convolution to the input layer sequentially. The inputs and output are similar to that of `conv_bn_relu_layer`.
    
    
    def bn_relu_conv_layer(input_layer, filter_shape, stride, 
                            second_conv_residual=False):
        in_channel = input_layer.get_shape().as_list()[-1]
        mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    
        if second_conv_residual is False:
            beta = tf.get_variable('beta', in_channel, tf.float32,
                                   initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable('gamma', in_channel, tf.float32,
                                    initializer=tf.constant_initializer(1.0, tf.float32))
        else:
            beta = tf.get_variable('beta_second_conv', in_channel, tf.float32,
                                   initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable('gamma_second_conv', in_channel, tf.float32,
                                    initializer=tf.constant_initializer(1.0, tf.float32))
    
        bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, 
                                              beta, gamma, BN_EPSILON)
        relu_layer = tf.nn.relu(bn_layer)
    
        if second_conv_residual is False:
            filter = create_variables(name='conv', shape=filter_shape)
        else: 
            filter = create_variables(name='conv2', shape=filter_shape)
        
        conv_layer = tf.nn.conv2d(relu_layer, filter, 
                                  strides=[1, stride, stride, 1], padding='SAME')
        return conv_layer

The `residual_block_new` function defines a residual block in ResNet. It takes in `input_layer` (a 4D tensor), `output_channel` (the shape of our output tensor), and `first_block` (whether or not this is the first residual block of our network). It returns a 4D tensor `output`.
    
    
    def residual_block_new(input_layer, output_channel, first_block=False):
        input_channel = input_layer.get_shape().as_list()[-1]
    
        if input_channel * 2 == output_channel:
            increase_dim = True
            stride = 2
        elif input_channel == output_channel:
            increase_dim = False
            stride = 1
        else:
            raise ValueError('Output and input channel does not match in residual block')
    
        if first_block:
            filter = create_variables(name='conv', 
                                      shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(input_layer, filter=filter, 
                                  strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, 
                                        [3, 3, input_channel, output_channel], stride)
        
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1,
                                   second_conv_residual=True)
        
        if increase_dim is True:
            pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='SAME')
            padded_input = tf.pad(pooled_input, 
            [[0, 0], [0, 0], [0, 0], [input_channel // 2, input_channel // 2]])
        else:
            padded_input = input_layer 
               
        output = conv2 + padded_input
        return output

The main `inference` function defines ResNet. It takes in `input_tensor_batch` (a 4D tensor), `n` (the number of residual blocks), `reuse` (setting it to be True if we want to build a train graph, False if we want to build a validation graph and share weights with a train graph). It returns the last layer in the network.
    
    
    def inference(input_tensor_batch, n, reuse, keep_prob_placeholder):
        layers = []
        with tf.variable_scope('conv0', reuse=reuse):
            conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 16], 1)
            layers.append(conv0)
    
        for i in range(n):
            with tf.variable_scope('conv1_%d' %i, reuse=reuse):
                if i == 0:
                    conv1 = residual_block_new(layers[-1], 16, first_block=True)
                else:
                    conv1 = residual_block_new(layers[-1], 16)
                layers.append(conv1)
    
        for i in range(n):
            with tf.variable_scope('conv2_%d' %i, reuse=reuse):
                conv2 = residual_block_new(layers[-1], 32)
                layers.append(conv2)
    
        for i in range(n):
            with tf.variable_scope('conv3_%d' %i, reuse=reuse):
                conv3 = residual_block_new(layers[-1], 64)
                layers.append(conv3)
            assert conv3.get_shape().as_list()[1:] == [16, 16, 64]
    
        with tf.variable_scope('fc', reuse=reuse):
            in_channel = layers[-1].get_shape().as_list()[-1]
            mean, variance = tf.nn.moments(layers[-1], axes=[0, 1, 2])
            beta = tf.get_variable('beta', in_channel, tf.float32,
                                   initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable('gamma', in_channel, tf.float32,
                                    initializer=tf.constant_initializer(1.0, tf.float32))
            bn_layer = tf.nn.batch_normalization(layers[-1], mean, variance, 
                                                  beta, gamma, BN_EPSILON)
            relu_layer = tf.nn.relu(bn_layer)
            global_pool = tf.reduce_mean(relu_layer, [1, 2])
            assert global_pool.get_shape().as_list()[-1:] == [64]
            cls_output, bbx_output = output_layer(global_pool, NUM_LABELS)
            layers.append(cls_output)
            layers.append(bbx_output)
            
        return cls_output, bbx_output, global_pool

## Image Pre-Processing

We need to pre-process our images before they are suitable for training. Here are the hyper-parameters that are arbitrarily set.
    
    
    shuffle = True
    localization = FLAGS.is_localization
    imageNet_mean_pixel = [103.939, 116.799, 123.68]
    global_std = 68.76
    IMG_ROWS = 64
    IMG_COLS = 64

The code below reads in the path to the images and resizes them to 64 x 64 dimension.
    
    
    def get_image(path, x1, y1, x2, y2):
        img = cv2.imread(path)
        if localization is True:
            if img is None or img.shape[0] == 0 or img.shape[1] == 0:
                img = np.zeros((1, IMG_ROWS, IMG_COLS, 0))
            img = cv2.resize(img, (IMG_ROWS, IMG_COLS))
            assert img.shape == (IMG_ROWS, IMG_COLS, 3)
        else:
            img = cv2.resize(img, (IMG_ROWS, IMG_COLS))
        img = img.reshape(1, IMG_ROWS, IMG_COLS, 3)
        return img

The code below loads the data NumPy arrays and whitens them with global average pixel ([103.939, 116.799, 123.68]) and the global standard deviation (68.76). We also augment the data by randomly flipping the images horizontally.
    
    
    def load_data_numpy(df):
        num_images = len(df)
        image_path_array = df['image_path'].as_matrix()
        label_array = df['category'].as_matrix()
        x1 = df['x1_modified'].as_matrix().reshape(-1, 1)
        y1 = df['y1_modified'].as_matrix().reshape(-1, 1)
        x2 = df['x2_modified'].as_matrix().reshape(-1, 1)
        y2 = df['y2_modified'].as_matrix().reshape(-1, 1)
        bbox_array = np.concatenate((x1, y1, x2, y2), axis=1)
    
        image_array = np.array([]).reshape(-1, IMG_ROWS, IMG_COLS, 3)
        adjusted_std = 1.0/np.sqrt(IMG_COLS * IMG_ROWS * 3)
    
        for i in range(num_images):
            img = get_image(image_path_array[i], 
            x1=x1[i, 0], y1=y1[i, 0], x2=x2[i, 0], y2=y2[i, 0])
            flip_indicator = np.random.randint(low=0, high=2)
            if flip_indicator == 0:
                img[0, ...] = cv2.flip(img[0, ...], 1)
            image_array = np.concatenate((image_array, img))
        image_array = (image_array - imageNet_mean_pixel) / global_std
        # Convert to BGR image for ResNet
        assert image_array.shape[1:] == (IMG_ROWS, IMG_COLS, 3)
        return image_array, label_array, bbox_array

## Training

### Helper Functions For Training

We write a **prepare_df** function, which takes the path of a csv file and its column as inputs, and then returns a Pandas dataframe as output.
    
    
    def prepare_df(path, usecols, shuffle=shuffle):
        df = pd.read_csv(path, usecols=usecols)
        if shuffle is True:
            order = np.random.permutation(len(df))
            df = df.iloc[order, :]
        return df

We also write a **loss** function, which takes in labels, logits, bounding boxes and their labels as inputs, and returns a sum loss of cross entropy loss and mean squared error loss.
    
    
    def loss(self, logits, bbox, labels, bbox_labels):
            labels = tf.cast(labels, tf.int64)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
            mse_loss = tf.reduce_mean(tf.losses.mean_squared_error(bbox_labels, bbox), name='mean_square_loss')
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
            return cross_entropy_mean + mse_loss

We also need a **top_k_error** function, which takes the predictions, labels, and arbitrary k value as inputs, and returns the top k error value.
    
    
    def top_k_error(self, predictions, labels, k):
            batch_size = predictions.get_shape().as_list()[0]
            in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
            num_correct = tf.reduce_sum(in_top1)
            return (batch_size - num_correct) / float(batch_size)

Lastly, we also write a couple of operations function to be applied to the training and validation dataset.
    
    
    def train_operation(self, global_step, total_loss, top1_error):
            tf.summary.scalar('learning_rate', self.lr_placeholder)
            tf.summary.scalar('train_loss', total_loss)
            tf.summary.scalar('train_top1_error', top1_error)
    
            ema = tf.train.ExponentialMovingAverage(0.95, global_step)
            train_ema_op = ema.apply([total_loss, top1_error])
            tf.summary.scalar('train_top1_error_avg', ema.average(top1_error))
            tf.summary.scalar('train_loss_avg', ema.average(total_loss))
    
            opt = tf.train.MomentumOptimizer(learning_rate=self.lr_placeholder, momentum=0.9)
            train_op = opt.minimize(total_loss, global_step=global_step)
            
            return train_op, train_ema_op
            
    def validation_op(self, validation_step, top1_error, loss):
            ema = tf.train.ExponentialMovingAverage(0.0, validation_step)
            ema2 = tf.train.ExponentialMovingAverage(0.95, validation_step)
            val_op = tf.group(validation_step.assign_add(1), ema.apply([top1_error, loss]), ema2.apply([top1_error, loss]))
            
            top1_error_val = ema.average(top1_error)
            top1_error_avg = ema2.average(top1_error)
            loss_val = ema.average(loss)
            loss_val_avg = ema2.average(loss)
            
            tf.summary.scalar('val_top1_error', top1_error_val)
            tf.summary.scalar('val_top1_error_avg', top1_error_avg)
            tf.summary.scalar('val_loss', loss_val)
            tf.summary.scalar('val_loss_avg', loss_val_avg)
            
            return val_op

### Training Main Function

Let’s move on to the actual training step. All of the code below will be included in a **train** function. First, we apply the prepare_df function into our train and validation data.
    
    
    train_df = prepare_df(FLAGS.train_path, usecols=['image_path', 'category', 'x1_modified', 'y1_modified', 'x2_modified', 'y2_modified'])
    
    vali_df = prepare_df(FLAGS.vali_path, usecols=['image_path', 'category', 'x1_modified', 'y1_modified', 'x2_modified', 'y2_modified'])

Then, we define a couple of important hyper-parameters to be used during training with TensorFlow. Below are the number of training samples, global step and validation step (which refer to the number of batches used during training).
    
    
    num_train = len(train_df)
    global_step = tf.Variable(0, trainable=False)
    validation_step = tf.Variable(0, trainable=False)

`logits` and `vali_logits` are the output of ResNet before going through the softmax function. `bbox` and `vali_bbox` are the bounding boxes of the images. These variables to perform inference on the test data.
    
    
    logits, bbox, _ = inference(self.image_placeholder, n=FLAGS.num_residual_blocks, reuse=False,keep_prob_placeholder=self.dropout_prob_placeholder)
    
    vali_logits, vali_bbox, _ = inference(self.vali_image_placeholder, n=FLAGS.num_residual_blocks, reuse=True, keep_prob_placeholder=self.dropout_prob_placeholder)

My loss function combines the regularization loss and the multi-label classification loss.
    
    
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = self.loss(logits, bbox, self.label_placeholder, self.bbox_placeholder)
    full_loss = tf.add_n([loss] + reg_losses)

Here are the variables for output predictions and top-1 error results.
    
    
    predictions = tf.nn.softmax(logits)
    top1_error = self.top_k_error(predictions, self.label_placeholder, 1)
    vali_loss = self.loss(vali_logits, vali_bbox, self.vali_label_placeholder, self.vali_bbox_placeholder)
    vali_predictions = tf.nn.softmax(vali_logits)
    vali_top1_error = self.top_k_error(vali_predictions, self.vali_label_placeholder, 1)

Here are the variables for training and validation operations.
    
    
    train_op, train_ema_op = self.train_operation(global_step, full_loss, top1_error)
    val_op = self.validation_op(validation_step, vali_top1_error, vali_loss)

The empty lists below are initialized to keep track of the training steps, training errors, and validation errors. min_error is an arbitrary variable to maintain the current minimum error value.
    
    
    step_list = []
    train_error_list = []
    vali_error_list = []
    min_error = 0.5

All the code below lies inside a `for` loop that iterates through all the steps: `for step in range(STEP_TO_TRAIN)`. Let’s define a couple of variables: 

  * `offset` is used to limit the batch size.
  * `train_batch_df` is the NumPy data array that contains training data batch. 
  * The function `load_data_numpy` is called on `train_batch_df` to return the 3 NumPy arrays of training batch, training batch labels, and training batch bounding boxes.
  * The function `generate_validation_batch` is called on validation data to return the 3 NumPy arrays of validation batch, validation batch labels, and validation batch bounding boxes.

    
    
    offset = np.random.choice(num_train - TRAIN_BATCH_SIZE, 1)[0]
    train_batch_df = train_df.iloc[offset:offset+TRAIN_BATCH_SIZE, :]
    batch_data, batch_label, batch_bbox = load_data_numpy(train_batch_df)
    vali_image_batch, vali_labels_batch, vali_bbox_batch = generate_validation_batch(vali_df)

The code below calculates the top 1 error value and loss value for validation data.
    
    
    if step == 0:
      if FULL_VALIDATION is True:
        top1_error_value, vali_loss_value = self.full_validation(vali_df, sess=sess, 
        vali_loss=vali_loss, vali_top1_error=vali_top1_error, batch_data=batch_data, 
        batch_label=batch_label, batch_bbox=batch_bbox)
        
        vali_summ = tf.Summary()
        vali_summ.value.add(tag='full_validation_error',
                            simple_value=top1_error_value.astype(np.float))
        vali_summ.value.add(tag='full_validation_loss',
                            simple_value=vali_loss_value.astype(np.float))
        summary_writer.add_summary(vali_summ, step)
        summary_writer.flush()
    
      else:
        _, top1_error_value, vali_loss_value = sess.run(
        [val_op, vali_top1_error, vali_loss], 
        {self.image_placeholder: batch_data, 
        self.label_placeholder: batch_label,
        self.vali_image_placeholder: vali_image_batch, 
        self.vali_label_placeholder: vali_labels_batch, 
        self.lr_placeholder: FLAGS.learning_rate,
        self.bbox_placeholder: batch_bbox,
        self.vali_bbox_placeholder: vali_bbox_batch,
        self.dropout_prob_placeholder: 1.0})
        
      print('Validation top1 error = %.4f' % top1_error_value)
      print('Validation loss = ', vali_loss_value)

The code below calculates the top 1 error value for training data. Strings of every iteration and corresponding loss values are also returned.
    
    
    if step % REPORT_FREQ == 0:
      summary_str = sess.run(summary_op, 
                            {self.image_placeholder: batch_data,
                             self.label_placeholder: batch_label,
                             self.bbox_placeholder: batch_bbox,
                             self.vali_image_placeholder: vali_image_batch,
                             self.vali_label_placeholder: vali_labels_batch,
                             self.vali_bbox_placeholder: vali_bbox_batch,
                             self.lr_placeholder: FLAGS.learning_rate,
                             self.dropout_prob_placeholder: 0.5})
      summary_writer.add_summary(summary_str, step)
    
      num_examples_per_step = TRAIN_BATCH_SIZE
      examples_per_sec = num_examples_per_step / duration
      sec_per_batch = float(duration)
    
      format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f ' 'sec/batch)')
      print (format_str % (datetime.now(), step, loss_value,
              examples_per_sec, sec_per_batch))
      print('Train top1 error = ', train_top1_error)

The code below updates the current lowest error via the min_error variable. It also updates the `step_list`, `train_error_list`, and `vali_error_list`.
    
    
    if top1_error_value < min_error:
        min_error = top1_error_value
        checkpoint_path = os.path.join(TRAIN_DIR, 'min_model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
        print('Current lowest error = ', min_error)
    
    step_list.append(step)
    train_error_list.append(train_top1_error)
    vali_error_list.append(top1_error_value)

The learning rate was 0.1 at the beginning and decayed to 0.01 at 25000 steps. The model was trained for 30000 steps in total. When training is finished, the model is saved into `model.ckpt` and the `error_df` data frame is saved into a separate csv file.
    
    
    if step == DECAY_STEP0 or step == DECAY_STEP1:
      FLAGS.learning_rate = FLAGS.learning_rate * 0.1
    
    if step % 10000 == 0 or (step + 1) == STEP_TO_TRAIN:
      checkpoint_path = os.path.join(TRAIN_DIR, 'model.ckpt')
      saver.save(sess, checkpoint_path, global_step=step)
    
      error_df = pd.DataFrame(data={'step':step_list, 'train_error':
          train_error_list, 'validation_error': vali_error_list})
      error_df.to_csv(TRAIN_DIR + TRAIN_LOG_PATH, index=False)

### Test Main Function

All the images are then evaluated/tested using the well-trained model. The nearest neighbor search is based on the values of the feature layer.
    
    
    def test(self):
      self.test_image_placeholder = tf.placeholder(dtype=tf.float32, 
                                    shape=[25, IMG_ROWS, IMG_COLS, 3])
      self.test_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[25])
      ##########################
      # Build test graph
      logits, global_pool = inference(self.test_image_placeholder, 
      n=FLAGS.num_residual_blocks, reuse=False,
      keep_prob_placeholder=self.dropout_prob_placeholder)
      
      predictions = tf.nn.softmax(logits)
      test_error = self.top_k_error(predictions, self.test_label_placeholder, 1)
    
      saver = tf.train.Saver(tf.all_variables())
      sess = tf.Session()
      saver.restore(sess, FLAGS.test_ckpt_path)
      print('Model restored!')
      ##########################
      test_df = prepare_df(FLAGS.test_path, 
      usecols=['image_path', 'category', 'x1', 'y1', 'x2', 'y2'], shuffle=False)
      test_df = test_df.iloc[-25:, :]
    
      prediction_np = np.array([]).reshape(-1, 6)
      fc_np = np.array([]).reshape(-1, 64)
      
      for step in range(len(test_df) // TEST_BATCH_SIZE):
          if step % 100 == 0:
              print('Testing %i batches...' %step)
              if step != 0:
                  print('Test_error = ', test_error_value)
    
          df_batch = test_df.iloc[step*25 : (step+1)*25, :]
          test_batch, test_label = load_data_numpy(df_batch)
    
          prediction_batch_value, test_error_value = sess.run([predictions, test_error],
          feed_dict={
              self.test_image_placeholder:test_batch, 
              self.test_label_placeholder: test_label})
          fc_batch_value = sess.run(global_pool, 
          feed_dict={
              self.test_image_placeholder:test_batch, 
              self.test_label_placeholder: test_label})
    
          prediction_np = np.concatenate((prediction_np, prediction_batch_value), axis=0)
          fc_np = np.concatenate((fc_np, fc_batch_value))
    
      print('Predictin array has shape ', fc_np.shape)
      np.save(FLAGS.fc_path, fc_np[-5:,:])

## Recommendation Results

In order to train the model, simply run the commands below:
    
    
    train = Train()
    train.train()

To test the model, run:
    
    
    train.test()

The recommendations to three example query images are shown below. We can see that the model can capture the style (including the sleeve length, the collar shape, and the slim/regular fit characteristics), the fabrics, and the printed pattern of the clothes.

In the first example, the model captures deep features include the dress category, the chiffon fabrics, and the repeated floral pattern. 

![](https://d2mxuefqeaa7sj.cloudfront.net/s_26D8191B6F13CAB8D7A633E957B00DECC09CB28AECA93C7EDFEB07C504C27EFB_1548019450205_Dresses-Examples.png)An example top-5 result on the dress category

The second example shows that the model can capture the dark color and the open zipper besides correctly classifying the query image into the jacket category.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_26D8191B6F13CAB8D7A633E957B00DECC09CB28AECA93C7EDFEB07C504C27EFB_1548019501723_Jacket-Examples.png)An example top-5 result on the jackets category

The third example shows that the model captures the soft fabric, the bright color and the blouse shirt category of the query image.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_26D8191B6F13CAB8D7A633E957B00DECC09CB28AECA93C7EDFEB07C504C27EFB_1548019508597_BlouseShirt-Examples.png)An example top-5 result on the blouse shirts category

### Ready to build, train, and deploy AI?

#### Get started with FloydHub's collaborative AI platform for free

###### [Try FloydHub for free ](https://www.floydhub.com/?utm_source=blog&utm_medium=banner-recommending-fashion-images&utm_campaign=try_floydhub_for_free)

## Next Steps

Fashion domain is an ideal space to apply deep learning. It’s easy to find a ton of public data and the current deep learning algorithms are capable of almost any computer vision tasks.

Here are a couple of areas that you can look at besides fashion recommendation:

### Attribute Recognition

Different clothes have different attributes. For example, your shoes can be available in different colors and sizes, your shirts can be available different textures and patterns, and your pants can be available in different materials and width. A good understanding of such clothing attributes is extremely useful to do any sort of comparison between clothing products.

### Image Generation

Artificial intelligence can not only learn a piece of clothing’s attributes, but also can create computer-generated images of similar-looking items. This is quite valuable, especially for retailers, to create personalized clothes or even predict broader fashion trends. Unfortunately, generating realistic-looking fashion images has been a challenging task due to their high-dimensions. In order to cope with that, researchers again look towards deep learning techniques. In particular, recent approaches in image generation have made heavy use of generative adversarial networks, a popular unsupervised machine learning model where there are two neural networks fighting against each other.

### Clothing Retrieval

Clothing retrieval is essentially a subset of image retrieval, an ongoing active research in computer vision domain. This technique attempts to identify the topic of an image, find the right keywords to index the image, and define the appropriate words to retrieve that image. There is a semantic gap in between these objectives, making the meaning of an image to be highly individual and subjective. With the large amount of image data, image retrieval on a big dataset becomes an even more challenging visual task.

* * *

## About James Le

James is currently studying at [RIT](https://www.rit.edu) for a Master’s degree to further his education in computer science and artificial intelligence. He has had professional experience in data science, product management, and technical writing. He is also an [AI Writer](https://floydhub.github.io/write-for-floydhub/) for FloydHub. 

You can follow along with James on [Twitter](https://twitter.com/@james_aka_yale) and [Medium](https://medium.com/@james_aka_yale).