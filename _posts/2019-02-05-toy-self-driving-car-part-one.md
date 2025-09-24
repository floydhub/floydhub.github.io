---
layout: "post"
title: "Building a Toy Self-Driving Car: Part One"
date: "2019-02-05 14:00:50 +0000"
slug: "toy-self-driving-car-part-one"
author: "Jaison Saji Chacko"
excerpt: "Learn the history and technology of autonomous cars in this Part 1 of a series on building a self-driving toy car with Raspberry Pi, Keras, and FloydHub GPUs."
feature_image: "__GHOST_URL__/content/images/2019/02/car-2.jpeg"
tags: "[]"
---

You don’t need to be a VC-funded startup to build your own self-driving car. Especially if that car is tiny, remote-controlled, and can easily fit on your desk.

In this four-part blog series, we’ll build our own self-driving (toy) car using a [Raspberry Pi](https://www.raspberrypi.org), a generic remote-control car, some basic electronic components, [FloydHub GPUs](https://www.floydhub.com), and the Keras deep learning framework.

If you’re not already revved up about this autonomous driving project, here’s a quick glimpse of my desk right now:

![](/assets/images/content/images/2019/02/car-1.jpeg)Components for our self-driving toy car

Let’s take a look at the road ahead. I’ll be sharing our journey in four posts:

  * _Part One_ : History of autonomous vehicles and overview of self-driving technology
  *  _Part Two_ : Build a custom RC car controller with Raspberry Pi and Python
  *  _Part Three_ : Implement, train, and deploy a _baseline_ self-driving car model with Keras
  *  _Part Four_ : Refactor, re-train, and re-deploy an _improved_ self-driving car model with Keras

Ready? Set? Go!

# A brief history of self-driving cars

Although interest in autonomous driving has recently gained momentum, the idea of self-driving technology goes back to the earliest days of the motorized vehicle.

For example, you might be familiar with technology that automatically controls the speed of a motor vehicle — commonly called _cruise control_. [First patented in the United States in 1950](http://pdfpiw.uspto.gov/.piw?Docid=02519859&homeurl=http%3A%2F%2Fpatft.uspto.gov%2Fnetacgi%2Fnph-Parser%3FSect2%3DPTO1%2526Sect2%3DHITOFF%2526p%3D1%2526u%3D%2Fnetahtml%2FPTO%2Fsearch-bool.html%2526r%3D1%2526f%3DG%2526l%3D50%2526d%3DPALL%2526S1%3D2519859.PN.%2526OS%3DPN%2F2519859%2526RS%3DPN%2F2519859&PageNum=&Rtype=&SectionNum=&idkey=NONE&Input=View+first+page), cruise control can arguably be traced back even further to the use of _governors_ in the 18th century that would regulate the fuel in steam engines, allowing the engines to maintain constant speeds.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_9983F48E5C5E09133AAD694DA274A11684A3639C41EBADE91CF350EFA68FBF9F_1548112013137_image.png)Steam-engine governor

But this long and fascinating history of self-driving technology was ultimately propelled forward into the modern era with the DARPA Grand Challenge in the early 2000s.

## The DARPA Grand Challenge

The first Grand Challenge was held on March 13, 2004, in Mojave Desert, United States. The Grand Challenge was the first long-distance racing competition for autonomous cars. It was organized by DARPA — one of the research arms of the United States Department of Defense. They’re also the folks who built the first version of the Internet back in the 1960s and 1970s (which was called the ARPANET). So they’ve clearly got their eyes pointed towards the future.

And, in 2004, that bright future was self-driving cars.

2004 DARPA Grand Challenge

Unfortunately, none of the cars in the 2014 challenge were able to cross the finish line, so no winners were declared. Carnegie Mellon University’s team did the best; their car Sandstorm traveled 11.78 km out of the 240 km route.

In 2005, DARPA conducted its second competition. The race was held October 8th. Almost all the cars that participated surpassed the previous year’s record — and five cars successfully completed the race.

2005 DARPA Grand Challenge

The Urban Challenge — which was the third DARPA Grand Challenge — was held on November 3rd, 2007. DARPA decided to kick things up a notch with the Urban Challenge. In the previous challenges, cars were expected to cover a predefined route across the desert. However, this time, the cars had to drive on urban roads with the other cars, following all the regular traffic rules and regulations. Despite being extremely difficult, the race was successfully completed by a few teams.

2007 DARPA Urban Challenge

## Commercial attention

The Urban Challenge was a major turning point in this history of self-driving technology. Naturally, commercial (and venture capital) attention was piqued.

### Google

In 2009, Google began [development](https://www.businessinsider.in/How-Googles-self-driving-car-project-rose-from-a-crazy-idea-to-a-top-contender-in-the-race-toward-a-driverless-future/Thrun-first-began-his-research-on-driverless-vehicles-at-Stanford-leading-a-student-and-faculty-team-that-designed-the-Stanley-robot-car-The-car-won-a-2-million-prize-at-the-2005-DARPA-Grand-Challenge-for-driving-132-miles-in-the-desert-on-its-own-/slideshow/55018879.cms) of its own self-driving cars. They hired the very best engineers from the teams that participated in the DARPA challenges. In 2016, Google’s self-driving car project spun out into a separate company called [Waymo](https://waymo.com/), which is now a stand-alone subsidiary of Google’s parent company [Alphabet](https://abc.xyz/).

### Traditional Car Manufacturers

By 2013, most of the major automotive companies, including [General Motors](https://www.designboom.com/technology/en-v-electric-networked-car-concept-by-gm-begins-pilot-testing/), [Mercedes Benz](https://www.wsj.com/articles/driverless-cars-for-the-road-ahead-1380221071?tesla=y), [Ford](https://gigaom.com/2012/04/09/ford-is-ready-for-the-autonomous-car-are-drivers/), and [BMW](https://www.thenational.ae/business/an-automated-adventure-at-the-wheel-of-a-driverless-bmw-1.371963), had publicly announced that they were also working on their own self-driving technology. 

### Uber

Uber — the ride-sharing company — began testing its [own self-driving cars in 2016 in Pittsburgh](https://www.uber.com/blog/pennsylvania/new-wheels/). Uber’s self-driving car prototype was a Ford Fusion Hybrid with a roof-full of radar, lasers and cameras collected road-mapping data and tested real world traffic conditions. 

![](https://www.gannett-cdn.com/-mm-/af944507a3dacca1c776bdb29d5d3e36ed9f668e/c=0-0-1368-1029/local/-/media/2016/05/19/USATODAY/USATODAY/635992551287139163-UberCar.jpg?width=534&height=401&fit=crop)Uber's self-driving prototype

Unfortunately, despite the presence of a human driver in the car, a fatal accident took the life of a woman in Tempe, Arizona [in March 2018](https://www.nytimes.com/2018/03/19/technology/uber-driverless-fatality.html). Uber recently regained permission to test self-driving cars, but [under the conditions of two human drivers being present in the cars at all times and a speed limit of 25 MPH](https://wach.com/news/auto-matters/uber-approved-to-resume-autonomous-car-tests-in-pittsburgh). This time around, Uber is using Volvo SUVs which have a inbuilt automatic braking system. They’re expected to begin testing again in Pittsburgh soon.

### Lyft

Lyft — not to be outdone by its arch rival ride-sharing competitor — has also started testing self-driving cars. Unlike Uber, they are using the [private 5,000 acre campus space of GoMentum Station](https://www.fastcompany.com/40541471/lyft-begins-testing-its-own-self-driving-cars-on-a-private-5000-acre-campus) in Contra Costa County, California. Lyft’s self driving efforts have generally tended to be more partnership oriented. They have successfully partnered with GM, Ford, Aptiv, Drive.ai, Waymo, and Jaguar.

### Tesla

Tesla — the first successful U.S. car company startup in decades — claims to be ahead of everyone in the game. [Their CEO Elon Musk announced recently](https://www.forbes.com/sites/jeanbaptiste/2018/11/07/tesla-could-have-full-self-driving-cars-on-the-road-by-2019-elon-musk-says/#6bebd15c62ac) announced that Tesla may have fully-functional self-driving electric cars this year:

> _"I don’t want to sound over-confident, but I would be very surprised if any of the car companies exceeded Tesla in self-driving, in getting to full self-driving._ _They’re just not good at software. And this is a software problem."_

### More self-driving car projects continue to be announced

In what, naturally, feels like a drag race for self-driving hegemony, more and companies continue to announce their own self-driving car projects. 

A few weeks ago, [Yandex](https://yandex.com/) — a Russia-based search engine giant — [demonstrated](https://globenewswire.com/news-release/2019/01/07/1681365/0/en/Yandex-Demonstrates-Self-Driving-Car-Technologies-on-the-streets-of-Las-Vegas-During-CES-2019.html) their self-driving car technologies on the streets of Las Vegas during CES 2019. Yandex has been testing its self driving cars against the rugged weather conditions of the streets of Russia. They [claim](https://yandex.com/blog/yacompany-com/ces2019-demo) to have already delivered over 2,000 self-driven rides with only a safety engineer in the front.

# Levels of self-driving cars

There’s an important rubric in the self-driving lexicon that’s worth mentioning up front, because you’ll inevitably hear it discussed in any detailed report about the progress of autonomous vehicles.

First framed by [SAE International (Society of Automotive Engineers)](https://www.sae.org/standards/content/j3016_201609/) in 2014, these levels outline the degree of autonomy of a self-driving vehicle.

### Level 1

Limited driver assistance. This includes systems that can control steering and acceleration/deceleration under specific circumstances, but not both at the same time.

### Level 2

Driver-assist systems that control both steering and acceleration/deceleration. These systems shift some of the workload away from the human driver, but still require that person to be attentive at all times.

### Level 3

Vehicles that can drive themselves in certain situations, such as in traffic on divided highways. When in autonomous mode, human intervention is not needed. But a human driver must be ready to take over when the vehicle encounters a situation that exceeds its limits.

### Level 4

Vehicles that can drive themselves most of the time, but may need a human driver to take over in certain situations.

### Level 5

Fully autonomous. Level 5 vehicles can drive themselves at all times, under all circumstances. They have no need for manual controls.

### Progress to date

As of 2016, few companies have claimed to be at Level 2, including:

  * [Tesla Autopilot](https://static.nhtsa.gov/odi/inv/2016/INCLA-PE16007-7876.PDF)
  * [Volvo Pilot Assist](https://www.sae.org/news/2016/01/volvos-2017-s90-has-standard-semi-autonomous-driving-system)
  * [Mercedes-Benz Drive Pilot](https://www.mercedes-benz.com/en/mercedes-benz/innovation/the-new-e-class-on-the-road-to-autonomous-driving-video/)

[According to Audi](https://www.audi-technology-portal.de/en/electrics-electronics/driver-assistant-systems/audi-a8-audi-ai-traffic-jam-pilot), the 2018 Audi A8 is claimed to be the first car to achieve level 3 autonomy with its AI traffic jam pilot. Meanwhile, Waymo has already been [running their level 4 autonomous cars](http://www.thedrive.com/tech/15848/waymo-is-already-running-cars-with-no-one-behind-the-wheel) in Arizona since mid-October 2017.

You can expect to see many more announcements in the coming months and years as the rest of these commercial efforts advance through these self-driving car levels.

### Let's talk about simulators

One of the major reasons why Waymo is [ahead of most companies](https://www.bloomberg.com/news/features/2018-05-07/who-s-winning-the-self-driving-car-race) in the field is mostly due to the fact that their cars have collectively covered more than 5 million miles on road and billions of miles in their self-driving car simulator. 

In 2016, Waymo’s car simulator known as CarCraft [logged over 2.5 billion virtual miles](https://www.theatlantic.com/technology/archive/2017/08/inside-waymos-secret-testing-and-simulation-facilities/537648/). CarCraft can simulate thousands of different scenarios and maneuvers every day. At any given time, there are 25000 simulated self-driving cars driving across fully modeled versions of Austin, Mountain View and Phoenix, as well as other test scenarios. 

According to the RAND corporation, [for self-driving cars to be even 20 percent better than humans, it would require 11 billion miles of validation](https://www.rand.org/content/dam/rand/pubs/research_reports/RR1400/RR1478/RAND_RR1478.pdf). This would take 500 years of non-stop driving by a fleet of 100 cars to cover this distance. 

Simulators are a great solution to this problem. They are a safe way for developers to safely test and validate performance of self-driving hardware and software. In September 2018, NVIDIA opened up their [DRIVE Constellation simulation platform](https://www.nvidia.com/en-us/self-driving-cars/drive-constellation/) for partners to integrate their world models, vehicle models and traffic scenarios. 

# The ethics of autonomous vehicles

Before we move on to the technology behind self-driving cars, it’s critical that we discuss some of the ethical issues surrounding the development of self-driving cars. As Rachel Thomas of fast.ai states in [her excellent guide to AI Ethics Resources](https://www.fast.ai/2018/09/24/ai-ethics-resources/):

> Everyone in tech should be concerned about the ethical implications of our work.

Self-driving artificial intelligence, in particular, presents many stark ethical questions. If you’ve ever driven a car, then you’ll know that driving is a constant stream of decisions. You need to be following the rules of the road, while at the same responding to other drivers and pedestrians, and also handling any unexpected events, like weather or other strange conditions. But when turning control over to an AI system, how should the vehicle handle its decision-making process? For example, what should the priority of the car be in the event of a potential accident? 

> Should the vehicle prioritize the protection of its own passengers? Or those passengers of another vehicle? Or what about those of a cyclist on the road? Or a animal that entered the highway? What about pedestrians?

Uber’s accident in Tempe which took the life of a pedestrian led to many debates regarding this very ethical dilemma. Driving is full of murky situations — especially during crosswalks, turns, and intersections. This is an incredibly important moment in AI — since these decisions will decide the way that our cars of the future behave. Algorithm will become policy. 

Self-driving cars are coming. This much is clear. Humans have greater perception and are better at mechanical tasks as a result of many years of evolution. For a machine to do the same as a human, or better, will take some time and effort. But a solution to this ethical problem must be provided before self-driving cars start replacing human drivers.

# The technology behind self-driving cars

Since we’re going to be building our own toy self-driving car in this blog post series, let’s dive into the technology that makes self-driving possible. 

The architecture of the autonomy system of self-driving cars is typically organized into two main parts:

  * the **perception system**
  * the **decision-making system**.

The _perception system_ is generally divided into many subsystems responsible for tasks such as:

  * self-driving-car localization
  * static obstacles mapping
  * moving obstacles detection and tracking
  * road mapping
  * traffic signalization detection and recognition

The perception system is also responsible for determining the state (position, speed, direction, etc) of the car at any given point of time using the input from various sensors. We’ll discuss these sensors in the next section of this post.

The _decision-making system_ is commonly partitioned as well into many subsystems responsible for tasks such as:

  * route planning
  * path planning
  * behavior selection
  * motion planning
  * control

The decision making system is also responsible for taking the car from one position to another, considering the state of the system as well as the current traffic rules. In order to make such decisions, the decision-making system needs to know the position of the car and its environment.

Here’s a handy diagram from [a recent paper](https://arxiv.org/abs/1901.04407) on the modules within a typical self-driving car architecture:

![](https://d2mxuefqeaa7sj.cloudfront.net/s_783BD3CBB12C9099DEFB0B63D5C52084EF05DB93D9046EB2B3F669C743BAF6BA_1549056796857_image.png)Overview of the typical hierarchical architecture of self-driving cars. TSD denotes Traffic Signalization Detection and MOT, Moving Objects Tracking

The **Localizer module** is responsible for providing the decision making system with the location. It makes use of offline maps, sensor data, and platform odometry to determine the position of the car. 

The **Mapper module** produces a merge of information present in the Offline Maps and an occupancy grid map computed online using sensors’ data and the current State. 

The **Route Planner module** computes a route from the starting position to the goal defined. The path planner then computes a set of paths. A route is a collection of waypoints whereas a path is a collection of poses. A pose is a coordinate pair in the Offline Maps, and the desired car’s orientation at the position defined by this coordinate pair.

The **Behavior Selector module** is responsible for choosing the current driving behavior, such as lane keeping, intersection handling, traffic light handling, etc.

The **Motion Planner module** is responsible for computing a trajectory from the current car’s state to the current goal, which follows the path defined by the Behavior Selector module, satisfies car’s kinematic and dynamic constraints, and provides comfort to the passengers.

The **Obstacle Avoider module** receives the trajectory computed by the Motion Planner and changes it (typically reducing the velocity), if necessary, to avoid collisions.

Finally, the **Controller module** receives the Motion Planner trajectory, eventually modified by the Obstacle Avoider, and computes and sends commands to the actuators of the steering wheel, throttle and brakes in order to make the car execute the modified trajectory as best as the physical world allows.

## Sensors used by self-driving cars

### LIDAR 

You’ve probably heard about LIDAR at this point. 

LIDAR, or LIght Detection And Ranging, is used to measure the distance to a target by emitting pulsed laser light and measuring the reflected pulses using a sensor. Combining inputs from multiple LIDAR modules around the car can be used to create an accurate map of its surroundings.

After the invention of laser in 1960, LIDAR was first tested on airplanes using downward facing lasers to map the ground surface. But it was only in late 1980s in which LIDAR measurements became reliable, when it was combined with GPS readings and inertial measurement units (IMUs). 

![](https://d2mxuefqeaa7sj.cloudfront.net/s_783BD3CBB12C9099DEFB0B63D5C52084EF05DB93D9046EB2B3F669C743BAF6BA_1544372024398_1.png)Point Cloud Image from a LIDAR

### RADAR

Radar (RAdio Detection And Ranging) modules are also commonly used in self-driving cars. They work almost the same way as a LIDAR — the major difference is that radar uses radio waves rather than lasers.

Radar was developed for the military back in 1930s to detect aggressors in the air or on the sea. Aircraft and missile detection is still one of the main uses of radar. It is also widely used in air traffic control, navigation systems, space surveillance, ocean surveillance and weather monitoring. 

### Cameras

Most self-driving cars utilize multiple cameras for mapping its surrounding. For example, a Tesla has 8 cameras around the car which gives a 360-degree view. This enables the Tesla vehicle to have full automation without requiring the help of other sensors. 

Unlike LIDAR, cameras can pick up lane markings, traffic lights, road signs and other signals, which gives a lot more information for the car to navigate on roads.

Video from Tesla’s website showing autonomous driving alongside different camera feeds.

  
Most self-driving cars use a combination of sensors and cameras, but with machine learning and computer vision playing a major role in self-driving technology, cameras are going to be the main component and might even replace other sensors completely over time. 

## Deep Learning for self-driving cars

As we know already, cameras are key components in most self-driving vehicles.

Most of the camera tasks fall into some type of _computer vision detection_ or _classification problem_. Recent advancements in deep learning and computer vision can enable self-driving cars to do these tasks easily.

Lets look at a deep learning pipeline by NVIDIA called DAVE-2, described in the paper ‘[End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf)’. The paper describes a convolutional neural network which is trained to map raw pixels from the camera feed to steering commands for the vehicle. 

### Network Architecture

The model consists of 5 convolutional layers, 1 normalization layer and 3 fully connected layer. The network weights are trained to minimize the mean-squared error between the steering command output by the network and the ground truth. 

![](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)This convolutional neural network (CNN) has about 27 million connections and 250 thousand parameters.

### Data Collection

The training data is the image feed from the cameras and the corresponding steering angle. 

The data collection is quite extensive considering the huge number of possible scenarios the system will encounter. The data is collected from a wide variety of locations, climate conditions, and road types. Also, training with data from the human drivers is not enough. The network should learn to recover from mistakes otherwise the car might drift off the lane.

In order to solve this problem, the data is augmented with additional images that shows different positions where the car is shifting away from the center of the lane and different rotations from the direction of the road. For example, the images for two specific off-center shifts from the left and right cameras and the remaining range of shifts and rotations are simulated using viewpoint transformation of the image from the nearest camera.

The collected data is labelled according to road type, weather conditions, and driver’s activity. The driver could be staying on lane, changing lane, turning and so on. 

In order to train a convolutional neural network (CNN) that can stay on lane, we take only the images where the driver is staying on lane. The images are also down-sampled to 10 frames-per-second (FPS), as many of the frames would be similar and wouldn’t provide more information for the CNN model. Also, to avoid a bias towards driving straight all the time, more frames that represent road curves are added to the training data.

### Training

In order to train the model, data from three cameras as well as the corresponding steering angle is used. The camera feeds and the steering commands are time-synchronized so each image input has a steering command corresponding to it. 

![](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/training-624x291.png)Training the neural network

Images are fed into the CNN model which outputs a proposed steering command. The proposed steering command is then compared with the actual steering command for the given image, and the weights are adjusted to bring the model output closer to the desired output. Once trained, the model is able to generate steering commands from the image feeds coming from the single center camera. 

![](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/inference-624x132.png)The trained network is used to generate steering commands from a single front-facing center camera.

### Evaluation

The trained model is evaluated in two steps, first in simulation and then in on-road tests.

#### Simulation Tests

The computer simulation uses a library of recorded video footage from cameras and their corresponding steering commands and renders images that approximate what would appear if the model was steering the car.

Using the simulation test, an **autonomy score** is determined for the trained model. The autonomy metric is calculated by counting the number of simulated human interventions required. An intervention occur when the simulated vehicle drifts off the center of the lane by more that one meter. An average of 6 seconds is considered for a real life human intervention where they have to regain control of the vehicle and bring it back to the center of the lane. 

So the autonomy score is calculated as:

![](https://s0.wp.com/latex.php?latex=%5Ctext%7Bautonomy%7D+%3D+%281+-+%5Cfrac%7B%5Ctext%7B%5C%23+of+interventions%7D+%5Ccdot+6+%5Ctext%7B%5Bseconds%5D%7D%7D%7B%5Ctext%7Belapsed+time%5Bseconds%5D%7D%7D%29+%5Ccdot+100&bg=ffffff&fg=000&s=0)

So, if we had 10 interventions in 600 seconds, the autonomy score would be 90%:

![](https://d2mxuefqeaa7sj.cloudfront.net/s_783BD3CBB12C9099DEFB0B63D5C52084EF05DB93D9046EB2B3F669C743BAF6BA_1549033420718_image.png)

#### Deploying to a Vehicle for Testing

Once the trained model achieves good performance in the simulator, it is loaded on the [DRIVE PX](http://www.nvidia.com/object/drive-px.html) in the test car. 

DRIVE PX is a computer specially designed for autonomous cars. Many major auto manufacturers including [Audi](https://nvidianews.nvidia.com/news/nvidia-audi-partner-to-put-world-s-most-advanced-ai-car-on-road-by-2020), [Tesla](https://www.nvidia.com/en-us/self-driving-cars/partners/tesla/), [Volvo](https://nvidianews.nvidia.com/news/volvo-cars-and-autoliv-select-nvidia-drive-px-platform-for-self-driving-cars) and [Toyota](https://nvidianews.nvidia.com/news/nvidia-and-toyota-collaborate-to-accelerate-market-introduction-of-autonomous-cars) use DRIVE PX. For the on-road tests, the performance metrics are calculated as the fraction of time during which the car is performing autonomous steering. 

Here is a video of DAVE-2 in action.

DAVE-2 self-driving car

  
In the coming blog posts we’ll see how to build our own self-driving toy car by drawing inspiration from the DAVE-2 system.

## Next steps

I hope you found this overview of self-driving car technology helpful. It’s been useful for me to research and better understand the decisions and trade-offs that have been made to achieve the incredible advances in autonomous driving recently.

In our next post, we’ll starting building our self-driving toy car! Our goal will be build a custom controller for an RC car using a Raspberry Pi and L298 Motor Driver Module. We’ll also be using the Raspberry Pi camera module to act as our main input device.

Time to heat up my solder iron!

* * *

### About Jaison Saji Chacko

 _This post is part one of a four-part FloydHub blog series on building your own toy self-driving car._

Jaison is a Machine Learning Engineer at [Mialo](http://twitter.com/MialoAI). He is based in Bangalore, India. He works mostly on computer vision. Jaison is also a [FloydHub AI Writer](https://floydhub.github.io/write-for-floydhub/).

You can follow along with Jaison on [Twitter](https://twitter.com/jaisonsaji) and [Github](https://github.com/jsn5).