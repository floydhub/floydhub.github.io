---
layout: "post"
title: "FloydHub Cloud Setup Challenge:  Jupyter + TensorFlow in 44 seconds [WR]"
date: "2020-05-12 16:12:59 +0000"
slug: "cloud-setup-challenge-jupyter-tensorflow"
author: "Alessio Gozzoli"
excerpt: "Is it possible for data science beginners to get up and running in under 90 seconds? FloydHub’s team takes on the setup cloud challenge - and walks away with the trophy. (For now!)"
feature_image: "__GHOST_URL__/content/images/2020/05/big-1.jpg"
tags: "[]"
---

_Here at Floydhub, we're always looking to push the envelope in terms of how we bring our users and readers new information about AI. We decided to set aside the envelope completely, and channel our favorite sports broadcasters to take on_[ _Cassie Kozyrkov’s recent Jupyter+Tensorflow challenge_](https://towardsdatascience.com/90-second-setup-challenge-jupyter-tensorflow-in-google-cloud-93038bcac996) _. Spoiler: We kind of smashed that record. By a lot._

[Mark - totally real TV commentator]: "Ladies, gentlemen, data scientists and enthusiasts — welcome to the first Cloud Challenge! Hi 👋 Janet, are you ready for this?!"

[Janet - other real TV commentator]: "Hi 👋 Marc, and hi to all the folks who are following us from home. I’m totally psyched for this. A few days ago we saw [Cassie](https://towardsdatascience.com/@kozyrkov?source=post_page-----93038bcac996----------------------) establishing the first & new world record in the fantastically obscure sport of setting up Jupyter + TensorFlow in the Cloud. A lot of folks have been talking about the challenge, but today we are going to see the first challenger tackle this feat head-on."

Marc: "I've personally heard rumors backstage of some big players putting their teams under intensive VR simulation to beat Cassie. However, for now, they’re just rumors. No one — let me underline that, _no one_ — has officially accepted the challenge. Until today. The FloydHub team thinks they’ve got this. I’m excited for the showdown."

> Challenge accepted 😁!
> 
> -- Alessio Gozzoli (@GGozzoli) [May 2, 2020](https://twitter.com/GGozzoli/status/1256598520066441216?ref_src=twsrc%5Etfw)

Janet: "Yes, Marc, you’re right. Unbelievable. I'm not sure what the other teams are planning, but it looks like FloydHub is determined to bring the title home today."

Marc: "I was just talking to Alessio, the FloydHub pilot for today and he seemed extremely determined to break the record. FloydHub's team leader, Sai, said that they created a personalized autogenic training to refine his preparation for this big event, and a couple of smart strategies to make Alessio super focused during the performance."

Janet: "That's what we call commitment! Uh, it looks like that FloydHub champion is approaching the starting grid. Let the cloud platform compute!"

## Cloud Challenge Regulations

Marc: "Before starting, Janet, would you like to remind our audience following at home what the rules of the competition are?"

Janet: "Sure thing, Marc! Let's dive into the setup.

  1. The Setup Cloud Challenge is a timed attack competition that pits competing Cloud Platforms for data science against each other.
  2. The goal is to measure the time required to spin up a machine with a TensorFlow environment on Jupyter (Notebook or Labs) in the shortest amount of time possible.
  3. Scripting is, of course, not allowed - this is the Platform & Infra championship."

Marc: "Thank you, Janet! It looks like Alessio has just finished his warm-up. Get ready to rumble!"

## The 44-seconds challenge

Janet: "As a reminder, the current world record is 84 seconds. That’s the number to beat. Here we go!"

![](/assets/images/content/images/2020/04/cloud-challenge-wrrecord.gif)It's official! We have a new World Record: 44s. If it was too fast for your eyes, don't worry you can see the slow-mo with technical commentary just below.

## What did we just watch?

Marc: "That was unbelievable, Janet! We saw the record nearly halved. That’s twice as fast the google cloud setup. Let's go through it with our slow-mo camera to help our viewers. First, what were the warm-up steps before the video kicked off?"

### Warm-up Step 0.1: Create FloydHub account

You have a couple of options here. You can [sign up](https://www.floydhub.com/signup) with an existing Google & GitHub account, or create a new username and password for a new account. Both are really easy to use. If you have trouble, just reach out to FloydHub support and they’ll help.

### Ready to build, train, and deploy AI?

#### Get started with FloydHub's collaborative AI platform for free

###### [Try FloydHub for free ](https://www.floydhub.com/?utm_source=blog&utm_medium=cloud-challenge&utm_campaign=try_floydhub_for_free)

### Warm-up Step 0.2: Fill out onboarding form

Fill out the onboarding questions and add your payment information to unblock the Free Trials (2h of GPU and 20h of CPU).

We didn't cover those screens for privacy reasons. But now we’ll go to the actual images from the challenge.

### Step 1: Create a new project

Marc: "What were they thinking over at FloydHub? They chose to start with one of the warmup steps of Cassie. Not sure if they were brave or overconfident, Janet. But it looks that this decision didn't backfire."

![](/assets/images/content/images/2020/05/new-project.png)

### Step 2: Create a Workspace

Janet: "The FloydHub Team made it damn easy for Alessio, and the audience as well, to not get lost during the setup. I guess this was one of the strategies they talked about before the competition."

![](/assets/images/content/images/2020/04/create-a-workspace-1.png)

In this case, Alessio chose to start from scratch, but you can start from any existing public GitHub repository and save a couple of commands later.

![](/assets/images/content/images/2020/04/create-a-workspace2.png)

Marc:**** "Here we can see how the user interface made a huge contribution during setup. Alessio adopted a similar setup to Cassie, but nothing prevents you from using a more powerful machine or a different environment (such as PyTorch)."

![](/assets/images/content/images/2020/04/create-a-workspace3.png)

###  Step 3: Start & Config your Workspace.

Marc: "Here the Workspace machine has been launched automatically, but if it were in shutdown or stopped states, you could restart your machine from this screen."

![](/assets/images/content/images/2020/04/running-workspace.png)

"Oh my God! What we've just seen here was really unbelievable. Alessio wasn't just waiting for the workspace to load the Jupyter Lab interface. He was modifying the settings on the fly. Before he attached a public MNIST dataset, he modified the IDLE timeout to stop the workspace in case he forgot to shut down the machine later! _That's what it means to have a strong consideration of every single penny spent on a service_."

![](/assets/images/content/images/2020/05/mount-data-2.png)Attach dataset to workspace![](/assets/images/content/images/2020/05/idle-settings.png)IDLE settings

###  Step 4: Import TensorFlow & list the data.

Janet: "And here we are approaching the finish line, just a couple of clicks left!"

![](/assets/images/content/images/2020/05/launcher.png)

Marc: "The world record has officially been smashed! U-N-B-E-L-I-E-V-A-B-L-E! Janet, I'm seriously astonished."

Janet: "Yes, Marc. That's crazy! _**Only 44 seconds to spin-up a machine with TensorFlow + Jupyter in the Cloud.**_ "

Marc: "Wait a second, Janet, the referees have to confirm that the setup is working as expected... let's wait the last few seconds and…"

![](/assets/images/content/images/2020/05/imports.png)

Janet: "It's official, Marc! We have the NEW WORLD RECORD."

###  Step 5: Don't forget to Shutdown the machine!

Marc: "Ready for the victory lap! Here's the team showing another step that's usually neglected in the setup for a Cloud Challenge. As we saw before, there's a cool option that automatically shuts down the machine when nothing is running, but it's always better to manually shutdown the machine when your work is done."

![](/assets/images/content/images/2020/04/shutdown.png)

### Game on!

Janet: "To Cassie’s question, 'Can you do better than me?,' well, these guys certainly can. This is the perfect setup for teams and practitioners who don't have the capacity to maintain a dedicated Infrastructure for their ML experiments. What do you think, Marc?"

Marc: "Without any doubts, this was a great showcase of FloydHub’s strengths. Google Cloud has certainly an incredible engine behind the scene, but in terms of usability & experience, FloydHub has certainly raised the bar. They are literally playing another game!"

Janet: "Hang on, Marc. We have Alessio at the microphone with Kevin for the post-challenge interview."

### Post Challenge Interview

[Kevin - in-place interviewer]: "Hi Alessio, amazing race today. Would you like to analyze the race with us?"

Alessio: "Hi Kevin, it was seriously tough, but the team did a fantastic job to put me on the right track with every single click. I think that our decision to start from _Create a new project_ probably penalized us a bit, but it gave me the opportunity to be more focused during the race from the very beginning.

I was a bit worried about the machine provisioning, since during the trials we experienced some delay & network lag. I'm physically quite distant from our servers. That could have impacted my gameday performance, but luckily everything went smoothly during the challenge. The team worked extremely hard last night to set up the infrastructure for today.

I want to thank the team one more time. There's also a special thanks to the folks who selected FloydHub to work & advance the Data Science field, you guys are the ones who made this possible."

Kevin: "How do you feel as a new world record holder? Do you think you can do better next time?"

Alessio: "You know, every record is there to be broken. So, I'm looking forward to the next challenger. But at the same time, I know that Cassie and the Google Cloud team will not wait too long before retrying. From our side, I can say that we still have about 10 seconds of improvements left to bring this to the 30-second setup cloud challenge. There are a couple of things that we can certainly improve, but I don't want to spoil anything! 😁"

Kevin: "You are a wise man. Thank you, Alessio, and congrats to the FloydHub team for the fantastic teamwork. Janet & Marc, I pass the line to you."

Janet: "Thank you, Kevin! Let's hope to hear from Cassie. Meanwhile, for our Premium Desktop subscribers, put yourselves full-screen mode, and let's go onboard with Alessio & Cassie! Go!"

![](/assets/images/content/images/2020/05/output.gif)

![](https://miro.medium.com/max/1400/1*Ghms-lXqfhnIobdALcWGnA.gif)

## Conclusion

Marc: "Today, we watched history being made at the temple of cloud usability. I don't think that I will be able to sleep tonight given the amount of adrenaline.”

Janet: “Same here, Marc. I still cannot believe what we've seen. That's it, folks. We’re signing off, and we’ll see you at the next Setup Cloud Challenge. Can’t wait to welcome new challengers!”