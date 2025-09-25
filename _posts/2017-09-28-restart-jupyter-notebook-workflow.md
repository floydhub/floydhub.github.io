---
author: Charlie Harrington
date: 2017-09-28 04:35:00 +0000
excerpt: 'We snuck in a new feature that should make your Jupyter notebook workflow
  on FloydHub oh-so-much easier - the restart button.

  Now you can spin up a Jupyter notebook from the FloydHub web dashboard in...'
feature_image: /assets/images/hero/restart-jupyter-notebook-workflow-hero.jpg
layout: post
slug: restart-jupyter-notebook-workflow
tags: '[]'
title: Restart Jupyter Notebook Workflow
---

We snuck in a new feature that should make your Jupyter notebook workflow on FloydHub oh-so-much easier - the **restart** button.

Now you can spin up a Jupyter notebook from the FloydHub web dashboard in one click. Just head on over to one of your previous Jupyter notebook jobs and tap _Restart_. We'll kick off a new job that continues from where you left off in your previous job.

Let me reiterate that - the restart button spins up a new Jupyter notebook session based on the output state of a previous job. It's a real-deal, continuous Jupyter notebook workflow that still maintains all the benefits of our version-controlled job system.

![restart](/assets/images/content/images/2018/06/out15.gif)

But that's not all - you can even tweak some of the job parameters before you restart.

For example, let's say you ran a Jupyter notebook job using a CPU instance, which is a great way to set up your experiment and do some preliminary exploration. When you are ready, you can switch to running your Notebook on a GPU instance. Just click Restart and you can choose to run this next iteration using a GPU!

We're excited about this one. It feels like magic.

Give it a try now - restart a job today. Let your training begin.