---
layout: "post"
title: "Hello, README files"
date: "2017-09-07 04:59:00 +0000"
slug: "hello-readme-files"
author: "Charlie Harrington"
excerpt: "We're introducing a new feature to help you explore projects and share your work
on FloydHub that you may already recognize - README files.

Starting today, whenever you include a README file in your ..."
feature_image: "__GHOST_URL__/content/images/2018/06/wave-1-min.png"
tags: "[]"
---

We're introducing a new feature to help you explore projects and share your work on FloydHub that you may already recognize - README files.

Starting today, whenever you include a README file in your project's code directory, FloydHub will detect and display the README from your latest job on your project's overview page.

![README Example](/assets/images/content/images/2018/06/readmegif.gif)

In fact, if you've already uploaded a README file along with one of your existing projects, then you should already be able to see your project README in action.

### Playing nice with others

We're focused on building a community of collaboration and learning at FloydHub. With project README files, we're excited for you to help other people understand:

  * why your project is useful
  * what they can do with your project
  * how they can run your project

This last bit is key for FloydHub projects - you'll want to include specific details on how to run your project using the `floyd run` command. If you're lucky - and using GitHub - you may already have a README file for your project. We recommend adding FloydHub-specific instructions to this existing file.

For example, a good FloydHub project README includes a code block outlining how someone can run your project on FloydHub:
    
    
    $ floyd run --env keras --gpu python deep_dream.py sample.jpg /output/
    $ floyd logs -t <RUN_ID>
    $ floyd output <RUN_ID>
    

Paying it forward with a README worth reading will help grow the deep learning community at FloydHub and beyond. And, let's be honest, we've all been saved by a good README for our own projects after some much needed time away from the computer.

### More summer reading

If you're looking for more info on how to write a great README, check out:

  * 18F's [Making READMEs readable](https://open-source-guide.18f.gov/making-readmes-readable/)
  * Wikipedia's [README](https://en.wikipedia.org/wiki/README) entry

Or, even better, check out these great projects on FloydHub with helpful READMEs on how to get started with deep learning now:

  * [@paulemmanuel](https://www.floydhub.com/paulemmanuel)'s [NYC Taxi Trip Duration](https://www.floydhub.com/paulemmanuel/projects/nyc_taxi_trip_duration)
  * [@vagmi](https://www.floydhub.com/vagmi)'s [Deep Q&A](https://www.floydhub.com/vagmi/projects/deepqa)