---
author: Naren Thiagarajan
date: 2018-07-03 23:05:12 +0000
excerpt: FloydHub has a lot of features that accelerate various stages of your ML
  workflow. In this post we are sharing the 5 most useful features and how to incorporate
  them in to your workflow.
feature_image: /assets/images/hero/advanced-features-hero.jpg
layout: post
slug: advanced-features
tags: '[]'
title: 5 advanced features that save you time on FloydHub
---

FloydHub has a lot of features that accelerate various stages of your ML workflow. In this post we are sharing the 5 most useful features and how to incorporate them in to your workflow.

# 1\. Update datasets directly on FloydHub

Easily add a few files to your large dataset

You can now easily update your dataset on FloydHub. You can even combine multiple datasets in to one! After you are done, a new version of the dataset is created with your changes.

# 2\. Work with private GitHub projects

You can clone your private GitHub repos directly into a workspace ([using a terminal](https://docs.floydhub.com/guides/workspace/#using-terminal)). You need to enter your GitHub username and password.
    
    
    root@floydhub:/floyd/home#
    root@floydhub:/floyd/home# git clone https://github.com/floydhub/demo-private.git
    Cloning into 'demo-private'...
    Username for 'https://github.com': username
    Password for 'https://username@github.com':
    remote: Counting objects: 7, done.
    remote: Compressing objects: 100% (4/4), done.
    remote: Total 7 (delta 0), reused 6 (delta 0), pack-reused 0
    Unpacking objects: 100% (7/7), done.
    Checking connectivity... done.
    root@floydhub:/floyd/home#
    root@floydhub:/floyd/home#
    

If you have [2-factor auth](https://help.github.com/articles/securing-your-account-with-two-factor-authentication-2fa/) (2FA) enabled on GitHub, you need to use a [personal access token](https://github.com/settings/tokens) instead of a password.

Before committing your changes you also need to configure your Git username and email address:
    
    
    root@floydhub:/floyd/home#
    root@floydhub:/floyd/home# cd demo-private/
    root@floydhub:/floyd/home/demo-private#
    root@floydhub:/floyd/home/demo-private# git config user.name "demouser"
    root@floydhub:/floyd/home/demo-private# git config user.email "demouser@gmail.com"
    root@floydhub:/floyd/home/demo-private#

You only have to do this once inside a workspace. After this you can commit and push your changes:
    
    
    root@floydhub:/floyd/home/demo-private#
    root@floydhub:/floyd/home/demo-private# vim README.md
    root@floydhub:/floyd/home/demo-private#
    root@floydhub:/floyd/home/demo-private# git commit -am "Update README"
    [master ca7cc59] Update README
     1 file changed, 2 insertions(+)
    root@floydhub:/floyd/home/demo-private# git push origin master
    Username for 'https://github.com': username
    Password for 'https://username@github.com':
    Counting objects: 3, done.
    Writing objects: 100% (3/3), 277 bytes | 0 bytes/s, done.
    Total 3 (delta 0), reused 0 (delta 0)
    To https://github.com/floydhub/demo-private.git
       7982e14..ca7cc59  master -> master
    root@floydhub:/floyd/home/demo-private#

### Ready to build, train, and deploy AI?

#### Get started with FloydHub's collaborative AI platform for free

###### [Try FloydHub for free ](https://www.floydhub.com/?utm_source=blog&utm_medium=banner&utm_campaign=try_floydhub_for_free)

# 3\. Submit training jobs from your workspace

Your workspace comes with the [floyd command line tool](https://docs.floydhub.com/guides/basics/install/) pre-installed and pre-configured. To run a command job, simply use the "floyd run" command just as you would do from your local machine. The jobs belong to the same project and are run on separate FloydHub instances.

![](/assets/images/content/images/2018/07/workspace_command.gif)Workspace comes with floyd-cli preinstalled. Just run "floyd run"

# 4\. Identify the best hyper-parameters

If you are trying to identify the best set of parameters for your model, you can try out various combinations as command jobs. Just make sure you are generating [training metrics](https://docs.floydhub.com/guides/jobs/metrics/#training-metrics) in your code. You can then view all your jobs under the project to identify the best performing jobs and stop the ones that not doing so well.

![](/assets/images/content/images/2018/07/job_list.png)Use description of the job to specify the parameters used. Makes it easy to compare them here.

# 5\. Get notified on Slack when your jobs finish

Setup [slack integration](https://www.floydhub.com/settings/notifications) and get immediate notifications when you training job finishes. The notification includes the job status and the values of training metrics like accuracy. If the model is not up to par - you can quickly try something different.

![](/assets/images/content/images/2018/07/slack.gif)Get slack notifications when you training finishes. Includes status and metrics!

* * *

We hope these workflows are very useful when you are using FloydHub. Do you have a cool workflow that you want to share with us? Send an email to support@floydhub.com.