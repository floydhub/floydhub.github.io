---
author: Alessio Gozzoli
date: 2020-06-02 13:27:29 +0000
excerpt: This Humans of ML interview with Han Xiao covers the ethics of AI, open-source
  entrepreneurship, how writing made Han a better coder, and more.
feature_image: /assets/images/hero/the-future-of-ai-is-open-hero.png
layout: post
slug: the-future-of-ai-is-open
tags: '[]'
title: The Future of AI is Open
---

For this Humans of Machine Learning ([#humansofml](https://twitter.com/hashtag/humansofml)) interview, I’m thrilled to share my conversation with [Han Xiao](https://www.linkedin.com/in/hxiao87/). I had been looking forward to this interview for a while. I’ve followed Han since the early days of [his personal blog](https://hanxiao.io/) and have really been inspired by him. I’ve always found his thoughts particularly interesting, detailed and clear, and I’m excited to share them with you.  
  
After the last two years as an AI team leader at Tencent and several successful open source projects, he has just started his journey as an entrepreneur. We certainly wish all the best for him.

In this conversation, we’ll cover Han’s journey, his takes on AI development, AI ethics and collaboration across borders, and his advice for starting an AI open source software business. Hope to enjoy our chat.

### Background

** _[Alessio]:_ _I’m very excited to talk to you today. First of all, congratulations on the launch of your first startup: Jina AI 🎉_**

 _**Perhaps to start, you can give our readers a quick biography. It’d also be great to hear how you started your journey in ML & AI.**_

[Han]: I'm super excited to be here as well! I'm the founder and CEO of [Jina AI](https://jina.ai/). Jina AI is a neural search company that provides cloud-native neural search solutions powered by AI and deep learning. I've been working in the field of AI, and especially in open-source AI, for some time. You may have heard of or used my previous work on [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) and [bert-as-service](https://github.com/hanxiao/bert-as-service). From 2018 to early 2020, I was an Engineering Lead at Tencent AI Lab, where I led a team to build the search infrastructure of China's everyday app _WeChat_. From 2014 - 2018, I worked at Zalando in Berlin on recommendation and search systems. I got my Ph.D. in Machine Learning from Technical University of Munich in 2014. Besides the professional work, I'm actively involved in nonprofit organizations as well. In 2019, I was representing Tencent as the board member of Linux Foundation AI. I'm also the Founder and Chairman of the [German-Chinese Association of Artificial Intelligence](https://www.gcaai.org/).

So who set me on the path of AI? Family has a big impact on me. My father was a computer science professor, so I was fortunate to have the chance to play the computer and learn programming at a very young age. The first two AI books he bought me were _Machine Learning_ by Tom Mitchelle and _Gödel, Escher, Bach: An Eternal Golden Braid_ by Douglas Hofstadter. The first "AI" program that I wrote was a rule-based chatbot in BASIC, like ELIZA.

Apart from personal interests and curiosity, my academic journey towards ML & AI can be traced back to my undergraduate study in China. My Bachelor thesis published in 2009 was about Latent Dirichlet Allocation and Gibbs sampling. The term LDA may be less-known to AI engineers nowadays, but the graphical model and Bayesian method were prevalent back in the days, and LDA was just like BERT today! In 2009, I came to Germany and joined the Computer Science faculty of Technical University of Munich. From 2009 to 2014, while I was doing my Master and Ph.D., I have worked on various ML topics, including parametric and non-parametric Bayesian methods, kernel-based methods, SVM, Gaussian/Dirichlet process, multi-arm bandit, and adversarial learning. Many of these methods are considered "shallow learning" and not widely used from today's perspective.

_**How have you seen AI change over the timespan you’ve been involved in the field?**_

I see AI development from two aspects:

![](/assets/images/content/images/2020/05/ai-development.png)

The first aspect is from 0 to 1. Zero-to-one is all about hardcore and fundamental research, e.g., new learning and training paradigms, new computing infrastructure, and new hardware architecture.

The second aspect is from 1 to N. One-to-N focuses more on usability and engineering, addressing problems such as adapting an algorithm from one domain to multiple domains; serving a model to billions of users; improving existing user experience with AI algorithms; and piping a set of algorithms to automate the workflow, all of which belong to the one-to-N innovation.

Before 2012, which I call it the pre-deep learning era, zero-to-one innovation was quite popular. People debated on the best "intelligent" algorithm to utilize data, with different supporters backing different data-driven methods, including decision forests, margin-based approaches, kernel methods, graphical models, parametric and non-parametric Bayesian models, and neural networks. They keep iterating and improving their favorite methodology until it outperformed its peers in some benchmark. Deep neural networks also joined this party, though were neither the earliest nor the buzziest attendee.  
Things have changed since 2013. As more and more researchers and engineers have recognized deep neural networks, people have realized that deep learning is not just hype, but rather that it can solve many complicated problems with a much higher accuracy than the traditional "shallow learning." Having agreed on the methodology, many _**move away**_ from the zero-to-one and join the party of one-to-N innovation (including myself). Now in 2020, deep neural networks are the de facto solution for image classification, machine translation, and speech recognition. It serves as the backbone of many everyday products such as facial payment, voice assistants, automatic customer service, and K-12 education.

**_**How do you see the balance between the research and engineering paradigms in AI playing out?**_**

It is the best time for AI engineering. The possibilities of applying AI into everyday life is enormous. World-wise, the AI consumer market is still uncontested. There is ample opportunity for growth. In the old days, developers often said if they don't like something, they could change it and make it better. Software was eating the world. Nowadays, AI engineers can say if we don't like something, we use AI to change it and make it smarter. AI is eating software.

**_**Andrew (Ng) says that AI will become a systematic engineering discipline as well. What changes do you foresee in the AI landscape that will enable this and what open challenges exist?**_**

I agree with Andrew's view on the development of AI. I'd like to add a comment about _**the emerging AI supply chain**_ : only a few people in the world will work on building new AI models from scratch, delivering "pre-trained models" to the rest of the AI community. Most AI engineers will try to make sense of those pre-trained models by adapting them to their applications, fine-tuning based on their data, incorporating domain knowledge, and scaling them to serve millions of customers. The image above illustrates this new AI supply chain.

![](/assets/images/content/images/2020/05/ai-supply-chain.png)

The reason behind this emerging AI supply chain model is the growing cost of a good AI model. To find a good model, you need time for trial & error and money to spend on burning GPU and retaining the best AI talent. Time, money, and talent—these are all precious resources for small companies.

But even within the big tech giants, only a small number of teams can afford to train such high-profile models from scratch. Sometimes the teams don't have the scalable infrastructure and experience, and other times it is considered risky from a project management perspective.

Apart from the economic reasons, from the machine learning perspective, it’s counterintuitive to train models repetitively from scratch on every new task, especially when those tasks share the same low-level information. For example, in NLP tasks such as news classification and sentiment analysis, the low-level knowledge of a language (e.g. the grammar) remains constant. In CV tasks such as object recognition and autonomous driving, there is no point in re-learning common concepts such as colors, textures, and reflections from task to task.

### Open Source Software & Strategy

** _**What is Jina** AI**? Would you like to share with us and our readers the journey of conceiving & building Jina AI?**_**

![](/assets/images/content/images/2020/05/jina-ai-banner.gif)

Jina AI is a neural search company. [Jina](https://github.com/jina-ai/jina/) is our core product, released on April 28th, 2020. The official tagline we put on [our Github repository](https://github.com/jina-ai) is: Jina is a cloud-native neural search solution powered by state-of-the-art AI and deep learning. To put it simply, you can use Jina to search for anything: image-to-image, video-to-video, tweet-to-tweet, audio-to-audio, code-to-code, etc. To understand what we want to achieve at Jina AI, I often explain Jina with the following two expressions.

  * **_**A "TensorFlow" for search**_. **TensorFlow, PyTorch, MXNet, and Mindspore are all universal frameworks for deep learning. You can use them for recognizing cats from dogs, or playing Go and DOTA. They are powerful and universal but not optimized for a specific domain. In Jina, we are focusing on one domain only: the search. We are building on top of the universal deep learning framework and providing an infrastructure for any AI-powered search applications. The next figure illustrates how we position ourselves.

![](/assets/images/content/images/2020/05/jina-is-tf-for-search.png)

  * ****A design pattern.** **There are design patterns for every era, from functional programming to object-oriented programming. Same goes for the search system. 30 years ago, it all started with a simple textbox. Many design patterns have been proposed for implementing the search system behind this textbox, some of which are incredibly successful commercially. In the era of neural search, a query can go beyond a few keywords; it could be an image, a video, a code snippet, or an audio file. When traditional symbolic search systems cannot effectively handle those data formats, people need a new design pattern for the underlying neural search system. That's what Jina is: a new design pattern for this new era.

_**Creating a successful open-source business is probably one of the most difficult, and I assume rewarding, achievements of a tech entrepreneur. What are the challenges you think you’ll face with this business model?**_

Running an open-source software (OSS) company requires courage, an open mindset, and a strong belief.

As an OSS company, you need courage to first show the codebase to the world. The code quality is now a symbol of the company. Are you following the best practices? Are you making a tech debt here and there? Open source is an excellent touchstone to help you understand and improve the quality of software engineering and development procedures.

Embracing the community is vital for an OSS company. It requires an open mindset. Doing open source is not the same as doing a press release or a spotlight speech; it is not a one-way communication. You need to walk into the community, talk to them, solve their issues, answer their questions, and accept their criticisms. You need to manage your ego and do trivial things such as maintenance and housekeeping.

Some people may think that big tech companies hold a better position when committing to open source because they can leverage better resources. That is not true. No matter how big the company is, each has its comfort zone built over the years. For many tech companies, open source is a new game, and the value it brings is often not quantifiable through short-term KPI/OKR. The rules of play are not familiar to everyone. Not every decision-maker in the company believes in it. It's like a person who has been playing Go for years with a high rank and enjoys it. Imagine that one day you just show up and tell this guy, “Hey, let's play mahjong, mahjong is fun!” Are you expecting this guy to say "sure"? Regardless of the company's size, it is always important to make everyone inside the company believes in the value of open source. After all, it is always the individual who gets things done.

I’ve always believed that open-source AI infrastructure is the future and that community is the key. I'm sure many share the same vision as I do. But my belief is so strong that I am jumping out of a tech giant and doing Jina AI as a startup from scratch. All of my team share this belief as strongly as I do. At Jina AI, we only do what we believe.

_**You have a string of popular AI open source projects:[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), [bert-as-a-service](https://github.com/hanxiao/bert-as-service) & [GNES](https://github.com/gnes-ai/gnes). What advice would you give to open source developers and tech entrepreneurs embracing open source? How can we build valuable AI open source projects?**_

For engineers who want to do open source on AI, this is the best time. Thanks to the deep learning frameworks and off-the-shelf pre-trained models, there are many opportunities in the end-to-end application market for individuals to make significant contributions. Ask your colleagues or friends, "Which AI package do you use for daily tasks such as machine translation, image enhancement, data compression, or code completion?" You’ll get different answers from person to person. That is an indicator that the market is still uncontested and that there is ample opportunity for growth and building a community around it.

One thing I like to remind AI open-source developers is to think about the _**sustainability**_ of the project. With new AI algorithms popping up every day, how do you keep up the pace? What is the scope of your project? How do you maintain the project when facing community requests? When I was developing bert-as-service, I received many requests to extend it to AlBERT, DistilBERT, BioBERT etc. I prioritized those that fit into my roadmap. Sometimes this meant causing hard feelings. But let's be frank, you can't solve every issue, not by yourself. It is not how open source works and certainly not how you work. The most considerable risk of open-source software is that the core developers behind are burned out. The best open source isn’t the shiniest, but the one that lives the longest. So keep your enthusiasm and stay long!

Open source = Open-source code + Open governance 

In the end, doing open-source projects is like being a startup. Technical advantage is only part of the story.

Uploading the code to Github is just a starting point, and there are tasks such as operating, branding, and community management to consider. Like entrepreneurship, you need to draw a "pie" that encapsulates the passions and dreams of the community, and you need to have determination and a precise target to not get sidetracked by the community issues.

As someone with a Machine Learning Ph.D., I've never believed that some black-magic algorithm would be the competitive advantage of an open-source project. Instead, I'm more convinced that sound engineering, attention to detail, slick user experience, and a community-driven governance model ultimately determine user retention.

The most important thing is often your understanding and belief in open source. If you are an idealist, then you will inspire other idealists to march with you. If you're a detail-oriented person, every little feature in your project will be worshipped by those who care about the details. If you are a warm-hearted person, then the community you build up will appreciate your selfless giving.

Whichever person you are, it's what you believe in open source that makes open source what it is.

### Skills

 _**You’ve talked previously about the importance of sharing knowledge. What pushed you to start doing that?**_

When I was an engineering lead at Tencent, I always encouraged my teammates to share and talk more about their discoveries. I told them, “Knowledge is like money; if you only keep it in your pocket, then you don't get anything in return.” The value of knowledge is revealed when you exchange it.

I believe that making a good speech or presentation is a must-have soft skill for developers. In China, the slogan "talk is cheap, show me the code" is pretty popular in tech companies. Many developers worship this tech culture and even print it on their T-shirts. But I don't agree with that. "Geeky" culture greatly restricts one's communication ability and the desire to improve it. Instead, it gives developers a misconception that nothing is more important or efficient than code. 

In 2017, I started to write my own tech blog for one simple reason: I wanted to master TensorFlow. "If you want to learn something, read about it. If you want to understand something, write about it. If you want to master something, teach it." So be it. I decided to learn and write what I learned in a crystal-cut way so that my readers would feel like I'm teaching them.

So I write. But not on a tightly defined schedule. I'm not posting every week and trying to create buzz or impress anyone. I only write when I find something worth sharing. In the end, I have only 18 blog posts published over the last three years, so the post frequency is one about every two months. I'm not a prolific writer by any means. But two months is often the right time for me to achieve a milestone and take a retrospective of what I achieved. Over the past three years, my readers have grown from zero to five thousand monthly. I have been contacted and thanked by readers from all over the world. Even my investors read it!

**_**I’m a huge fan of your blog, too! Every time I read one of your articles I can sense the passion and also the effort you put into every single word and visualization. Can you please describe the process you follow to structure your blog articles and your presentations?**_**

There is no strict pattern of my blog posts. I write what is worth sharing. Sometimes only writing in text is not compelling enough to convey the idea, so I rely on illustrations. I use Google Slides to make all the illustrations in my blog. When I make an illustration, my purpose is the same: to convey knowledge more clearly and effectively. That includes making detailed decisions such as choosing the right color for each component and changing the line connector styles. Each artwork is revised multiple times before publishing. Writing and drawing also help you refine your ideas. I can't tell you how many times I suddenly realized the code needed to be refactored during my writing.

**_**What is your advice to our readers who aim to become better communicators?**_**

Write more. Write clearly. Just like the code needs to be refactored, good writing also needs to be revised. Write and then read it out loud. Don't begrudge yourself the words and not cut them. As a developer, if you don't have a tech blog yet, then you need one now. Don't write something like "Hello, World! Here comes my first post, and I'm going to make it bigggg!" as the first post. That often won't end well. Instead, write how you find a bug and fixed it and what you have learned from that. This not only helps you to organize your thoughts but also helps others who meet the same problem. For example, my first blog post was "[Extremely Stupid Mistakes I Made With Tensorflow and Python](https://hanxiao.io/2017/05/19/Extremely-Stupid-Mistakes-I-made-with-Tensorflow-and-Python/)".

### Across Borders

** _**You are bridging two different worlds with Jina AI in Germany and China. China’s AI tech is sometimes viewed with suspicion in Europe. There are also perhaps cultural and political biases at play. Do you see those aspects manifesting as friction in your collaborations?**_**

As long as there is a difference, there is a misunderstanding _._ Frankly, I would be surprised if a European does not question the ethics of Chinese AI development. Having stayed in Germany for many years, I'm comfortable with people's suspicions and criticisms about China and Chinese technology, just like I always question the German food "Mett" (minced raw pork). When you don't know something, you are afraid of it. We all are. That's human nature. The real question is, are you willing to talk openly about it?

In last year's Darwin's Circle Summit in Austria, I joined a panel session and was asked on stage about AI development and data privacy in China. I made an analogy.

"It's like eating soup. You European and Western guys like eating soup before the main dish, we Chinese like eating soup after the main dish. Either way, we both like soup, and we will eat it sooner or later."

The moderator laughed out loud, and many audiences raved about this analogy after the meeting. I want to convey that people in China do pay attention to data privacy and AI ethics, but each country has different methods and priorities when developing AI. There is no real difference: the ethical issues are common to all societies and countries and need to be resolved sooner or later.

After all, advancing AI technology is the common interest of mankind. Openly discussing the AI ethical problem is what every country needs. A proper understanding of the ethical risk is as important as solving those challenging problems in AI.

As an example, a leading international conference on natural language processing EMNLP 2019 received a paper on the "read-attend-comment" attention model from a university in Beijing co-authored with a US-based research lab. The paper proposed an end-to-end deep learning model that writes comments based on news content. This algorithm can be easily used for disinformation and trolling on the Internet, contributing to a more divided society. Unfortunately, that ethical risk was not mentioned in the paper. 

Sometimes researchers tend to get too caught up in the technology and forget that they are always part of society. No matter how smart you are in your field of expertise, being smart is never an excuse to avoid social responsibility.

Some might argue, “Oh please, don't you judge me. I'm just a PhD student that needs this publication to graduate!” But if you work in AI, then you know that today's AI is highly engineered. The bar has been lowered, and the technology you've just published can be deployed into practice in a few days. Knowing that it may have a negative social impact but deciding not to report the ethical issue in the paper is either a cop-out or our education on ethics and compliance in AI is falling behind. 

With gunpowder, dynamite, and nuclear power, history has shown us time and time again that whenever there is a technological advancement that can hurt mankind, it will. So why do we still have faith in technological progress? Because you cannot blame the technology itself for the way people choose to use it. Governance and open compliance is the key. Technological innovation is very dangerous without a compliance framework and ethical constraints.

**_**What difficulties will China's AI technology encounter when going international?**_**

Internationalization is hard. For many years, China has mastered copying things and often making them better than the originals. But crossing borders is not about copying, it is about understanding, especially on the culture level.

![](/assets/images/content/images/2020/05/taobao-vs-zalando.png)

I always use the Taobao App (a popular shopping app from Alibaba) as an example. When you first open this app, the over-crowded UI may scare you quite a bit with its colorful banners, shiny elements, pop-ups, and floating elements. But Chinese people enjoy it. On the contrary, many European shopping apps maintain a minimal interface with a highly restrictive color palette and simple shapes. The style difference in apps is a perfect reflection of the cultural difference. 

The Chinese shopping experience is all about navigating through bustling markets and packed crowds. It's in the hustle and bustle that you get a taste of life. Stopping in front of the tall glass windows of a European boutique and savoring the designer's new clothes in the window? Nah, that is not the Chinese way of shopping. I hardly think anyone would copy such a crowded UI and put it into the European market and expect it to be as successful as Taobao, not even with the original Taobao team.

So what is the lesson here? Cultural understanding helps a company make the step aboard. Mutual understanding is the foundation of any cross-border collaboration. Every society has its own lifestyle. If you can't understand it, trust it, don't judge it.

Today, the world seems to be at the downfall of globalization. Rabid nationalism and racism have gained popularity in multiple countries. Cross-border collaboration, especially in the high-competitive field such as AI, is at an impasse. But don't lose your faith. Nationalism and racism are eternal human themes, they rise and fall. They will fall when we realize that as humans, we really aren't so different after all.

### Personal-philosophical

 _**How is CoVid-19 affecting your world? What changes do you think companies should make in the wake of the pandemic?**_

Tech culture and best practices will be emphasized more, especially on how to work efficiently as an asynchronous distributed team. Companies that have a good tech culture seem to be affected less by CoVid-19. When face-to-face communication doesn't work anymore, how people work together also changes. How people pass on knowledge, how people collaborate with others, and how teams build trust are different now (and a good team bonding meal dinner out at a restaurant isn’t possible anymore). One thing I know for sure? Well-written documentation will be finally appreciated and actually read!

_**Before we close, let’s venture into some hypothetical terrain. How far would you say we are from AGI?**_

I can see that superintelligence is near. Machines can already outsmart humans in many complicated games by taking advantage of speed. Superintelligence will significantly exceed the cognitive performance of humans in many domains of interest. So maybe the turning point (singularity) is not about the generality of AI, but the speed. We are not far away from the day that years of human knowledge and experience can be picked up by AI in a few seconds. When that day comes, our social structure will face a radical change.

_**Can AI be conscious?**_

From my point of view, consciousness is not a vital sign for AI "species," or is not that important, at least. The real question is, do you recognize an unconsciously smart AI as an intelligent life?

**_**What is your opinion about the debate between Symbolic & Connectionist AI?**_**

Connectionism is in vogue. I rarely see such debate nowadays. But I do like that such debate happens from time to time, as it reminds us not to focus on one approach when solving problems. Instead, it reminds us to think big, look beyond, and try more ways with an open mindset.

_**What are you hoping to learn or tackle next in your career?**_

Building Jina AI into a world-level AI company is my only career mission now and in the future. Entrepreneurship is my next journey.

_**What is your top book recommendation?**_

_Gödel, Escher, Bach: An Eternal Golden Braid_ is really a good starting point for those who are interested in AI. I highly recommend it to young developers and non-developers. You don't need any programming background to read this book; you just need to be curious about intelligence itself.

_**Han** _****, thanks so much for taking the time to chat with me today. Where can people follow or reach you out****_**to build the search systems of the future** _****?****__

Thanks again for having me. I’m very active on LinkedIn, feel free to connect with me via <https://www.linkedin.com/in/hxiao87/> or Twitter (@[hxiao](https://twitter.com/hxiao)). If you are interested in knowing more about Jina and what we are doing here, make sure to follow the links below:

  * Github: <https://github.com/jina-ai/jina/>
  * Opensource: [https://opensource.jina.ai](https://opensource.jina.ai/)
  * Website: [https://jina.ai](https://jina.ai/)
  * Twitter: [@JinaAI_](https://twitter.com/JinaAI_) (with underscore at the end)
  * LinkedIn: <https://www.linkedin.com/company/jinaai>
  * Press: [press@jina.ai](mailto:press@jina.ai)