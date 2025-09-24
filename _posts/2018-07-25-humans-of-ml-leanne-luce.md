---
layout: "post"
title: "Revolutionizing the Fashion Industry with Artificial Intelligence"
date: "2018-07-25 14:12:20 +0000"
slug: "humans-of-ml-leanne-luce"
author: "Charlie Harrington"
excerpt: "The FloydHub #HumansOfML Interview with Leanne Luce -- designer, technologist, and author of the upcoming book \"Artificial Intelligence for Fashion.\""
feature_image: "__GHOST_URL__/content/images/2018/07/Leanne-Luce---Color-Square.jpeg"
tags: "[]"
---

Leanne Luce is a designer and technologist who builds wearable systems, new manufacturing processes, and software for the fashion industry. She’s also the founder of [The Fashion Robot](https://thefashionrobot.com) — a blog about AI, robotics, culture, and fashion. Since graduating from the Rhode Island School of Design in 2013, Leanne has focused her career on the intersection of innovation and apparel design. Leanne has worked at Harvard University and Otherlab on the development of soft actuators and human interfaces for state of the art powered exoskeletons.

Her latest project is her forthcoming book _Artificial Intelligence for Fashion: How AI is Revolutionizing the Fashion Industry_ , published by Apress and available now on [Amazon](https://amzn.to/2uW5gKF). 

![Leanne Luce's book: Artificial Intelligence for Fashion: How AI is Revolutionizing the Fashion Industry](/assets/images/content/images/2019/08/book.jpg)

While we’re waiting for our copy to arrive, we chatted with Leanne about her book and how she got started with training machine learning models with FloydHub.

### Ready to build, train, and deploy AI?

#### Get started with FloydHub's collaborative AI platform for free

###### [Try FloydHub for free ](https://www.floydhub.com/?utm_source=blog&utm_medium=banner&utm_campaign=try_floydhub_for_free)

****You’re at the center of an inherently changing industry**** —****my outdated but potentially coming-back-in-style cargo shorts and Tevas speak to that. But it sounds like some of the recent advances in AI are changing fashion at an even more fundamental level. For those of us outside of the fashion world, what’s going on right now with AI and fashion?****

Before I answer your question, I’m really excited that you brought up your personal style to kick us off. I think there’s sometimes a perception that there is a dividing line between people who care about fashion and people who care about AI or machine learning. It’s one of the reasons that I felt it was really important to write this book.

In the fashion industry, AI is touching every piece of the value chain from consumer-side product discovery all the way back to manufacturing. Companies built with AI and data at the core of their businesses, like Stitch Fix, are outperforming traditional businesses in the fashion industry. [[1](https://www.forbes.com/sites/andriacheng/2018/06/08/stitch-fix-just-proves-this-again-data-is-the-new-hit-fashion/#de3687b2059f)]

Fashion has been in desperate times recently. Last year, 2017, was nicknamed “the retail apocalypse” by Business Insider because of over 8,000 store closures. [[2](https://www.businessinsider.com/retailers-bankruptcy-filings2017-9)] As brick-and-mortar shopping is flailing, brands are looking to up their e-commerce game and AI is a part of their solution. Actually, in the Business of Fashion / McKinsey Annual Report, they reported that 75% of fashion retailers planned to invest in AI in 2018/2019. [[3](https://cdn.businessoffashion.com/reports/The_State_of_Fashion_2018_v2.pdf)] These pain points are accelerating adoption across the industry.

****Why did you want to publish a book about this topic?****

There were a few reasons why this book felt important to write. There was definitely a need that I saw, but for me it was also personal. I studied apparel design and I admire the industry, but I felt immediately after graduating that it was an industry in pain. When I was trying to learn about AI myself, I kept coming across two types of articles: articles written by people with a technical background full of equations and code, and on the other side of the spectrum business articles that said nothing at all about the underlying technology. I really wanted to create a book that would split the difference of these two and make AI accessible to anyone with the curiosity to pick it up.

****Before we go any further, can we do a Reddit-style ELI5 (Explain Like I’m Five) for machine learning? Give us your best shot.****

I have a 5 year old nephew, I’ll have to try this out on him… machine learning is when humans teach computers to learn and then teach humans something.

****That was great! You know, getting started with AI and machine learning can often seem intimidating to people without strong backgrounds in math or coding. I love that your book explicitly does not include any “equations, algorithms, or code.” How are you breaking through these technical jargon barriers for your readers?****

Diving into AI and machine learning is incredibly intimidating, maybe more intimidating than blockchain even? 😝 It was really important to me that anyone could pick up this book and learn something. The book focuses on high-level concepts and emphasizes the reasons why the technology is important and how it can be used. If I have to explain what a database is to explain why NLP is important, then I do.

****As part of your research for your book, you trained your own deep learning models on FloydHub. Can you tell us a bit more about that? Why did you do that? And what exactly did you build on FloydHub?****

I did! As I was writing the book, I knew I wanted to talk about GANs (Generative Adversarial Networks) because the fashion industry is such a visual industry and GANs are an exciting area right now in machine learning. I had talked about training a GAN to generate Fashion Blogger images with a friend of mine over a year ago, but never quite made it happen. 

When it came time to write the Chapter on GANs, I really wanted to get hands-on, so I started looking for examples to learn from. I came across this [blog post](https://medium.com/@rayheberer/training-a-tensorflow-pix2pix-cgan-on-floydhub-to-generate-body-transformations-2e550e287804) and it talked about using FloydHub, which led me to your [DCGAN example page](https://docs.floydhub.com/examples/dcgan/). I implemented that model almost verbatim. I had a small dataset and some issues with overfitting, so I added dropout to help address that.

The images that I generated with this model are small images, very gestural and still abstract, but I kind of love how they capture the essence of the Fashion blogger.

![Output from Leanne's DCGAN-Fashion-Blogger project on FloydHub](/assets/images/content/images/2019/08/generated.png)Output from Leanne's DCGAN-Fashion-Blogger project on FloydHub

****Are your machine learning models on FloydHub public? How can people try out your machine learning models themselves?****

Yes! Anyone who wants to use the model, can find it here: <https://www.floydhub.com/leanneluce/projects/dcgan-fashion-blogger>

I’ll also be posting a step-by-step on how to us it on my blog **thefashionrobot.com** in the coming weeks. I’ve been calling the project #robotswearingclothes.

****You’ve focused on the fashion industry with your work -- but what other fields or industries might be similarly impacted by AI? Do you have any advice for people in these industries?****

I have focused on fashion, but AI is broadly impacting every industry. One that I’m also kind of interested in is the home improvement and construction industry. It has some of the characteristics of the fashion industry in that it is both functional and deeply rooted in craft.

In general, it seems like AI is reaching an inflection point. Tools available now, like FloydHub, are making it easier than ever before to start machine learning projects for a wide range of users. My advice is always the same when it comes the topic of technology: be curious and get good at learning.

****What’s next for you?****

Great question! Right now, I’m working on promoting my book and starting to explore business opportunities in the space.

****Where can people go to learn more about you and your work?****

Anyone who is interested in following my work can follow my blog and subscribe to my mailing list on thefashionrobot.com. Thank you to everyone at FloydHub!