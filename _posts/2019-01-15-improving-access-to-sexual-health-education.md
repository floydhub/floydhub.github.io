---
author: Charlie Harrington
date: 2019-01-15 18:06:13 +0000
excerpt: 'Irving Amukasa built an AI chatbot to answer questions about reproductive
  health for young people in Kenya. Learn more in this #humansofml interview.'
feature_image: /assets/images/hero/improving-access-to-sexual-health-education-hero.png
layout: post
slug: improving-access-to-sexual-health-education
tags: '[]'
title: Improving Access to Sexual Health Education in Kenya with Artificial Intelligence
---

Irving Amukasa wants to stop the spread of misinformation about sexual health in his home country of Kenya. 

His startup SophieBot built a conversational app that uses natural language processing (NLP) and deep learning techniques to provide users with real-time sexual and reproductive health information.

Take a look at SophieBot in action:

![](https://d2mxuefqeaa7sj.cloudfront.net/s_957965DE9A07188945F5CC6424CEF37B618E012780535A59846EA219319FA55D_1544573663724_sophie.png)SophieBot's Android app

Irving is the CEO and Co-Founder of [SophieBot](http://misssophiebot.ml/), where he’s focused on the technical and business challenges of growing his Nairobi-based startup. Irving, in his own words, is a “self-taught Android developer and a self-taught AI developer.” While he’s not scouring arxiv for NLP papers, Irving mentors high school innovators on mobile development. He’s also building an AI to play the game 2048.

I’ve learned a ton from Irving since we first met on Twitter, and I’m thrilled to share my conversation with Irving for this [Humans of Machine Learning](https://floydhub.github.io/tag/humans-of-ml/) ([#humansofml](https://twitter.com/hashtag/humansofml?src=hash)) interview. In this post, we’ll dig into the machine learning and deep learning techniques that power SophieBot, as well as Irving’s own journey building an AI startup in Africa.

**So what exactly is SophieBot?**

We’re a Kenyan AI startup that answers your questions on sexual health. Let me give you a bit of background about why we’re building this company.

In Africa, conversations about sexual health are often considered taboo and shameful — even immoral. This problem is particularly acute in Kenya, where there are many sexual and reproductive health challenges.

According to a 2014 [Kenya demographic and health survey](https://dhsprogram.com/pubs/pdf/fr308/fr308.pdf):

  * 47% of girls and 57% of boys ages 20-24 had their first sexual experience by age 18
  * 20% of teenage girls between 15-19 years old is either pregnant or has already given birth
  * Nearly 50% of new HIV infections occur among young people between the ages of 15-24
  * 56% of maternal deaths occur among girls below the age of 19

It was important to us to make sure this youthful population not only gets necessary information on sexual and reproductive health, but also accurate and helpful information — without fear of judgement, when they need it most.

**Is that why you decided to build a chatbot?**

Exactly. We recognized another major trend of increasing access to mobile devices in Kenya and Africa. We saw this technology as an opportunity — we could be the first to to deliver sexual health education via an app or a chatbot in messaging apps.

With SophieBot — we can answer important sexual and reproductive health questions for our users, no matter where they are or what they’re doing.

**How’s that been going so far?**

We have been at it since June 2016. So far, we have been asked over forty thousand questions from over 4,000 users from all over the world.

**That’s fantastic. Let’s dive into the technology behind SophieBot. How does SophieBot work?**

Sophie is a chat app that can be delivered across Android, Telegram, and Facebook Messenger. We’ve built versions for all the major messaging platforms.

Our users can ask Sophie free-form questions like:

> What is an STD?

And she’ll respond with an appropriate response, like,

> Sexually transmitted infections (STIs) are infections transmitted by having unprotected sex with an infected partner. STIs are some of the most common communicable diseases in Kenya, particularly among young people aged 15-24 years.

In the backend, whenever we get a question, we’re using parsing the sentence, and then determining what’s the best response to provide.

**I’m curious — how did your original MVP of Sophie work?**

Well, a little bit more background is necessary. I have been developing chatbots well before the hype. In 2014, as a young and naive Android developer, I had the crazy idea to build an app to automate my own personal conversations. In my usual fashion, I purchased the domain before a single line of code was written (it was going to be called Snub).

A little research led me to the [Artificial Intelligence Markup Language (AIML) framework](https://www.tutorialspoint.com/aiml/). The framework was so robust it had interpreters for every language. Soon I had a Java interpreter ready to run on an Android app. Rather than building Snub, I ended up shipping three different chatbot apps: 

  * one a bible as a bot
  * one a dictionary as a bot
  * an ALICE smalltalk bot to simulate human conversation

But, by June 2016, two years later, none of these chatbot projects had really taken off. However, this was around the time that I realized how big of a problem sexual health is in Kenya and Africa.

I saw a natural opportunity to adapt my earlier chatbot experiments with AIML to something that might be helpful in improving access to proper sexual health information.

Within a month, I had refactored the interface for my previous apps to create the first version of Sophie Bot. Like my previous chatbots, Sophie was entirely rules-based using the AIML framework.

**This the first time I’ve heard of AIML. How does it work?**

It’s an XML-like markup language that powers a "rules-engine.” I can show you an real-life example. Here’s the AIML markup that powered the very first version of Sophie:
    
    
    <category>
      <pattern>
        HI
      </pattern>
      <think>
        <set name="topic">
          Hi
        </set>
      </think>
      <template>
        Hi there, I'm Sophie. What would you like to know?
      </template>
    </category>
    
    <category>
      <pattern>
        WNDEF *
      </pattern>
      <template>
        Hmmm, Let me think about that
      </template>
    </category>
    
    <category>
      <pattern>
        WNDEF SEXUAL HEALTH
      </pattern>
      <template>
        Good health is considered to be a state of a complete physical, mental and social well being and not merely Physical well being means good health and hygiene for your genitals and related systems.Mental well A combination of all these factors makes for a (sexually) healthy YOU!.
       </template>
    </category>
    <category>
      <pattern>
        WNDEF PULLING OUT WORK
      </pattern>
      <template>
        Pulling out before the man ejaculates, known as the withdrawal method, is not a foolproof method for birth control.
      </template>
    </category>

**If I’m reading that correctly, AIML is a markup language takes a pattern and responds with a template?**

That’s exactly right.

**So how did your MVP launch go?**

They say you’re supposed to be embarrassed by your first product launch -- or else you’re launching too late. Well, I launched Sophie Bot with only literally 3 questions.

Sophie Bot could say hi back, answer only two strictly defined questions, and offer a default answer if a question wasn't defined in her knowledge base. Definitely not my ideal version of the product, but I thought it could be a useful start. My plan was to monitor new questions and add them to our knowledge base over time. 

But the response was incredible.

**What do you mean by that?**

Within a month of launching Sophie Bot, I had recruited a team of five, we had won a $10,000 in a sexual health innovation challenge by [UNFPA](https://kenya.unfpa.org/pt/node/4064), and was already featured on national press, as you can see in this YouTube video:

Even more importantly, we had answered questions about sexual health for 250 users. We were onto something with Sophie. But our approach with AIML was obviously flawed.

**What were the problems with AIML?**

By December 2016, we had 6,000 unique questions in our dataset. We quickly hit a point we had too many questions to manually keep updating our knowledge base and rules-engine. We had to find new approaches to answering our users’ questions.

**Okay, is this where you started diving into NLP and Deep Learning?**

Yes. You have to remember that, at the time, I had no background at all in data science. We were entirely learning as we were going.

We decided to dig deep into NLP techniques to solve our scaling problems. When I looked into it, I considered deep learning techniques overkill — which turned about to be quite wrong.

My first stop was [here](https://pythonprogramming.net/search/?q=nlp). That’s where we got the NLP intuition and techniques — but our next problem was translating that to our specific domain. At one point, we naively tried to repurpose a sentiment analysis model with our answers in our dataset as labels. Spoiler alert — our problem isn't a classification problem.

Next up, I found a PyData talk by Edward Bullen about an NLP bot that got me really excited, but he himself conceded it wasn’t really practical:

What he had achieved, though, is automating our earlier rule-based strategy using Python and an SQL database. He made the bot classify questions and statements, but it couldn’t do much more about delivering answers — and that was our biggest problem.

This wasn't an entirely a waste of time though. It was at the time we developed an NLP pipeline to gather even more cool insights on the data we had collected. We then could automatically extract and group topics through topic modeling and a K-means clustering.

**Can you give me a sense for some of the most common topics?**

Topic wise — most of our users are asking about STIs, safe days for sexual activity, whether “pulling out” works, and unwanted pregnancies. These are important topics for any young person to understand, which is why reproductive health education is so important for everyone.

**It sounds like you’ve recently turned back towards Deep Learning techniques. What are you working on right now with Deep Learning?**

Do you remember when I said deep learning was overkill for us? I was totally wrong. 

We hit 30,000 questions asked in April 2018 — a new peak after a successful press run from the Nairobi Innovation week. Pressure was mounting to find new promising approaches.Enter Andre Karparthy's blog on [The Unreasonable Effectiveness of Recursive Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). Yes, that's the name of the blog post. Deep into that rabbit hole, I re-discovered Long Short Term Memory Networks (LSTM). It’s not that I hadn’t heard of an LSTM implementation of bots before — I had just kept dismissing the idea. But after reading Karpathy’s post, I decided to dig back into Deep Learning concepts.

The first few times through the article, the concept was so baffling I had to walk through building an LSTM from scratch with matrix multiplications in order to get the proper intuition. We ended up writing an Encoder Decoder LSTM with an attention mechanism and trained it on our dataset. We’re still working on this concept, and hoping to deploy it soon.

**What are your future plans for improving the AI behind Sophie?**

We want to build one end-to-end model to train NLP chatbots only using previous real user conversations. Currently, "in the lab", we are testing Reinforcement learning on LSTM trained chatbots, inspired by [this implementation](https://github.com/pochih/RL-Chatbot).

More models and implementations we are prospecting include:

  1. [Knowledge Graphs](https://www.slideshare.net/christophewillemsen/knowledge-graphs-chatbots-with-neo4j)
  2. [Knowledge Graphs + Neural networks](https://arxiv.org/abs/1710.08502)
  3. [Evolutionary algorithms](https://pdfs.semanticscholar.org/d242/4a7f83be06b3abd99485ee22f8fd90cb602c.pdf)
  4. [NLP's Imagenet moment](http://ruder.io/nlp-imagenet/)

I’m very open to new ideas and suggestions, so if anyone reading this has any better ideas or suggestions on models and implementations that we can try out with our dataset, please feel free to reach out to me on Twitter! Just please don't say Dialogflow, IBMWatson, or Luis.

**Are you serving your trained models behind a deployed API? Or are you considering on-device “edge computing” approach for the Sophie app?**

Our current version of SophieBot using AIML is entirely served client-side on the app. This helps with data privacy concerns. When we switch to the LSTM model, we may look to move towards an API-based approach. But that’s to-be-determined, as we are still experimenting.

**How has data privacy impacted your research for Sophie?**

Weirdly enough, we’ve been having some issues with the Play Store. We keep getting pulled down for various reasons, and we’re working diligently with them to resolve these issues.

But anonymity of our users is our single most important value proposition. At Sophie, we are only interested in answering the questions our users ask — with no tracing to the original user. We’re constantly looking to get better at answering their questions, and you don’t need personally identifying information of any kind to do this.

**Tell me more about your team. Are you all engineers?**

![](https://lh3.googleusercontent.com/1usx8ehGTHolx8n8nLIl-W9ZhUQCXOPPNrvO_30etgY-4PQyLLpz61na9z5_a_VsAS7CLdKwAkR7HJI59SF1oor6KRJdSV0CYZyYyb-qolASftWY4jMA3S-A2SE_oCVT544e9gcJ)

We are a diverse team. Not everyone is an engineer, although we love when our tech skills rub off on each other. In this photo above, from left-to-right, you’ve got:

  * Rash, a dev
  * Derrick, my literal right-hand-man. He leads tech and maintains our live deployments
  * Then there's me
  * Beside me is Beverly — she runs operations and finances
  * John handles our social media strategy
  * Nic — handles business development

**What’s your business model?**

We’ve learned that our value is automating customer support for businesses in this space. This will enabled us to make Sophie Bot free forever for our users. That’s why we’re focusing all our efforts to build this generalized tech to serve the entire market well.

**What’s next for Sophie Bot the company?**

We are gathering steam right now. We were recently featured in a Microsoft [whitepaper](https://info.microsoft.com/ME-DIGTRNS-WBNR-FY19-11Nov-02-AIinAfrica-MGC0003244_01Registration-ForminBody.html). We’re hoping to hire an even smarter team of engineers than us to solve our problem even faster.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_DA5DE06B89E6859CEFF1A3F7DF493F1CFFC1DA8A44C5A44A04678AC7C5986F89_1544859328190_bjsd.JPG)Growth in SophieBot usage over time

**Do you have any advice for people who want to teach themselves machine learning or deep learning?**

As I mentioned earlier, I’m a self-taught developer. Nearly all of my technical learning curve has been done on the job.

> My advice, then, is start now. 

If you don’t have the time to go get a Stanford PhD, then just start using AI to solve problems you really want to solve. That’s the best way to learn.

**That was great! Thank you, Irving, for speaking with me. Where can people go to learn more about you and Sophie Bot?**

Thanks! Our company site is here: [m](http://misssophiebot.ml/)iss[[sophiebot.ml](http://misssophiebot.ml/)](http://misssophiebot.ml/). You can also check out my personal [Twitter](https://twitter.com/iamukasa).