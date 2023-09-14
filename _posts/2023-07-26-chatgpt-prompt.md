---
layout: post
title: 'Notes for ChatGPT Prompt Engineering for Developers'
date: 2023-07-26
permalink: /posts/2023/07/chatgpt-prompt/
excerpt: "A tutorial of using prompt."
category:
  - NLP

---
# Basic Information
Course Name: [ChatGPT Prompt Engineering for Developers](https://learn.deeplearning.ai/chatgpt-prompt-eng/lesson/1/introduction) \
GitHub: [chatgpt-prompt-engineering](https://github.com/ralphcajipe/chatgpt-prompt-engineering)


# Two types of large language models(LLMs)
## Base LLM 
* Predicts next word, based on text training data

**Example1:** \
Once Upon a time, there was a unicorn *that lived in a magical forest with all her unicorn friends.*

## Instruction Tuned LLM

* Tries to follow instructions.
* Fine-tune on instructions and good attempts at following those instructions.
* RLHF: Reinforcement learning with Human Feedback, to make the system better able to be helpful and follow instructions.
* Helpful, honest, harmless.

**Example 2:** \
What is the captial of France?
*The captial of France is Paris.*




# Guideline for Prompting
## Principles of Prompting
* Principle 1: Write clear and specfic instructions
Longer prompt provides much more clarity and context for the model, which can actually lead to more detailed and relevant output.




**Tactic 1: Use delimiters**

Use delimiter to specific the content.
* Triple quotes: """
* Triple backticks: ```
* Triple dashes: ---
* Angle brackets: <>
* XML tags:<tag> </tag>

**Tactic 2: Ask for structured output**

HTML, JSON

**Tactic 3: Chekc whether conditions are satisfied**

Check assumptions required to do the task

**Tactic 4: Few-shot prompting**

* Give successful examples of completing tasks

* Then ask model to perform the task




## Principle 2: Give the model time to think

**Tactic 1: Specify the steps required to complete a task (ask for output in a specified format)**
* Step 1: ...
* Step 2: ...
* ...
* Step N: ...

**Tactic 2: Instruct the model to work out its own solution before rushing to a conclusion**
* Model Limitations: Hallucinations
* Hallucinations: Makes statements that sound plausible but are not true.
* Reducing hallucinations: First find relevant information, then answer the question based on the relevant information. 







# Iterative 
<img src='/images/chatgpt-prompt-engineering/iterative.jpg'>
Tuning the prompt is similar with the model tuning, that you need to adjust the prompt util reaching the expected results.

Iterative process is trying something first, analyzing where the result does not give what you want. Then clarify instructions, give the model more time to think adn refine prompts with a batch of examples.

# Summary & Inferring & Transforming

* Sentiment  Recognition
* Machine translation
* Named Entity Recognition
* Translation
* Reading Comprehension
* ...

LLM can do a lot NLP tasks by using the varities of Prompts.


# Expanding
<img src='/images/chatgpt-prompt-engineering/temperature.png'>
Temperature: randomness of the model, when the temperature is low, it generate the reliable results, otherwise the answer become variety as the temperature higher.

# Chatbot

messages = [{"role": "", "content":"" }]

Roles: {system, user, assistant}

A full conversation contains dialogs from user and assistant, the system role is to set behavior of assistants.

# Conclusion
* A good prompt can help the model generate far better response. 
* Zero-shot heavily rely on model's ability of understand and expressing. Few-shot could give some examples that model could follow and give more reasonable answer.
* Chain Of Thought can be helpful on logic problems.