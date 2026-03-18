\# Initial Evaluation: Base Model vs Fine-Tuned Adapter



\## Overview



This document summarizes a first-pass qualitative comparison between the base \*\*Qwen2.5-7B-Instruct\*\* model and a \*\*QLoRA fine-tuned adapter\*\* trained on the \*\*HuggingFaceH4/no\_robots\*\* dataset.



The goal of this evaluation was not to prove benchmark superiority, but to examine whether the fine-tuning run measurably changed model behavior and whether those changes appeared useful.



\## General Pattern



Across the examples reviewed, the fine-tuned adapter did change the model’s response style, but not in a uniformly positive or negative way.



The clearest overall pattern was:



\- more direct and literal instruction-following

\- more concise or implementation-oriented responses in some cases

\- occasional loss of technical precision or robustness

\- mixed impact on overall response quality



In other words, the fine-tuning run seems to have shifted \*\*style\*\* more clearly than it improved \*\*general capability\*\*.



\---



\## Example 1: Explaining QLoRA



\### Prompt

Explain QLoRA in plain English.



\### Base Model

The base model gave a broader and more technically grounded explanation. It described QLoRA as an efficient way to fine-tune large language models while avoiding the cost of changing all parameters.



\### Fine-Tuned Adapter

The fine-tuned adapter gave a cleaner and more beginner-friendly explanation, with a clearer teaching style and a more structured presentation.



\### Assessment

The adapter response was easier to read, but it introduced a technical inaccuracy in how it described QLoRA. The base model was slightly less polished stylistically, but more technically reliable.



\### Takeaway

The adapter improved readability and approachability, but sometimes at the cost of precision.



\---



\## Example 2: Palindrome Function



\### Prompt

Write a short Python function that checks whether a string is a palindrome.



\### Base Model

The base model produced a more robust answer. It normalized the string by removing non-alphanumeric characters and converting to lowercase, then compared the cleaned string to its reverse. It also included sample test cases.



\### Fine-Tuned Adapter

The adapter gave a shorter and more literal answer:



```python

def is\_palindrome(s):

&#x20;   return s == s\[::-1]

```


### Assessment



The adapter followed the instruction for a “short” function more directly, but the base model produced a more useful real-world answer by accounting for punctuation and capitalization.



\### Takeaway



The adapter tended toward brevity and literalness, while the base model remained more robust.



\---



\## Example 3: Supervised Learning vs Reinforcement Learning



\### Prompt



Summarize the main difference between supervised learning and reinforcement learning.



\### Base Model



The base model gave a tighter and more effective summary, contrasting the two approaches in terms of feedback, objective, and use cases.



\### Fine-Tuned Adapter



The adapter gave a correct explanation, but it was more verbose and less disciplined as a summary.



\### Assessment



The base model handled the summarization task better. The adapter did not improve clarity here and instead drifted toward a more generic explanatory style.



\### Takeaway



The fine-tuned adapter did not consistently improve conceptual summarization quality.



\---



\## Example 4: Coffee Shop Promotion Ideas



\### Prompt



Give me three creative ideas for a neighborhood coffee shop promotion.



\### Base Model



The base model gave an elaborate and imaginative answer with strong detail, but it tended to sprawl and became less practical.



\### Fine-Tuned Adapter



The adapter gave three clearer, more actionable suggestions:



\- coffee tasting event

\- community art show

\- book club



\### Assessment



In this case, the adapter produced the stronger response. The answer was more concise, easier to scan, and more implementation-ready.



\### Takeaway



The adapter sometimes improved usefulness by producing more direct and practical responses.



\---



\## Overall Conclusion



This initial evaluation suggests that the fine-tuned adapter successfully changed the behavior of the base model, but not in a way that can be described as simply “better” or “worse.”



Instead, the adapter appears to have shifted the model toward:



\- more literal instruction-following

\- more concise and practical answers in some settings

\- less nuance or technical precision in others



This is still a meaningful result. It demonstrates that single-GPU QLoRA fine-tuning on consumer hardware can measurably alter model behavior, but that careful qualitative evaluation is essential to determine whether those changes are desirable for a given use case.





\## Next Steps



Future evaluation improvements could include:



\- a larger prompt set

\- task-specific scoring rubrics

\- category-based evaluation by prompt type

\- benchmark-style quantitative assessment

\- comparison across multiple datasets or training configurations

