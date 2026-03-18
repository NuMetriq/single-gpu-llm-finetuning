\# Experiment: 500-Example Fine-Tuning Run



\## Objective



This experiment scaled the original debug fine-tuning run from 100 training examples to 500 training examples while keeping the rest of the setup fixed.



\## What Changed



\- Training examples: 100 → 500

\- Evaluation examples: 50 → 100

\- Base model: unchanged

\- Dataset: unchanged

\- Method: unchanged

\- Hardware: unchanged

\- Evaluation prompts: unchanged



\## Hypothesis



Training on a larger subset might produce a stronger and more stable behavioral shift than the 100-example run.



\## Results Summary



The 500-example run did produce a stronger behavioral shift, but not a clearly better one.



Compared with the 100-example adapter, the 500-example adapter appeared:



\- more concise

\- more literal in following prompt wording

\- less elaborate

\- less nuanced

\- less creative on open-ended prompts



In several examples, the larger run amplified the same stylistic pattern seen in the smaller run, but in a more extreme form.



\## Prompt-Level Observations



\### QLoRA explanation

The 500-example adapter gave a shorter but less informative and less precise explanation than both the base model and the 100-example adapter.



\### Professional rewrite

The 500-example adapter produced a clean and professional rewrite. This was one of the stronger results from the larger run.



\### Coffee shop promotion

The 500-example adapter regressed noticeably. Its suggestions became repetitive and much less creative than the 100-example adapter.



\### Supervised learning vs reinforcement learning

The 500-example adapter remained correct, but did not improve summarization quality relative to the base model.



\### Palindrome function

The 500-example adapter produced the shortest and most literal answer, but also the least robust and least informative one.



\## Conclusion



The 500-example run amplified the stylistic effects of fine-tuning, especially brevity and literal instruction-following, but it did not clearly improve general response quality.



Compared with the 100-example run, the larger run often appeared more compressed and less nuanced. This suggests that simply increasing the number of training examples did not automatically produce a better adapter under the current training configuration.



\## Interpretation



This result is still valuable. It shows that:



\- single-GPU QLoRA fine-tuning can reliably change model behavior

\- a larger fine-tuning subset can amplify those changes

\- stronger behavioral change is not the same as better task performance

\- evaluation is necessary to judge whether a fine-tuned adapter is actually useful



\## Next Steps



Potential follow-up experiments:

\- try a lower learning rate

\- train for fewer or more carefully tuned steps

\- use a more diverse evaluation set

\- compare multiple dataset sizes systematically

\- test whether a different base model or dataset produces a more desirable shift

