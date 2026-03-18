# Single-GPU LLM Fine-Tuning with QLoRA on an RTX 3080 Ti

Parameter-efficient fine-tuning of **Qwen2.5-7B-Instruct** on the **No Robots** dataset using **QLoRA** on a single **NVIDIA RTX 3080 Ti (12 GB VRAM)**.

## Overview

This project explores whether a modern 7B instruction-tuned language model can be fine-tuned locally on consumer hardware using parameter-efficient methods rather than full fine-tuning.

The project uses:

- **Base model:** Qwen2.5-7B-Instruct
- **Dataset:** HuggingFaceH4/no_robots
- **Method:** QLoRA
- **Hardware:** NVIDIA RTX 3080 Ti (12 GB VRAM)
- **Training approach:** single-GPU local fine-tuning with Unsloth

The main goal was to build a portfolio-quality end-to-end LLM fine-tuning pipeline, including environment setup, data loading, training, adapter saving, inference, and evaluation.

## Motivation

Large language model fine-tuning is often discussed as though it requires expensive multi-GPU infrastructure. This project tests a more practical question:

**How far can a single consumer GPU go for local instruction fine-tuning?**

This repo is meant to demonstrate:

- practical understanding of local LLM workflows
- familiarity with parameter-efficient fine-tuning
- ability to work through environment and compatibility issues
- disciplined evaluation of fine-tuned behavior rather than just reporting that training completed

## Project Goals

- Fine-tune a 7B instruct model locally on a single GPU
- Use a public instruction dataset appropriate for portfolio work
- Save and reload LoRA adapters successfully
- Compare the base model against the fine-tuned adapter on shared prompts
- Document the process, results, and limitations clearly

## Hardware

- **GPU:** NVIDIA GeForce RTX 3080 Ti
- **VRAM:** 12 GB
- **Platform:** Windows
- **CUDA:** 12.1

## Model and Dataset

### Base Model
- **Qwen2.5-7B-Instruct**

### Dataset
- **HuggingFaceH4/no_robots**

The No Robots dataset was chosen because it is a compact, instruction-oriented dataset suitable for experimentation and portfolio work. The dataset includes prompt-response style conversational data organized through a `messages` field.

## Method

This project uses **QLoRA** rather than full fine-tuning.

Why QLoRA:

- full fine-tuning of a 7B model is not realistic on a 12 GB GPU
- QLoRA dramatically reduces memory requirements
- adapter-based training makes it practical to fine-tune and evaluate large models locally

Training was ultimately run with **Unsloth** after issues with the standard Hugging Face trainer stack.

## Repository Structure

```text
.
├── data/
│   └── eval_prompts.json
├── outputs/
│   ├── eval/
│   └── qwen-no-robots-qlora-unsloth/
├── reports/
│   └── initial_eval.md
├── src/
│   ├── gpu_check.py
│   ├── inspect_dataset.py
│   ├── train.py
│   ├── inference.py
│   └── evaluate.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```



## Setup



### 1. Clone the repository



```PowerShell
git clone https://github.com/NuMetriq/single-gpu-llm-finetuning.git
cd single-gpu-llm-finetuning
```



### 2. Create and activate a virtual environment



```PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```



### 3. Install dependencies



```PowerShell
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install datasets peft accelerate bitsandbytes
pip install --no-deps git+https://github.com/unslothai/unsloth-zoo.git
pip install "unsloth[cu121-torch250] @ git+https://github.com/unslothai/unsloth.git" --no-build-isolation
pip uninstall -y torchao
```



## Training



A debug-scale training run was used first to prove the pipeline worked before scaling up.



Example training command:



```PowerShell
python src\train.py
```



The training script:

- loads Qwen2.5-7B-Instruct in 4-bit
- formats the No Robots dataset into chat text
- applies LoRA adapters
- fine-tunes using QLoRA
- saves the adapter locally



## Inference and Evaluation



After training, the adapter was evaluated by comparing:

- **base model outputs**
- **fine-tuned adapter outputs**

on the same prompt set.



Example inference command:



```PowerShell
python src\inference.py
```



Evaluation outputs were saved to:



```
outputs/eval/base_vs_adapter_results.json
```



## Evaluation Summary



I evaluated the fine-tuned adapter qualitatively by comparing it against the base model on the same set of prompts.

The adapter clearly changed the model’s behavior, but not in a uniformly positive or negative way. Instead, it appears to have shifted the model toward more direct, literal, and practically framed responses.

Observed strengths of the fine-tuned adapter:

- more concise and scannable answers
- stronger literal instruction-following
- more practical, implementation-oriented suggestions in some cases

Observed weaknesses of the fine-tuned adapter:

- occasional loss of technical precision
- reduced robustness on some coding-style prompts
- less disciplines summarization in some explanatory tasks

Observed strengths of the base model:

- stronger nuance and technical grounding
- better robustness in some answers
- better summarization quality in certain conceptual prompts

Overall, the fine-tuning run appears to have changed response style more clearly than it improved general capability. In some prompts, that stylistic shift made the model more useful; in others, it reduced precision or depth. This is a meaningful result: single-GPU QLoRA fine-tuning on consumer hardware can measurably alter model behavior, but careful evaluation is necessary to determine whether those changes are desirable.



## Example Finding



### 1. Plain-English explanation of QLoRA


On a conceptual explanation prompt, the fine-tuned adapter produced a cleaner and more beginner-friendly answer, but it introduced a technical inaccuracy in how it described QLoRA. The base model was slightly less tidy, but more technically grounded.

**Takeaway**: the adapter improved readability, but sometimes at the cost of precision.

### 2. Short Python palindrome function


On a coding-style prompt, the adapter followed the instruction more literally and returned a shorter function. However, the base model provided a more robust solution that handled punctuation and capitalization and included better examples.

**Takeaway**: the adapter tended toward brevity and literalness, while the base model remained more robust.

### 3. Neighborhood coffee shop promotion ideas


On a business-idea prompt, the adapter gave a stronger response. Its answer was concise, practical, and immediately usable. The base model was more elaborate and imaginative, but less focused.

**Takeaway**: the adapter sometimes improved usefulness by producing more direct and implementation-ready responses.



## Challenges Encountered

This project involved a number of practical engineering issues that are common in local LLM work:

- Windows long-path problems during package installation
- Hugging Face trainer / TRL API mismatches across versions
- mixed-precision training failures involving bf16/amp interactions
- environment conflicts involving `xformers`, `torchao`, and Unsloth
- VRAM limits during inference when trying to load both base and adapter models simultaneously
- fallback from Unsloth inference to standard Transformers + PEFT for more reliable evaluation

Resolving these issues was a major part of the project and reflects the real-world complexity of local LLM fine-tuning.



## Key Takeaways

- A **7B model can be fine-tuned locally** on a **3080 Ti** using QLoRA
- The main bottlenecks are often **software compatibility and memory management**, not just raw compute
- Fine-tuning can clearly shift model behavior, but evaluation is necessary to understand whether the shift is actually desirable
- Adapter-based workflows are much more practical than full fine-tuning on consumer GPUs



## Limitations

- This project used a relatively small-scale initial run
- Evaluation was qualitative rather than benchmark-based
- The fine-tuned adapter does not consistently outperform the base model
- Results are sensitive to training configuration, dataset subset size, and prompt selection



## Future Improvements

- scale training to a larger portion of the dataset
- add a more systematic evaluation suite
- test additional datasets and prompt categories
- compare multiple base models
- evaluate whether the adapter improves consistency within specific task domains



## License

This repository is licensed under the **MIT License** for original code and documentation.

Model, dataset, and derivative-weight usage remain subject to their respective upstream licenses.



## Acknowledgments

- **Qwen** for the base model
- **Hugging Face** for model and dataset hosting
- **Unsloth** for enabling efficient local fine-tuning
- **HuggingFaceH4/no_robots** for the public instruction dataset