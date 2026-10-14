| model | task | condition | T | K | N | acc | unparse % | N_b | prompt | backend | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-7B-Instruct | mmlu | iid | 1.0 | 100 | 500 | 0.7042 | 0.00 | 43 | chat_template | vllm | 212.8 |
| Qwen/Qwen2.5-7B-Instruct | mmlu | sharedseed | 1.0 | 100 | 500 | 0.7049 | 0.00 | 41 | chat_template | vllm | 249.4 |
| Qwen/Qwen2.5-7B-Instruct | mmlu | sharedprompt | 1.0 | 100 | 500 | 0.7076 | 0.00 | 93 | chat_template | vllm | 1164.7 |
| Qwen/Qwen2.5-7B-Instruct | mmlu | iid | 0.5 | 30 | 500 | 0.7035 | 0.00 | 25 | chat_template | vllm | 59.5 |
| Qwen/Qwen2.5-7B-Instruct | mmlu | iid | 1.5 | 30 | 500 | 0.6989 | 0.54 | 85 | chat_template | vllm | 53.7 |
| Qwen/Qwen2.5-7B-Instruct | hellaswag | iid | 1.0 | 50 | 200 | 0.8519 | 0.00 | 16 | chat_template | vllm | 43.1 |
| Qwen/Qwen2.5-7B-Instruct | gsm8k | iid | 1.0 | 30 | 100 | 0.5097 | 0.00 | 62 | chat_template | vllm | 537.5 |
| Qwen/Qwen2.5-7B-Instruct | gsm8k | iidlong | 1.0 | 30 | 100 | 0.9033 | 0.00 | 19 | chat_template | vllm | 672.6 |
| mistralai/Mistral-7B-Instruct-v0.3 | mmlu | iid | 1.0 | 100 | 500 | 0.5708 | 0.69 | 83 | chat_template | vllm | 378.2 |
| mistralai/Mistral-7B-Instruct-v0.3 | hellaswag | iid | 1.0 | 50 | 200 | 0.6892 | 1.36 | 65 | chat_template | vllm | 86.8 |
| meta-llama/Meta-Llama-3-8B-Instruct | mmlu | iid | 1.0 | 100 | 500 | 0.6318 | 0.02 | 173 | chat_template | vllm | 176.0 |
| meta-llama/Meta-Llama-3-8B-Instruct | hellaswag | iid | 1.0 | 50 | 200 | 0.6686 | 1.65 | 86 | chat_template | vllm | 42.2 |
| meta-llama/Meta-Llama-3-8B-Instruct | mmlu | sharedprompt | 1.0 | 100 | 500 | 0.5867 | 0.04 | 264 | chat_template | vllm | 1063.1 |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | mmlu | iid | 1.0 | 100 | 500 | 0.3949 | 0.25 | 406 | chat_template | vllm | 81.3 |
| microsoft/Phi-3-mini-4k-instruct | mmlu | iid | 1.0 | 100 | 500 | 0.6425 | 1.28 | 271 | chat_template | vllm | 165.2 |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | mmlu | greedyrot | 0.0 | 4 | 500 | 0.4270 | 0.00 | 336 | chat_template | vllm | 14.4 |
| Qwen/Qwen2.5-0.5B-Instruct | mmlu | greedyrot | 0.0 | 4 | 500 | 0.3920 | 0.55 | 361 | chat_template | vllm | 67.5 |
| Qwen/Qwen2.5-1.5B-Instruct | mmlu | greedyrot | 0.0 | 4 | 500 | 0.5810 | 0.05 | 195 | chat_template | vllm | 62.6 |
| microsoft/phi-2 | mmlu | greedyrot | 0.0 | 4 | 500 | 0.5145 | 0.80 | 232 | raw | vllm | 62.7 |
| microsoft/Phi-3-mini-4k-instruct | mmlu | greedyrot | 0.0 | 4 | 500 | 0.6810 | 0.40 | 154 | chat_template | vllm | 45.1 |
| microsoft/Phi-3.5-mini-instruct | mmlu | greedyrot | 0.0 | 4 | 500 | 0.6635 | 0.30 | 157 | chat_template | vllm | 48.8 |
| Qwen/Qwen2.5-3B-Instruct | mmlu | greedyrot | 0.0 | 4 | 500 | 0.6375 | 0.00 | 183 | chat_template | vllm | 69.0 |
| Qwen/Qwen2-7B-Instruct | mmlu | greedyrot | 0.0 | 4 | 500 | 0.6740 | 0.00 | 149 | chat_template | vllm | 98.7 |
| Qwen/Qwen2.5-7B-Instruct | mmlu | greedyrot | 0.0 | 4 | 500 | 0.7240 | 0.00 | 130 | chat_template | vllm | 94.6 |
| mistralai/Mistral-7B-Instruct-v0.2 | mmlu | greedyrot | 0.0 | 4 | 500 | 0.5035 | 12.45 | 245 | chat_template | vllm | 58.7 |
| mistralai/Mistral-7B-Instruct-v0.3 | mmlu | greedyrot | 0.0 | 4 | 500 | 0.5765 | 0.00 | 223 | chat_template | vllm | 55.5 |
| HuggingFaceH4/zephyr-7b-beta | mmlu | greedyrot | 0.0 | 4 | 500 | 0.5230 | 8.45 | 227 | chat_template | vllm | 58.7 |
| allenai/OLMo-2-1124-7B-Instruct | mmlu | greedyrot | 0.0 | 4 | 500 | 0.5580 | 0.25 | 216 | chat_template | vllm | 96.1 |
| 01-ai/Yi-1.5-6B-Chat | mmlu | greedyrot | 0.0 | 4 | 500 | 0.6265 | 0.05 | 206 | chat_template | vllm | 40.8 |
| deepseek-ai/deepseek-llm-7b-chat | mmlu | greedyrot | 0.0 | 4 | 500 | 0.4755 | 1.30 | 279 | chat_template | vllm | 48.0 |
| ibm-granite/granite-3.1-8b-instruct | mmlu | greedyrot | 0.0 | 4 | 500 | 0.6270 | 0.50 | 195 | chat_template | vllm | 157.2 |
| mistralai/Ministral-8B-Instruct-2410 | mmlu | greedyrot | 0.0 | 4 | 500 | 0.5870 | 0.00 | 184 | chat_template | vllm | 53.2 |
| google/gemma-2-2b-it | mmlu | greedyrot | 0.0 | 4 | 500 | 0.5405 | 0.00 | 232 | chat_template | vllm | 78.4 |
| meta-llama/Llama-3.1-8B-Instruct | mmlu | greedyrot | 0.0 | 4 | 500 | 0.6650 | 0.00 | 185 | chat_template | vllm | 47.9 |
| google/gemma-2-9b-it | mmlu | greedyrot | 0.0 | 4 | 500 | 0.7135 | 0.00 | 146 | chat_template | vllm | 142.5 |
| mistralai/Mistral-Small-Instruct-2409 | mmlu | greedyrot | 0.0 | 4 | 500 | 0.6735 | 5.35 | 175 | chat_template | vllm | 189.8 |
| meta-llama/Llama-3.2-1B-Instruct | mmlu | greedyrot | 0.0 | 4 | 500 | 0.3325 | 1.20 | 435 | chat_template | vllm | 9.3 |
| meta-llama/Llama-3.2-3B-Instruct | mmlu | greedyrot | 0.0 | 4 | 500 | 0.5775 | 0.05 | 196 | chat_template | vllm | 22.1 |
| meta-llama/Meta-Llama-3-8B-Instruct | mmlu | greedyrot | 0.0 | 4 | 500 | 0.6380 | 0.00 | 174 | chat_template | vllm | 51.8 |
| ensemble (24 models) | mmlu | greedy across models | 0 | 24 | 500 | 0.5864 | - | 407 | - | - | - |

## R2: greedy MMLU on the 4 option rotations (pass r = rotation r)

| model | acc r0 | acc r1 | acc r2 | acc r3 | acc mean | unparse % (r0-r3) | article_ambiguous | letter share A/B/C/D % (all rotations) | near ties r0-r3 (gap < 0.13) |
|---|---|---|---|---|---|---|---|---|---|
| HuggingFaceTB/SmolLM2-1.7B-Instruct | 0.434 | 0.416 | 0.428 | 0.430 | 0.427 | 0.0/0.0/0.0/0.0 | 0 | 8/22/16/54 | 61/47/75/63 |
| Qwen/Qwen2.5-0.5B-Instruct | 0.406 | 0.386 | 0.402 | 0.374 | 0.392 | 0.8/0.4/0.4/0.6 | 0 | 21/9/12/58 | 44/59/58/58 |
| Qwen/Qwen2.5-1.5B-Instruct | 0.596 | 0.576 | 0.576 | 0.576 | 0.581 | 0.2/0.0/0.0/0.0 | 0 | 25/25/23/27 | 7/10/15/15 |
| microsoft/phi-2 | 0.528 | 0.510 | 0.512 | 0.508 | 0.514 | 0.6/0.8/1.0/0.8 | 0 | 27/27/27/19 | 93/95/104/83 |
| microsoft/Phi-3-mini-4k-instruct | 0.684 | 0.672 | 0.682 | 0.686 | 0.681 | 0.4/0.4/0.4/0.4 | 0 | 22/26/27/24 | 18/17/19/28 |
| microsoft/Phi-3.5-mini-instruct | 0.664 | 0.672 | 0.660 | 0.658 | 0.663 | 0.4/0.4/0.2/0.2 | 0 | 19/28/26/26 | 11/1/7/7 |
| Qwen/Qwen2.5-3B-Instruct | 0.642 | 0.640 | 0.622 | 0.646 | 0.637 | 0.0/0.0/0.0/0.0 | 0 | 25/29/21/26 | 3/0/3/2 |
| Qwen/Qwen2-7B-Instruct | 0.654 | 0.668 | 0.682 | 0.692 | 0.674 | 0.0/0.0/0.0/0.0 | 0 | 23/27/22/28 | 5/6/11/7 |
| Qwen/Qwen2.5-7B-Instruct | 0.710 | 0.740 | 0.720 | 0.726 | 0.724 | 0.0/0.0/0.0/0.0 | 0 | 25/26/25/24 | 6/3/1/3 |
| mistralai/Mistral-7B-Instruct-v0.2 | 0.524 | 0.520 | 0.482 | 0.488 | 0.504 | 12.4/11.0/13.2/13.2 | 4 | 32/12/26/18 | 4/3/2/1 |
| mistralai/Mistral-7B-Instruct-v0.3 | 0.574 | 0.592 | 0.560 | 0.580 | 0.577 | 0.0/0.0/0.0/0.0 | 0 | 35/19/21/25 | 11/9/8/8 |
| HuggingFaceH4/zephyr-7b-beta | 0.542 | 0.544 | 0.498 | 0.508 | 0.523 | 9.6/9.2/7.8/7.2 | 42 | 21/18/26/27 | 15/12/13/8 |
| allenai/OLMo-2-1124-7B-Instruct | 0.556 | 0.562 | 0.566 | 0.548 | 0.558 | 0.4/0.2/0.2/0.2 | 0 | 21/26/26/27 | 30/23/32/19 |
| 01-ai/Yi-1.5-6B-Chat | 0.638 | 0.624 | 0.622 | 0.622 | 0.626 | 0.0/0.0/0.0/0.2 | 0 | 22/30/26/21 | 5/1/3/2 |
| deepseek-ai/deepseek-llm-7b-chat | 0.454 | 0.470 | 0.514 | 0.464 | 0.475 | 1.8/1.2/1.2/1.0 | 0 | 24/33/19/22 | 22/22/17/31 |
| ibm-granite/granite-3.1-8b-instruct | 0.628 | 0.644 | 0.614 | 0.622 | 0.627 | 0.4/0.6/0.4/0.6 | 0 | 24/22/27/27 | 21/24/18/26 |
| mistralai/Ministral-8B-Instruct-2410 | 0.598 | 0.600 | 0.572 | 0.578 | 0.587 | 0.0/0.0/0.0/0.0 | 0 | 34/26/21/20 | 51/53/70/58 |
| google/gemma-2-2b-it | 0.568 | 0.556 | 0.536 | 0.502 | 0.540 | 0.0/0.0/0.0/0.0 | 0 | 20/18/29/33 | 13/11/10/10 |
| meta-llama/Llama-3.1-8B-Instruct | 0.684 | 0.676 | 0.652 | 0.648 | 0.665 | 0.0/0.0/0.0/0.0 | 0 | 21/25/25/29 | 22/27/22/24 |
| google/gemma-2-9b-it | 0.728 | 0.730 | 0.702 | 0.694 | 0.714 | 0.0/0.0/0.0/0.0 | 0 | 25/24/27/25 | 4/5/4/6 |
| mistralai/Mistral-Small-Instruct-2409 | 0.686 | 0.660 | 0.674 | 0.674 | 0.673 | 5.6/5.4/5.2/5.2 | 0 | 20/24/23/29 | 27/24/26/21 |
| meta-llama/Llama-3.2-1B-Instruct | 0.350 | 0.338 | 0.332 | 0.310 | 0.333 | 1.0/1.2/1.0/1.6 | 0 | 83/5/6/5 | 41/42/39/38 |
| meta-llama/Llama-3.2-3B-Instruct | 0.582 | 0.574 | 0.562 | 0.592 | 0.578 | 0.0/0.2/0.0/0.0 | 0 | 23/24/31/23 | 54/47/57/65 |
| meta-llama/Meta-Llama-3-8B-Instruct | 0.644 | 0.642 | 0.640 | 0.626 | 0.638 | 0.0/0.0/0.0/0.0 | 0 | 27/30/20/23 | 22/21/22/22 |

## R2 replicate A vs B (separate engine starts; near tie = letter gap < 0.13)

| model | letter agreement r0-r3 | correct agreement r0-r3 | letter flips | flips on near ties | near-tie rate (all cells) | texts differ | max letter-logprob diff |
|---|---|---|---|---|---|---|---|
| HuggingFaceTB/SmolLM2-1.7B-Instruct | 1.000/1.000/1.000/1.000 | 1.000/1.000/1.000/1.000 | 0 | 0 | 0.123 | 0 | 0.0 |
| Qwen/Qwen2.5-0.5B-Instruct | 1.000/1.000/1.000/1.000 | 1.000/1.000/1.000/1.000 | 0 | 0 | 0.110 | 0 | 0.0 |
| Qwen/Qwen2.5-1.5B-Instruct | 1.000/1.000/1.000/1.000 | 1.000/1.000/1.000/1.000 | 0 | 0 | 0.024 | 0 | 0.0 |
| microsoft/phi-2 | 1.000/1.000/1.000/1.000 | 1.000/1.000/1.000/1.000 | 0 | 0 | 0.188 | 0 | 0.0 |
| microsoft/Phi-3-mini-4k-instruct | 1.000/1.000/1.000/1.000 | 1.000/1.000/1.000/1.000 | 0 | 0 | 0.041 | 0 | 0.0 |
| microsoft/Phi-3.5-mini-instruct | 1.000/1.000/1.000/1.000 | 1.000/1.000/1.000/1.000 | 0 | 0 | 0.013 | 0 | 0.0 |
| Qwen/Qwen2.5-3B-Instruct | 1.000/1.000/1.000/1.000 | 1.000/1.000/1.000/1.000 | 0 | 0 | 0.004 | 0 | 0.0 |
| Qwen/Qwen2-7B-Instruct | 1.000/1.000/0.994/0.994 | 1.000/1.000/0.994/0.996 | 6 | 6 | 0.024 | 16 | 1.1219 |
| Qwen/Qwen2.5-7B-Instruct | 1.000/1.000/1.000/1.000 | 1.000/1.000/1.000/1.000 | 0 | 0 | 0.006 | 0 | 0.4994 |
| mistralai/Mistral-7B-Instruct-v0.2 | 1.000/1.000/1.000/1.000 | 1.000/1.000/1.000/1.000 | 0 | 0 | 0.005 | 0 | 0.4906 |
| mistralai/Mistral-7B-Instruct-v0.3 | 1.000/1.000/1.000/1.000 | 1.000/1.000/1.000/1.000 | 0 | 0 | 0.018 | 0 | 0.0 |
| HuggingFaceH4/zephyr-7b-beta | 1.000/1.000/1.000/1.000 | 1.000/1.000/1.000/1.000 | 0 | 0 | 0.024 | 0 | 0.0 |
| allenai/OLMo-2-1124-7B-Instruct | 1.000/1.000/1.000/1.000 | 1.000/1.000/1.000/1.000 | 0 | 0 | 0.052 | 0 | 0.0 |
| 01-ai/Yi-1.5-6B-Chat | 1.000/1.000/1.000/1.000 | 1.000/1.000/1.000/1.000 | 0 | 0 | 0.005 | 0 | 0.0 |
| deepseek-ai/deepseek-llm-7b-chat | 1.000/1.000/1.000/1.000 | 1.000/1.000/1.000/1.000 | 0 | 0 | 0.046 | 0 | 0.2266 |
| ibm-granite/granite-3.1-8b-instruct | 1.000/1.000/1.000/1.000 | 1.000/1.000/1.000/1.000 | 0 | 0 | 0.044 | 0 | 0.0 |
| mistralai/Ministral-8B-Instruct-2410 | 1.000/1.000/1.000/1.000 | 1.000/1.000/1.000/1.000 | 0 | 0 | 0.116 | 0 | 0.0002 |
| google/gemma-2-2b-it | 1.000/1.000/1.000/1.000 | 1.000/1.000/1.000/1.000 | 0 | 0 | 0.022 | 0 | 0.0 |
| meta-llama/Llama-3.1-8B-Instruct | 1.000/1.000/1.000/1.000 | 1.000/1.000/1.000/1.000 | 0 | 0 | 0.048 | 0 | 0.0 |
| google/gemma-2-9b-it | 1.000/1.000/1.000/1.000 | 1.000/1.000/1.000/1.000 | 0 | 0 | 0.009 | 0 | 0.0 |
| mistralai/Mistral-Small-Instruct-2409 | 1.000/1.000/1.000/1.000 | 1.000/1.000/1.000/1.000 | 0 | 0 | 0.049 | 0 | 0.0 |
| meta-llama/Llama-3.2-1B-Instruct | 1.000/1.000/1.000/1.000 | 1.000/1.000/1.000/1.000 | 0 | 0 | 0.080 | 0 | 0.1074 |
| meta-llama/Llama-3.2-3B-Instruct | 1.000/1.000/1.000/1.000 | 1.000/1.000/1.000/1.000 | 0 | 0 | 0.112 | 0 | 0.0 |
| meta-llama/Meta-Llama-3-8B-Instruct | 1.000/1.000/1.000/1.000 | 1.000/1.000/1.000/1.000 | 0 | 0 | 0.043 | 0 | 0.0958 |
| total | | | 6 | 6 | | | |

## v2 greedy (re-parsed with v3.1) vs R2 replicate A rotation 0 (same questions, original order)

| model | letter agreement | letters differ | correct agreement | acc v2 | acc A r0 | v2 source |
|---|---|---|---|---|---|---|
| HuggingFaceTB/SmolLM2-1.7B-Instruct | 0.984 | 8 | 0.990 | 0.440 | 0.434 | archive/results/v2_greedy |
| Qwen/Qwen2.5-0.5B-Instruct | 0.958 | 21 | 0.974 | 0.400 | 0.406 | archive/results/v2_greedy |
| Qwen/Qwen2.5-1.5B-Instruct | 0.980 | 10 | 0.986 | 0.598 | 0.596 | archive/results/v2_greedy |
| microsoft/phi-2 | 0.940 | 30 | 0.964 | 0.524 | 0.528 | archive/results/v2_greedy |
| microsoft/Phi-3-mini-4k-instruct | 0.984 | 8 | 0.990 | 0.690 | 0.684 | archive/results/v2_greedy |
| microsoft/Phi-3.5-mini-instruct | 0.992 | 4 | 0.992 | 0.668 | 0.664 | archive/results/v2_greedy |
| Qwen/Qwen2.5-3B-Instruct | 0.992 | 4 | 0.992 | 0.642 | 0.642 | archive/results/v2_greedy |
| Qwen/Qwen2-7B-Instruct | 0.994 | 3 | 0.994 | 0.652 | 0.654 | archive/results/v2_greedy |
| Qwen/Qwen2.5-7B-Instruct | 0.984 | 8 | 0.986 | 0.696 | 0.710 | archive/results/v2_greedy |
| mistralai/Mistral-7B-Instruct-v0.2 | 0.992 | 4 | 0.994 | 0.522 | 0.524 | archive/results/v2_greedy |
| mistralai/Mistral-7B-Instruct-v0.3 | 0.996 | 2 | 0.996 | 0.574 | 0.574 | archive/results/v2_greedy |
| HuggingFaceH4/zephyr-7b-beta | 0.990 | 5 | 0.992 | 0.542 | 0.542 | archive/results/v2_greedy |
| allenai/OLMo-2-1124-7B-Instruct | 0.844 | 78 | 0.908 | 0.548 | 0.556 | archive (non-canonical v2 prompt ids) |
| 01-ai/Yi-1.5-6B-Chat | 0.880 | 60 | 0.914 | 0.628 | 0.638 | archive (non-canonical v2 prompt ids) |
| deepseek-ai/deepseek-llm-7b-chat | 0.448 | 276 | 0.678 | 0.348 | 0.454 | archive (non-canonical v2 prompt ids) |
| ibm-granite/granite-3.1-8b-instruct | 0.986 | 7 | 0.990 | 0.626 | 0.628 | archive/results/v2_greedy |
| mistralai/Ministral-8B-Instruct-2410 | 0.978 | 11 | 0.990 | 0.596 | 0.598 | archive/results/v2_greedy |
| google/gemma-2-2b-it | 0.998 | 1 | 1.000 | 0.568 | 0.568 | archive/results/v2_greedy |
| meta-llama/Llama-3.1-8B-Instruct | 0.980 | 10 | 0.982 | 0.686 | 0.684 | archive/results/v2_greedy |
| google/gemma-2-9b-it | 0.996 | 2 | 0.998 | 0.730 | 0.728 | archive/results/v2_greedy |
| mistralai/Mistral-Small-Instruct-2409 | 0.988 | 6 | 0.994 | 0.680 | 0.686 | archive/results/v2_greedy |
| meta-llama/Llama-3.2-1B-Instruct | 0.990 | 5 | 0.996 | 0.350 | 0.350 | archive/results/v2_greedy |
| meta-llama/Llama-3.2-3B-Instruct | 0.984 | 8 | 0.990 | 0.580 | 0.582 | archive/results/v2_greedy |
| meta-llama/Meta-Llama-3-8B-Instruct | 0.990 | 5 | 0.992 | 0.644 | 0.644 | archive/results/v2_greedy |
