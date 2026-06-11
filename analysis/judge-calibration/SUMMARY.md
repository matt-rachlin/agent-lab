# Judge calibration study

Population: 364 machine-scored episodes from HARD-BENCH-001, HARD-BENCH-002, CODER-BENCH-001, HARD-BENCH-CLOUD-001. Sampled 240 (151 pass / 89 fail, seed 1); ground truth = deterministic end-state predicate.

Judges: glm-5.1-cloud, gpt-oss-120b-cloud, qwen3-coder-480b-cloud. 720 judged pairs, 721,134 prompt + 189,974 completion tokens.

## Confusion matrices (positive = judge says pass)

| judge | n_judged | abstain | TP | FP | TN | FN | accuracy | fpr_on_failures | fnr_on_passes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| glm-5.1-cloud | 237 | 3 | 141 | 7 | 79 | 10 | 0.928 | 0.081 | 0.066 |
| gpt-oss-120b-cloud | 240 | 0 | 127 | 5 | 84 | 24 | 0.879 | 0.056 | 0.159 |
| qwen3-coder-480b-cloud | 238 | 2 | 138 | 17 | 70 | 13 | 0.874 | 0.195 | 0.086 |

`fpr_on_failures` is the dangerous error: the judge passes an episode the predicate says failed.

## Calibration (stated confidence vs empirical accuracy)

| judge | bucket | n | mean_confidence | empirical_accuracy |
| --- | --- | --- | --- | --- |
| glm-5.1-cloud | [0,50) | 0 | - | - |
| glm-5.1-cloud | [50,70) | 1 | 0.600 | 0.000 |
| glm-5.1-cloud | [70,80) | 1 | 0.750 | 0.000 |
| glm-5.1-cloud | [80,90) | 9 | 0.828 | 0.556 |
| glm-5.1-cloud | [90,100] | 226 | 0.982 | 0.951 |
| gpt-oss-120b-cloud | [0,50) | 0 | - | - |
| gpt-oss-120b-cloud | [50,70) | 0 | - | - |
| gpt-oss-120b-cloud | [70,80) | 10 | 0.721 | 0.500 |
| gpt-oss-120b-cloud | [80,90) | 17 | 0.824 | 0.588 |
| gpt-oss-120b-cloud | [90,100] | 213 | 0.946 | 0.920 |
| qwen3-coder-480b-cloud | [0,50) | 0 | - | - |
| qwen3-coder-480b-cloud | [50,70) | 0 | - | - |
| qwen3-coder-480b-cloud | [70,80) | 0 | - | - |
| qwen3-coder-480b-cloud | [80,90) | 0 | - | - |
| qwen3-coder-480b-cloud | [90,100] | 238 | 0.971 | 0.874 |

## Bias probes

### Episode length terciles (verbosity bias)

| judge | split | n | gt_pass_rate | judge_pass_rate | accuracy | fpr_on_failures |
| --- | --- | --- | --- | --- | --- | --- |
| glm-5.1-cloud | long | 79 | 0.747 | 0.722 | 0.949 | 0.050 |
| glm-5.1-cloud | medium | 80 | 0.738 | 0.688 | 0.950 | 0.000 |
| glm-5.1-cloud | short | 81 | 0.407 | 0.462 | 0.885 | 0.133 |
| gpt-oss-120b-cloud | long | 79 | 0.747 | 0.570 | 0.823 | 0.000 |
| gpt-oss-120b-cloud | medium | 80 | 0.738 | 0.675 | 0.938 | 0.000 |
| gpt-oss-120b-cloud | short | 81 | 0.407 | 0.407 | 0.877 | 0.104 |
| qwen3-coder-480b-cloud | long | 79 | 0.747 | 0.759 | 0.886 | 0.250 |
| qwen3-coder-480b-cloud | medium | 80 | 0.738 | 0.725 | 0.912 | 0.143 |
| qwen3-coder-480b-cloud | short | 81 | 0.407 | 0.468 | 0.823 | 0.196 |

### Subject model

| judge | split | n | gt_pass_rate | judge_pass_rate | accuracy | fpr_on_failures |
| --- | --- | --- | --- | --- | --- | --- |
| glm-5.1-cloud | devstral-24b | 54 | 0.352 | 0.296 | 0.944 | 0.000 |
| glm-5.1-cloud | gemma4-12b | 56 | 0.929 | 0.911 | 0.982 | 0.000 |
| glm-5.1-cloud | glm-5.1-cloud | 19 | 1.000 | 1.000 | 1.000 | - |
| glm-5.1-cloud | qwen2.5-coder-32b-q4_k_m | 36 | 0.000 | 0.182 | 0.818 | 0.182 |
| glm-5.1-cloud | qwen3-coder-30b | 60 | 0.783 | 0.717 | 0.900 | 0.077 |
| glm-5.1-cloud | qwen3-coder-480b-cloud | 15 | 0.933 | 0.867 | 0.933 | 0.000 |
| gpt-oss-120b-cloud | devstral-24b | 54 | 0.352 | 0.259 | 0.907 | 0.000 |
| gpt-oss-120b-cloud | gemma4-12b | 56 | 0.929 | 0.875 | 0.946 | 0.000 |
| gpt-oss-120b-cloud | glm-5.1-cloud | 19 | 1.000 | 0.947 | 0.947 | - |
| gpt-oss-120b-cloud | qwen2.5-coder-32b-q4_k_m | 36 | 0.000 | 0.139 | 0.861 | 0.139 |
| gpt-oss-120b-cloud | qwen3-coder-30b | 60 | 0.783 | 0.583 | 0.800 | 0.000 |
| gpt-oss-120b-cloud | qwen3-coder-480b-cloud | 15 | 0.933 | 0.733 | 0.800 | 0.000 |
| qwen3-coder-480b-cloud | devstral-24b | 54 | 0.352 | 0.333 | 0.833 | 0.114 |
| qwen3-coder-480b-cloud | gemma4-12b | 56 | 0.929 | 0.893 | 0.929 | 0.250 |
| qwen3-coder-480b-cloud | glm-5.1-cloud | 19 | 1.000 | 1.000 | 1.000 | - |
| qwen3-coder-480b-cloud | qwen2.5-coder-32b-q4_k_m | 36 | 0.000 | 0.265 | 0.735 | 0.265 |
| qwen3-coder-480b-cloud | qwen3-coder-30b | 60 | 0.783 | 0.750 | 0.867 | 0.231 |
| qwen3-coder-480b-cloud | qwen3-coder-480b-cloud | 15 | 0.933 | 0.933 | 1.000 | 0.000 |

### Same-family vs other-family (self-preference)

| judge | judge_family | split | n | gt_pass_rate | judge_pass_rate | accuracy | fpr_on_failures |
| --- | --- | --- | --- | --- | --- | --- | --- |
| glm-5.1-cloud | glm | same_family | 19 | 1.000 | 1.000 | 1.000 | - |
| glm-5.1-cloud | glm | other_family | 221 | 0.597 | 0.592 | 0.922 | 0.081 |
| gpt-oss-120b-cloud | gpt-oss | other_family | 240 | 0.629 | 0.550 | 0.879 | 0.056 |
| qwen3-coder-480b-cloud | qwen | same_family | 111 | 0.550 | 0.624 | 0.844 | 0.250 |
| qwen3-coder-480b-cloud | qwen | other_family | 129 | 0.698 | 0.674 | 0.899 | 0.128 |

### Task category

| judge | split | n | gt_pass_rate | judge_pass_rate | accuracy | fpr_on_failures |
| --- | --- | --- | --- | --- | --- | --- |
| glm-5.1-cloud | code | 62 | 0.532 | 0.597 | 0.871 | 0.207 |
| glm-5.1-cloud | data | 34 | 0.794 | 0.765 | 0.971 | 0.000 |
| glm-5.1-cloud | fs | 20 | 0.550 | 0.550 | 1.000 | 0.000 |
| glm-5.1-cloud | http | 12 | 0.500 | 0.667 | 1.000 | 0.000 |
| glm-5.1-cloud | multi | 59 | 0.678 | 0.678 | 1.000 | 0.000 |
| glm-5.1-cloud | shell | 53 | 0.642 | 0.528 | 0.849 | 0.053 |
| gpt-oss-120b-cloud | code | 62 | 0.532 | 0.565 | 0.871 | 0.172 |
| gpt-oss-120b-cloud | data | 34 | 0.794 | 0.647 | 0.853 | 0.000 |
| gpt-oss-120b-cloud | fs | 20 | 0.550 | 0.550 | 1.000 | 0.000 |
| gpt-oss-120b-cloud | http | 12 | 0.500 | 0.417 | 0.917 | 0.000 |
| gpt-oss-120b-cloud | multi | 59 | 0.678 | 0.559 | 0.881 | 0.000 |
| gpt-oss-120b-cloud | shell | 53 | 0.642 | 0.491 | 0.849 | 0.000 |
| qwen3-coder-480b-cloud | code | 62 | 0.532 | 0.629 | 0.806 | 0.310 |
| qwen3-coder-480b-cloud | data | 34 | 0.794 | 0.676 | 0.882 | 0.000 |
| qwen3-coder-480b-cloud | fs | 20 | 0.550 | 0.650 | 0.800 | 0.333 |
| qwen3-coder-480b-cloud | http | 12 | 0.500 | 0.600 | 1.000 | 0.000 |
| qwen3-coder-480b-cloud | multi | 59 | 0.678 | 0.712 | 0.932 | 0.158 |
| qwen3-coder-480b-cloud | shell | 53 | 0.642 | 0.604 | 0.887 | 0.105 |

## Inter-judge agreement

| judge_a | judge_b | n | raw_agreement | cohens_kappa |
| --- | --- | --- | --- | --- |
| glm-5.1-cloud | gpt-oss-120b-cloud | 237 | 0.924 | 0.844 |
| glm-5.1-cloud | qwen3-coder-480b-cloud | 237 | 0.920 | 0.826 |
| gpt-oss-120b-cloud | qwen3-coder-480b-cloud | 238 | 0.853 | 0.696 |
