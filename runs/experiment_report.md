# Experimental Evaluation & Statistical Analysis Report

- **Total Executions:** 90 runs (6 Tasks × 5 Constraint Conditions × 3 Repetitions)
- **Total Processing Time:** 11194.66s | **Mean API Latency:** 981.1 ms
- **Total API Tokens:** 32,121,191 (Prompt: 30,950,265, Completion: 1,170,926)
- **Estimated Total Cost:** \$12.8799 USD

---

## 1. Task-Performance & Coordination Metrics by Condition (Mean ± SD [95% CI])

| Condition | Runs | Success Rate | DAG Progress (%) | Makespan (s) | Valid Action Rate | Violation Rate | Conflicts | Recovery Rate | Workload Gini |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reference** | 18 | 27.8% ± 46.1% [4.9, 50.7] | 64.2% ± 31.8% | 42.2 ± 5.8 [39.3, 45.1] | 76.1% ± 4.9% | 23.9% | 3.94 ± 1.89 | 100.0% | 0.106 |
| **Capability-limited** | 18 | 5.6% ± 23.6% [-6.2, 17.3] | 45.0% ± 27.1% | 45.0 ± 0.0 [45.0, 45.0] | 78.9% ± 5.4% | 21.1% | 1.28 ± 1.27 | 100.0% | 0.173 |
| **Observation-limited** | 18 | 5.6% ± 23.6% [-6.2, 17.3] | 62.9% ± 29.4% | 44.9 ± 0.2 [44.8, 45.1] | 74.4% ± 6.0% | 25.6% | 3.83 ± 2.33 | 100.0% | 0.109 |
| **Communication-limited** | 18 | 0.0% ± 0.0% [0.0, 0.0] | 55.7% ± 20.3% | 45.0 ± 0.0 [45.0, 45.0] | 82.2% ± 4.5% | 17.8% | 2.39 ± 1.33 | 100.0% | 0.034 |
| **Combined** | 18 | 0.0% ± 0.0% [0.0, 0.0] | 35.7% ± 19.2% | 45.0 ± 0.0 [45.0, 45.0] | 85.8% ± 3.7% | 14.2% | 1.39 ± 1.20 | 100.0% | 0.051 |

---

## 2. Natural-Language Communication & Independent LLM Judge Metrics (Mean ± SD [95% CI])

| Condition | Messages | Comm Tokens | Grounded Rate | Redundant Rate | Judge Relevance | Judge Clarity | Judge Grounding | Judge Utility | Judge Overall (1-5) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reference** | 34.7 ± 8.9 | 1126.5 ± 292.8 | 99.9% ± 0.4% | 0.7% ± 1.4% | 4.22 ± 0.65 | 4.50 ± 0.51 | 3.56 ± 1.04 | 3.28 ± 0.57 | **3.71** [3.48, 3.95] |
| **Capability-limited** | 39.8 ± 11.1 | 1255.6 ± 383.6 | 99.6% ± 0.9% | 4.8% ± 7.9% | 3.67 ± 0.59 | 4.06 ± 0.24 | 3.17 ± 0.86 | 2.94 ± 0.73 | **3.22** [3.01, 3.44] |
| **Observation-limited** | 44.1 ± 11.3 | 1334.7 ± 336.1 | 100.0% ± 0.0% | 0.9% ± 1.6% | 4.22 ± 0.73 | 4.22 ± 0.43 | 2.72 ± 0.89 | 3.17 ± 0.71 | **3.43** [3.22, 3.64] |
| **Communication-limited** | 10.5 ± 3.9 | 214.1 ± 77.4 | 100.0% ± 0.0% | 0.3% ± 1.5% | 4.89 ± 0.32 | 4.67 ± 0.49 | 3.94 ± 1.16 | 3.11 ± 0.83 | **3.98** [3.69, 4.28] |
| **Combined** | 10.8 ± 3.6 | 232.4 ± 74.7 | 100.0% ± 0.0% | 0.0% ± 0.0% | 4.67 ± 0.49 | 4.61 ± 0.50 | 3.44 ± 1.10 | 3.11 ± 0.83 | **3.81** [3.51, 4.10] |

---

## 3. Paired Statistical Comparisons vs. Reference Condition (Paired t-test, Wilcoxon Signed-Rank & Cohen's $d$)

| Limitation Condition | Metric | Ref Mean | Cond Mean | Diff ($\Delta$) | Paired $t$ ($p$-value) | Wilcoxon $W$ ($p$-value) | Cohen's $d$ (Magnitude) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Capability-limited** | Task Success Rate | 0.278 | 0.056 | -0.222 | t=-2.20 (p=0.0416)* | W=0.0 (p=0.1250) | d=-0.61 (medium) |
| **Capability-limited** | DAG Completion (%) | 64.233 | 45.050 | -19.183 | t=-3.82 (p=0.0014)* | W=12.0 (p=0.0041)* | d=-0.65 (medium) |
| **Capability-limited** | Makespan (s) | 42.167 | 45.000 | +2.833 | t=2.07 (p=0.0542) | W=0.0 (p=0.0625) | d=+0.69 (medium) |
| **Capability-limited** | Valid Action Rate | 0.761 | 0.789 | +0.028 | t=1.91 (p=0.0731) | W=37.0 (p=0.0342)* | d=+0.54 (medium) |
| **Capability-limited** | Resource Conflicts | 3.944 | 1.278 | -2.667 | t=-4.62 (p=0.0002)* | W=10.5 (p=0.0018)* | d=-1.65 (large) |
| **Capability-limited** | Number of Peer Messages | 34.722 | 39.778 | +5.056 | t=2.13 (p=0.0477)* | W=42.5 (p=0.1123) | d=+0.50 (medium) |
| **Capability-limited** | Total Comm Tokens | 1126.500 | 1255.556 | +129.056 | t=1.32 (p=0.2033) | W=72.0 (p=0.5798) | d=+0.38 (small) |
| **Capability-limited** | Grounded Message Rate | 0.999 | 0.996 | -0.003 | t=-1.12 (p=0.2782) | W=1.0 (p=0.2500) | d=-0.39 (small) |
| **Capability-limited** | Redundant Message Rate | 0.007 | 0.048 | +0.041 | t=2.28 (p=0.0359)* | W=8.5 (p=0.0132)* | d=+0.73 (medium) |
| **Capability-limited** | LLM Judge Overall Score (1-5) | 3.714 | 3.223 | -0.491 | t=-3.23 (p=0.0049)* | W=20.5 (p=0.0086)* | d=-1.07 (large) |
| **Observation-limited** | Task Success Rate | 0.278 | 0.056 | -0.222 | t=-1.72 (p=0.1037) | W=3.5 (p=0.2188) | d=-0.61 (medium) |
| **Observation-limited** | DAG Completion (%) | 64.233 | 62.856 | -1.378 | t=-0.26 (p=0.8017) | W=41.5 (p=0.8000) | d=-0.04 (negligible) |
| **Observation-limited** | Makespan (s) | 42.167 | 44.944 | +2.778 | t=2.02 (p=0.0599) | W=1.0 (p=0.0625) | d=+0.68 (medium) |
| **Observation-limited** | Valid Action Rate | 0.761 | 0.744 | -0.018 | t=-1.04 (p=0.3140) | W=70.5 (p=0.5277) | d=-0.32 (small) |
| **Observation-limited** | Resource Conflicts | 3.944 | 3.833 | -0.111 | t=-0.17 (p=0.8634) | W=39.5 (p=0.6885) | d=-0.05 (negligible) |
| **Observation-limited** | Number of Peer Messages | 34.722 | 44.111 | +9.389 | t=2.79 (p=0.0125)* | W=25.0 (p=0.0157)* | d=+0.93 (large) |
| **Observation-limited** | Total Comm Tokens | 1126.500 | 1334.722 | +208.222 | t=1.94 (p=0.0689) | W=47.0 (p=0.0987) | d=+0.66 (medium) |
| **Observation-limited** | Grounded Message Rate | 0.999 | 1.000 | +0.001 | t=1.00 (p=0.3313) | W=0.0 (p=1.0000) | d=+0.33 (small) |
| **Observation-limited** | Redundant Message Rate | 0.007 | 0.009 | +0.002 | t=0.44 (p=0.6671) | W=21.0 (p=0.9102) | d=+0.15 (negligible) |
| **Observation-limited** | LLM Judge Overall Score (1-5) | 3.714 | 3.429 | -0.285 | t=-1.76 (p=0.0958) | W=42.0 (p=0.0610) | d=-0.63 (medium) |
| **Communication-limited** | Task Success Rate | 0.278 | 0.000 | -0.278 | t=-2.56 (p=0.0204)* | W=0.0 (p=0.0625) | d=-0.85 (large) |
| **Communication-limited** | DAG Completion (%) | 64.233 | 55.678 | -8.556 | t=-1.35 (p=0.1939) | W=42.5 (p=0.1957) | d=-0.32 (small) |
| **Communication-limited** | Makespan (s) | 42.167 | 45.000 | +2.833 | t=2.07 (p=0.0542) | W=0.0 (p=0.0625) | d=+0.69 (medium) |
| **Communication-limited** | Valid Action Rate | 0.761 | 0.822 | +0.061 | t=3.51 (p=0.0027)* | W=23.5 (p=0.0074)* | d=+1.30 (large) |
| **Communication-limited** | Resource Conflicts | 3.944 | 2.389 | -1.556 | t=-2.93 (p=0.0094)* | W=16.0 (p=0.0127)* | d=-0.95 (large) |
| **Communication-limited** | Number of Peer Messages | 34.722 | 10.500 | -24.222 | t=-13.78 (p=0.0000)* | W=0.0 (p=0.0002)* | d=-3.55 (large) |
| **Communication-limited** | Total Comm Tokens | 1126.500 | 214.056 | -912.444 | t=-14.14 (p=0.0000)* | W=0.0 (p=0.0000)* | d=-4.26 (large) |
| **Communication-limited** | Grounded Message Rate | 0.999 | 1.000 | +0.001 | t=1.00 (p=0.3313) | W=0.0 (p=1.0000) | d=+0.33 (small) |
| **Communication-limited** | Redundant Message Rate | 0.007 | 0.003 | -0.003 | t=-0.90 (p=0.3793) | W=3.0 (p=0.6250) | d=-0.24 (small) |
| **Communication-limited** | LLM Judge Overall Score (1-5) | 3.714 | 3.983 | +0.269 | t=1.53 (p=0.1456) | W=40.0 (p=0.1546) | d=+0.50 (medium) |
| **Combined** | Task Success Rate | 0.278 | 0.000 | -0.278 | t=-2.56 (p=0.0204)* | W=0.0 (p=0.0625) | d=-0.85 (large) |
| **Combined** | DAG Completion (%) | 64.233 | 35.728 | -28.506 | t=-5.42 (p=0.0000)* | W=1.5 (p=0.0006)* | d=-1.08 (large) |
| **Combined** | Makespan (s) | 42.167 | 45.000 | +2.833 | t=2.07 (p=0.0542) | W=0.0 (p=0.0625) | d=+0.69 (medium) |
| **Combined** | Valid Action Rate | 0.761 | 0.858 | +0.097 | t=8.46 (p=0.0000)* | W=0.0 (p=0.0000)* | d=+2.25 (large) |
| **Combined** | Resource Conflicts | 3.944 | 1.389 | -2.556 | t=-5.57 (p=0.0000)* | W=5.5 (p=0.0007)* | d=-1.61 (large) |
| **Combined** | Number of Peer Messages | 34.722 | 10.833 | -23.889 | t=-11.23 (p=0.0000)* | W=0.0 (p=0.0002)* | d=-3.54 (large) |
| **Combined** | Total Comm Tokens | 1126.500 | 232.444 | -894.056 | t=-13.05 (p=0.0000)* | W=0.0 (p=0.0000)* | d=-4.18 (large) |
| **Combined** | Grounded Message Rate | 0.999 | 1.000 | +0.001 | t=1.00 (p=0.3313) | W=0.0 (p=1.0000) | d=+0.33 (small) |
| **Combined** | Redundant Message Rate | 0.007 | 0.000 | -0.007 | t=-2.08 (p=0.0531) | W=0.0 (p=0.1250) | d=-0.69 (medium) |
| **Combined** | LLM Judge Overall Score (1-5) | 3.714 | 3.809 | +0.096 | t=0.54 (p=0.5961) | W=63.0 (p=0.5382) | d=+0.18 (negligible) |

---

## 4. Failure Categories, Processing Time, API Latency & Cost Breakdown by Condition

| Condition | Precondition / Dep Errors | Resource Conflicts | Skill Mismatches | Comm Budget Blocks | Injected Failures | Processing Time (s) | Mean API Latency (ms) | Total API Tokens | Est. Cost (USD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reference** | 219 | 71 | 0 | 0 | 20 | 2098.73s | 983.1 ms | 6,398,206 | \$2.5662 |
| **Capability-limited** | 169 | 23 | 30 | 0 | 20 | 2282.73s | 968.4 ms | 7,309,272 | \$2.9327 |
| **Observation-limited** | 208 | 69 | 0 | 0 | 20 | 2291.27s | 1005.5 ms | 7,299,115 | \$2.9293 |
| **Communication-limited** | 242 | 43 | 0 | 2 | 20 | 2221.59s | 972.3 ms | 5,485,622 | \$2.1972 |
| **Combined** | 139 | 25 | 42 | 0 | 20 | 2300.34s | 974.0 ms | 5,628,976 | \$2.2545 |

---

## 5. Do Communication Characteristics Predict Task Success or Recovery?

| Communication Characteristic | $r$ (Task Success) | $p$-value | $r$ (DAG Progress) | $r$ (Recovery Rate) | Mean (Successful Runs) | Mean (Unsuccessful Runs) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Grounded Message Rate (Simulator)** | +0.062 | 0.5590 | +0.143 | +0.000 | 1.000 | 0.999 |
| **Redundant Message Rate (Simulator)** | -0.098 | 0.3598 | -0.118 | +0.000 | 0.000 | 0.015 |
| **LLM Judge Relevance (1-5)** | -0.138 | 0.1931 | +0.027 | +0.000 | 4.000 | 4.361 |
| **LLM Judge Clarity (1-5)** | +0.010 | 0.9232 | +0.025 | +0.000 | 4.429 | 4.410 |
| **LLM Judge Grounding (1-5)** | +0.094 | 0.3761 | -0.053 | +0.000 | 3.714 | 3.337 |
| **LLM Judge Coordination Utility (1-5)** | +0.179 | 0.0909 | +0.208 | +0.000 | 3.571 | 3.084 |
| **LLM Judge Non-Redundancy (1-5)** | -0.155 | 0.1451 | -0.051 | +0.000 | 1.714 | 2.470 |
| **LLM Judge Planning Coherence (1-5)** | +0.223 | 0.0349 | +0.085 | +0.000 | 4.000 | 3.277 |
| **LLM Judge Failure Explanation (1-5)** | +0.164 | 0.1231 | +0.230 | +0.000 | 4.857 | 4.410 |
| **LLM Judge Overall Score (1-5)** | +0.064 | 0.5492 | +0.077 | +0.000 | 3.756 | 3.621 |
| **Number of Peer Messages** | +0.177 | 0.0943 | +0.281 | +0.000 | 38.143 | 27.133 |
| **Total Communication Tokens** | +0.176 | 0.0971 | +0.261 | +0.000 | 1175.143 | 803.771 |
