# Evaluation results

Run on 2026-10-05 with `python eval/run_eval.py`: 26 questions, 906 chunks from 32 documents, top 5 chunks per question.

## Summary

| Mode | Hit@5 | MRR | Context recall | Context precision |
|---|---|---|---|---|
| dense | 88% | 0.815 | 0.721 | 0.262 |
| hybrid | 96% | 0.891 | 0.801 | 0.292 |
| hybrid+rerank | 96% | 0.905 | 0.798 | 0.285 |

Mean retrieval latency per question (CPU): dense 0.01 s, hybrid 0.03 s, hybrid+rerank 4.95 s.

Retrieval metrics only (`--retrieval-only`); `python eval/run_eval.py` adds the answer metrics.

## Keyword questions (20)

| Mode | Hit@5 | MRR |
|---|---|---|
| dense | 90% | 0.860 |
| hybrid | 100% | 0.975 |
| hybrid+rerank | 100% | 0.967 |

## Paraphrase questions (6)

| Mode | Hit@5 | MRR |
|---|---|---|
| dense | 83% | 0.667 |
| hybrid | 83% | 0.611 |
| hybrid+rerank | 83% | 0.700 |

## Per question

Rank = position of the first relevant chunk in the top 5 (miss = none retrieved).

| ID | Style | Question | Rank: dense | Rank: hybrid | Rank: hybrid+rerank |
|---|---|---|---|---|---|
| q01 | keyword | What is the distance range of the RPLiDAR A1M8? | 1 | 1 | 1 |
| q02 | keyword | What is the I2C address of the MPU-6050 when the AD0 pin is high? | 1 | 1 | 1 |
| q03 | keyword | What full-scale ranges can the gyroscope on the MPU-6050 be set to? | 1 | 1 | 1 |
| q04 | keyword | Which MPU-6050 register sets the power mode and clock source, and what is its address? | 1 | 1 | 1 |
| q05 | keyword | What is the default value of the MPU-6050 WHO_AM_I register? | 1 | 1 | 1 |
| q06 | keyword | What is the default baud rate of the NEO-6M GPS module? | 5 | 1 | 1 |
| q07 | keyword | How long does the NEO-6M take to get its first fix from a cold start? | 1 | 1 | 3 |
| q08 | keyword | What is the maximum supply voltage of the L298 motor driver? | 1 | 1 | 1 |
| q09 | paraphrase | What distances can the ultrasonic ranging module measure, and how accurate is it? | 1 | 1 | 1 |
| q10 | paraphrase | How long must the trigger pulse be to start an ultrasonic distance measurement? | 1 | 1 | 1 |
| q11 | keyword | What is the stall torque of the MG996R servo? | 1 | 1 | 1 |
| q12 | keyword | How much flash memory does the Arduino Mega 2560 have? | 1 | 1 | 1 |
| q13 | keyword | How many digital I/O pins and analog inputs does the Arduino Mega 2560 have? | miss | 1 | 1 |
| q14 | paraphrase | What power supply does the Raspberry Pi 4 need, and which CPU does it run? | 2 | 3 | 1 |
| q15 | keyword | What is the default value of inflation_radius in the Nav2 inflation layer? | 1 | 1 | 1 |
| q16 | paraphrase | Which setting controls how fast the cost drops off as you move away from an obstacle, and what is its default? | miss | miss | miss |
| q17 | keyword | At what frequency does the Nav2 controller server run by default? | 1 | 1 | 1 |
| q18 | keyword | Which planner plugin does the Nav2 planner server load if planner_plugins is not set? | 1 | 1 | 1 |
| q19 | keyword | What behaviors does the Nav2 behavior server load by default? | 1 | 1 | 1 |
| q20 | paraphrase | How does Nav2 work out where the robot is on a known, static map? | 2 | 3 | 5 |
| q21 | keyword | What are the default min_particles and max_particles in AMCL? | 1 | 1 | 1 |
| q22 | paraphrase | Which part of the system provides the map to odom transform, and which provides odom to base_link? | 1 | 1 | 1 |
| q23 | keyword | What motion models does the MPPI controller support? | 1 | 1 | 1 |
| q24 | keyword | What is the default motion model of the Smac Hybrid-A* planner? | miss | 2 | 1 |
| q25 | keyword | When running Nav2 with SLAM, which Nav2 servers should not be launched? | 1 | 1 | 1 |
| q26 | keyword | What is the default desired_linear_vel of the Regulated Pure Pursuit controller? | 1 | 1 | 1 |

## Method

- **Test set** (`eval/testset.jsonl`): questions written from the corpus, each with a reference answer and the exact phrase(s) from the source document that answer it. *Keyword* questions use the part code or parameter name; *paraphrase* questions describe it in other words.
- **Relevant chunks:** every chunk of the source document whose section title + text contains an evidence phrase.
- **Hit@5:** share of questions with at least one relevant chunk in the top 5. **MRR:** mean of 1 / rank of the first relevant chunk (0 when missed).
- **Context recall / precision:** ragas `IDBasedContextRecall` (share of relevant chunks retrieved) and `IDBasedContextPrecision` (share of retrieved chunks that are relevant). Precision is capped well below 1, because most questions have only one or two relevant chunks among the five retrieved.
