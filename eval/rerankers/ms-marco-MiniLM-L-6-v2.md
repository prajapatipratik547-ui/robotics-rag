# Evaluation results

Run on 2026-10-06 with `python eval/run_eval.py`: 85 questions, 906 chunks from 32 documents, top 5 chunks per question, reranker `Xenova/ms-marco-MiniLM-L-6-v2`.

## Summary

| Mode | Hit@5 | MRR | Context recall | Context precision |
|---|---|---|---|---|
| dense | 84% | 0.695 | 0.775 | 0.209 |
| hybrid | 95% | 0.804 | 0.875 | 0.238 |
| hybrid+rerank | 95% | 0.845 | 0.892 | 0.245 |

Mean retrieval latency per question (CPU): dense 0.01 s, hybrid 0.02 s, hybrid+rerank 1.11 s.

Retrieval metrics only (`--retrieval-only`); `python eval/run_eval.py` adds the answer metrics.

## Is the difference real? (95% bootstrap confidence intervals)

Resampling the 85 questions 10,000 times. Differences are paired: both modes are scored on the same resampled questions. *Clear* means the interval for Hit@5 or MRR excludes 0.

| Mode | Hit@5 | 95% CI | MRR | 95% CI |
|---|---|---|---|---|
| dense | 84% | 75% to 91% | 0.695 | 0.610 to 0.777 |
| hybrid | 95% | 91% to 99% | 0.804 | 0.734 to 0.869 |
| hybrid+rerank | 95% | 91% to 99% | 0.845 | 0.781 to 0.904 |

| Difference | Hit@5 (pts) | 95% CI | MRR | 95% CI | Clear? |
|---|---|---|---|---|---|
| hybrid vs dense | +12 | +6 to +19 | +0.108 | +0.050 to +0.174 | yes |
| hybrid+rerank vs hybrid | +0 | -4 to +4 | +0.041 | -0.010 to +0.095 | no |
| hybrid+rerank vs dense | +12 | +6 to +19 | +0.150 | +0.076 to +0.228 | yes |

## Keyword questions (44)

| Mode | Hit@5 | MRR |
|---|---|---|
| dense | 95% | 0.848 |
| hybrid | 100% | 0.904 |
| hybrid+rerank | 100% | 0.902 |

## Paraphrase questions (41)

| Mode | Hit@5 | MRR |
|---|---|---|
| dense | 71% | 0.531 |
| hybrid | 90% | 0.696 |
| hybrid+rerank | 90% | 0.785 |

## Per question

Rank = position of the first relevant chunk in the top 5 (miss = none retrieved).

| ID | Style | Question | Rank: dense | Rank: hybrid | Rank: hybrid+rerank |
|---|---|---|---|---|---|
| q01 | keyword | What is the distance range of the RPLiDAR A1M8? | 1 | 1 | 1 |
| q02 | keyword | What is the I2C address of the MPU-6050 when the AD0 pin is high? | 1 | 1 | 1 |
| q03 | keyword | What full-scale ranges can the gyroscope on the MPU-6050 be set to? | 1 | 1 | 1 |
| q04 | keyword | Which MPU-6050 register sets the power mode and clock source, and what is its address? | 1 | 1 | 2 |
| q05 | keyword | What is the default value of the MPU-6050 WHO_AM_I register? | 1 | 1 | 1 |
| q06 | keyword | What is the default baud rate of the NEO-6M GPS module? | 5 | 1 | 1 |
| q07 | keyword | How long does the NEO-6M take to get its first fix from a cold start? | 1 | 1 | 2 |
| q08 | keyword | What is the maximum supply voltage of the L298 motor driver? | 1 | 1 | 1 |
| q09 | paraphrase | What distances can the ultrasonic ranging module measure, and how accurate is it? | 1 | 1 | 1 |
| q10 | paraphrase | How long must the trigger pulse be to start an ultrasonic distance measurement? | 1 | 1 | 1 |
| q11 | keyword | What is the stall torque of the MG996R servo? | 1 | 1 | 1 |
| q12 | keyword | How much flash memory does the Arduino Mega 2560 have? | 1 | 1 | 1 |
| q13 | keyword | How many digital I/O pins and analog inputs does the Arduino Mega 2560 have? | miss | 1 | 1 |
| q14 | paraphrase | What power supply does the Raspberry Pi 4 need, and which CPU does it run? | 2 | 3 | 3 |
| q15 | keyword | What is the default value of inflation_radius in the Nav2 inflation layer? | 1 | 1 | 1 |
| q16 | paraphrase | Which setting controls how fast the cost drops off as you move away from an obstacle, and what is its default? | miss | miss | miss |
| q17 | keyword | At what frequency does the Nav2 controller server run by default? | 1 | 1 | 1 |
| q18 | keyword | Which planner plugin does the Nav2 planner server load if planner_plugins is not set? | 1 | 1 | 1 |
| q19 | keyword | What behaviors does the Nav2 behavior server load by default? | 1 | 1 | 1 |
| q20 | paraphrase | How does Nav2 work out where the robot is on a known, static map? | 2 | 3 | 1 |
| q21 | keyword | What are the default min_particles and max_particles in AMCL? | 1 | 1 | 1 |
| q22 | paraphrase | Which part of the system provides the map to odom transform, and which provides odom to base_link? | 1 | 1 | 1 |
| q23 | keyword | What motion models does the MPPI controller support? | 1 | 1 | 1 |
| q24 | keyword | What is the default motion model of the Smac Hybrid-A* planner? | miss | 2 | 1 |
| q25 | keyword | When running Nav2 with SLAM, which Nav2 servers should not be launched? | 1 | 1 | 1 |
| q26 | keyword | What is the default desired_linear_vel of the Regulated Pure Pursuit controller? | 1 | 1 | 1 |
| q27 | keyword | Which behavior tree library does Nav2 use? | 2 | 2 | 2 |
| q28 | paraphrase | How can I stop a robot from ever entering certain areas of the map, or limit its speed there, in Nav2? | miss | 2 | 1 |
| q29 | paraphrase | What happens in Nav2 if one of its servers crashes after it has been brought up? | miss | 2 | 1 |
| q30 | keyword | Which method runs when a ROS 2 lifecycle node goes through its configuration stage, and what should it set up? | 1 | 1 | 1 |
| q31 | paraphrase | What kind of message does a client send to Nav2's top-level navigator to make the robot drive to a goal? | miss | 2 | 1 |
| q32 | paraphrase | Why does the behavior server subscribe to the local costmap instead of keeping its own copy? | 1 | 1 | 1 |
| q33 | paraphrase | How does Nav2's route server differ from a normal freespace planner? | 1 | 1 | 1 |
| q34 | keyword | Which action server in nav2_waypoint_follower accepts goals given in GPS coordinates? | 1 | 1 | 1 |
| q35 | paraphrase | My robot isn't round. Should I give Nav2 a circle radius or the robot's actual shape? | 1 | 1 | 1 |
| q36 | keyword | How much can cache_obstacle_heuristic speed up the Smac planners? | 1 | 1 | 1 |
| q37 | paraphrase | Which global planner do the Nav2 maintainers suggest for a car-like robot with Ackermann steering? | 1 | 1 | 1 |
| q38 | keyword | What does the use_composition launch option do, and what is its default? | 1 | 1 | 1 |
| q39 | paraphrase | How can a robot that drives equally well forwards and backwards reach a goal without turning around 180 degrees? | 1 | 1 | 1 |
| q40 | paraphrase | Which Nav2 costmap layer is meant for a 3D lidar? | 2 | 1 | 2 |
| q41 | keyword | What is the default bt_loop_duration of the BT Navigator? | 1 | 1 | 1 |
| q42 | paraphrase | To watch the navigate-to-pose behavior tree live in Groot2, which port do I connect to by default? | 5 | 1 | 1 |
| q43 | keyword | What is the default action_server_result_timeout in the BT Navigator? | 3 | 1 | 1 |
| q44 | paraphrase | By default, how big is each cell of a Nav2 costmap? | 4 | 3 | 1 |
| q45 | keyword | Which plugins does a Nav2 costmap load when the plugins parameter is not set? | 1 | 1 | 1 |
| q46 | paraphrase | How do I make a costmap move along with the robot so it stays centred on it? | 1 | 2 | 1 |
| q47 | keyword | What does track_unknown_space do in Costmap 2D, and what is its default? | 1 | 1 | 1 |
| q48 | paraphrase | What cost does a cell need in the occupancy grid map to count as a lethal obstacle by default? | 1 | 1 | 1 |
| q49 | keyword | What is the default obstacle_max_range of an obstacle layer data source? | 1 | 1 | 1 |
| q50 | paraphrase | Up to what distance does the obstacle layer clear space by ray-tracing, by default? | 2 | 4 | 1 |
| q51 | paraphrase | How can I keep the static map as the dominant source of information, so the robot never drives through places that are not on it? | miss | 1 | 1 |
| q52 | keyword | What is the default tolerance of the NavFn planner? | 1 | 1 | 1 |
| q53 | paraphrase | By default, does the NavFn global planner search with A* or with Dijkstra's algorithm? | 2 | 2 | 1 |
| q54 | paraphrase | Why are an IMU and wheel encoders often used together for odometry? | 2 | 1 | 1 |
| q55 | keyword | What must a robot's odometry system publish for Nav2 to work? | 5 | 5 | 3 |
| q56 | paraphrase | How do you compute a differential-drive robot's linear and angular velocity from its two wheel speeds? | miss | 2 | miss |
| q57 | keyword | Which ros2_control controller does the Nav2 guide recommend for wheel-encoder odometry? | 1 | 1 | 1 |
| q58 | paraphrase | Which ROS message type would a sonar or infrared distance sensor publish? | 1 | 1 | 1 |
| q59 | keyword | In the Nav2 sensor setup tutorial, what is the maximum range of sam_bot's simulated lidar? | 1 | 1 | 1 |
| q60 | paraphrase | How accurate is a standalone GPS receiver, and what can bring that down to about a centimetre? | miss | 4 | 2 |
| q61 | keyword | Which robot_localization node converts GPS fixes into cartesian coordinates for Nav2? | 4 | 4 | 3 |
| q62 | paraphrase | Why are the EKFs in the Nav2 GPS tutorial run in 2D mode? | 2 | 2 | 2 |
| q63 | paraphrase | In the GPS setup, where does the map origin come from if no datum is set? | 1 | 1 | 1 |
| q64 | paraphrase | How far ahead in time does the MPPI controller predict by default? | 1 | 1 | 1 |
| q65 | keyword | What is the default batch_size of the MPPI controller? | 1 | 1 | 1 |
| q66 | paraphrase | How far must the robot drive before AMCL updates its particle filter? | 2 | 3 | 1 |
| q67 | paraphrase | What is the tightest turn the Hybrid-A* planner assumes the vehicle can make by default? | 1 | 1 | 1 |
| q68 | keyword | What does failure_tolerance mean in the Nav2 controller server, and what is its default? | 1 | 1 | 1 |
| q69 | paraphrase | How often does the Nav2 behavior server run its behavior plugins by default? | 1 | 1 | 1 |
| q70 | paraphrase | What supply voltage range does the InvenSense 6-axis motion sensor run from? | 1 | 1 | 3 |
| q71 | paraphrase | How much sensor data can the motion sensor buffer on-chip so the host can read it in bursts? | miss | 1 | 1 |
| q72 | keyword | What accelerometer full-scale ranges does the MPU-6050 support? | 2 | 2 | 2 |
| q73 | paraphrase | How is the IMU's sample rate derived from its divider register? | 3 | 1 | 1 |
| q74 | keyword | What is the address of the MPU-6050 ACCEL_CONFIG register? | 1 | 1 | 1 |
| q75 | paraphrase | How often can the u-blox GPS module compute a new position, and how accurate is it horizontally? | miss | miss | miss |
| q76 | keyword | What supply voltage does the NEO-6M need? | 3 | 3 | 2 |
| q77 | keyword | Which protocols does the NEO-6M support? | 1 | 2 | 2 |
| q78 | paraphrase | What voltage can I feed into the Mega board's DC barrel jack? | 2 | 1 | 1 |
| q79 | keyword | How much SRAM and EEPROM does the ATmega2560 on the Arduino Mega 2560 have? | 1 | 1 | 1 |
| q80 | paraphrase | How much continuous current can each channel of the dual full-bridge motor driver deliver? | miss | miss | 4 |
| q81 | paraphrase | How does the L298 let you control the motor current? | miss | miss | miss |
| q82 | paraphrase | How many distance measurements per second does the Slamtec 360° laser scanner take, and how fast does it spin? | miss | 5 | 4 |
| q83 | keyword | What laser wavelength does the RPLIDAR A1 use? | 1 | 1 | 1 |
| q84 | paraphrase | How much current can the Pi 4's USB ports supply in total? | 2 | 1 | 2 |
| q85 | keyword | What wireless connectivity does the Raspberry Pi 4 Model B have? | 1 | 1 | 1 |

## Method

- **Test set** (`eval/testset.jsonl`): questions written from the corpus, each with a reference answer and the exact phrase(s) from the source document that answer it. *Keyword* questions use the part code or parameter name; *paraphrase* questions describe it in other words.
- **Relevant chunks:** every chunk of the source document whose section title + text contains an evidence phrase.
- **Hit@5:** share of questions with at least one relevant chunk in the top 5. **MRR:** mean of 1 / rank of the first relevant chunk (0 when missed).
- **Context recall / precision:** ragas `IDBasedContextRecall` (share of relevant chunks retrieved) and `IDBasedContextPrecision` (share of retrieved chunks that are relevant). Precision is capped well below 1, because most questions have only one or two relevant chunks among the five retrieved.
