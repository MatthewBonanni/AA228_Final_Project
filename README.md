# AA228_Final_Project - F1 Tire Strategy Optmimization
This project contains a set of programs to collect F1 lap data and train policies that learn the optimal pit stop strategy for F1 cars.

# Steps for Policy Training
## 1. Collect your Dataset
Use the file `scrape_data.py` to collect race data from the [FastF1 API.](https://docs.fastf1.dev/) This dataset contains lap times and telemetry for nearly every F1 Grand Prix, including practices, qualifying, and race sessions. We have provided a dataset for the years 2021 and 2022 in the `data/f1_dataset.h5` file. You may want to augment this data with data from more seasons or additional sessions like practices and test days.

* The `years` list specifies which years' race data is collected. Adding more years can help provide more data, but consider F1 rule changes and how that affects the learning.
* The `cols` and `weather_cols` lists at the top of the file list the column names that are output for each lap in the final dataset.
 * We don't recommend changing this as these are the data that are used in the downstream learning tasks. Items added here will need to also be added to downstream tasks. 
* Driver Names, Team Names, and Track Names are converted to integer IDs in the dataset. The key to correlate these IDs to the names are stored in the `data` folder.

## 2. Train the Lap Time Predictor (Optional)
We provide a pretrained model for predicting lap times based on race state in the `outputs` folder that is already integrated into our MDP policy evaluation. However if you are adding data or want to try to improve the Neural Net, use the `estimator/RaceNet.py` to train the neural network. The model is trained using Pytorch and having experience with training models in this framework is very helpful.

## 3. Offline Policy Training
Use the `q_learn.py` file to perform offline training. This file generates race strategy policies for the tracks specified in the `tracks` list. These policies are then stored in the `learning/q_pols` folder. This script also generates transition models for the on-track events. These transition models are not required for policy evaluation, but help provide more realistic results in policy optimization. Some track Q-learning policies and transition functions are provided.

## 4. Online Policy Optimization and Evaluation
Online policy optmization and evaluation are performed in the `race_MDP.py` file. 
### Optmimizer
We have currently implemented a local policy search using Hooke-Jeeves method in the `learning/optimizer.py` file. Additional optmizers can be added here, including gradient based policy search.
### Evaluation
We have provided multiple policy rollout functions, including a single rollout, Monte-Carlo rollout, and a Trajectory rollout that can take in a sequence of track events. These rollout are implemented in `learning/RaceMDP.py`.
### Policies
We have defined two straightforward parameterized policies. 
* The first is `AgeBasedRandomTirePolicy` that is parameterized only by the lap it should change tires on. This is not a good policy, but it is a good baseline.
* The second policy is `AgeSequencePolicy` that optimizes the sequence of tires and what laps to change on for a given track. This parameterization is found to perform well when optimized, but does not take into account track events.
* We have also implemented a class that evaluated the Q-learned policies. These policies can be loaded by specifying the TrackID in the `QLearnPolicy` initialization.
  
Additional policies should be added in the `learning/strategy.py` file and inherit the `Policy` class.



