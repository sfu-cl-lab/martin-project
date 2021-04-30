# martin-project
Martin's MSc project repository

## Dataset
The dataset contains play-by-play data from all manually tagged UBC Men's Volleyball games in 2017/2018, 2018/2019 and 2019/2020 seasons. Additionally, a number of games between other teams in Canada West are included as they were used for game planning for UBC.

There are two versions of the dataset: one where locations are described categorically using zones (Z1, Z2, ...) and one where they are represented with X and Y coordinates.

#### Columns
- Season
- GameID
- PlayerTeam: team executing last action
- PlayerName: player executing last action
- RewardValue, RewardDistance: +1 for home point, -1 for away point, RewardDistance contains distance to end of current rally
- SetNumber: current set (1-5)
- ScoreMax: maximum of the home and away scores in current set (eg. for 21-18 it would be 21)
- ScoreDiff: away score minus home score in current set (eg. for 21-18 it would be -3)
- ActionHome, ActionAway: flags for action performed by home or away team
- ActionType: one of Serve, Receive, sEt, Attack, Block, Dig or Free-ball
- ActionStartZone, ActionEndZone: start and end location of action performed
- ActionSpeed: flag denoting a spin serve or a fast set (vs. float serve or high set)
- ActionOutcome: one of the possible action outcomes: = (error), -, /, !, +, # (perfect)

To incorporate action history, up to 10 most recent actions (from beginning of rally) are included in each data point.

## Notebooks:
- DecisionTree: contains experiments with fitting a decision tree classifier to the dataset, predicting whether the point will go to the home team (1) or away team (-1). Prediction probabilities also considered using ratio of training data in leaves. 5-fold cross validation is used to prevent overfitting.
- RandomForest: same as above but using a random forest classifier.
- I have also added several other notebooks such as NeuralNet and MimicTree that experiment with other models.
- Questions: contains various example analyses using values from the reinforcement learning neural net.

## RL Code
- Them nn_code subfolder contains Python code for the reinforcement learning neural network.
