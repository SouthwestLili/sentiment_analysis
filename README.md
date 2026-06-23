# Sentiment Analysis with Linear Classifiers

This project builds a sentiment classifier for Amazon food product reviews.
The goal is to predict whether a review is positive or negative based on the
text of the review.

The original reviews were rated on a 5-point scale. In this project, the
labels are simplified into two classes:

- `+1`: positive review
- `-1`: negative review

We implement several linear classification algorithms from scratch and compare
their performance.

## 1. Project Goal

The main goal of this project is to understand how basic machine learning
classifiers work.

In this project, we will:

1. Implement linear classifiers.
2. Convert text reviews into numerical feature vectors.
3. Train classifiers on a training dataset.
4. Evaluate classifiers on validation and test datasets.
5. Tune hyperparameters.
6. Try simple feature engineering methods.

The three main learning algorithms are:

- Perceptron
- Average Perceptron
- Pegasos

## 2. Required Software

This project uses Python.

Recommended version:

```bash
Python 3.11
```

Required Python libraries:

```bash
numpy
matplotlib
```

You can install them with:

```bash
pip install numpy matplotlib
```

If you use Conda, you can create or activate an environment:

```bash
conda create -n 6.86x python=3.11
conda activate 6.86x
pip install numpy matplotlib
```

If you already have a working Conda environment for this course, you can reuse
it.

## 3. Project Files

The project folder contains several important files.

### `project1.py`

This is the main file where most functions are implemented.

Important functions include:

- `hinge_loss_single`
- `hinge_loss_full`
- `perceptron_single_step_update`
- `perceptron`
- `average_perceptron`
- `pegasos_single_step_update`
- `pegasos`
- `classify`
- `classifier_accuracy`
- `bag_of_words`
- `extract_bow_feature_vectors`

### `main.py`

This file is used to run experiments.

It loads the data, trains classifiers, prints accuracy values, and plots
results.

You may need to uncomment different sections as you move through the project.

### `utils.py`

This file contains helper functions provided by the course staff.

Examples include:

- loading datasets
- plotting toy data
- tuning hyperparameters
- finding important words

Usually, you do not need to edit this file.

### `test.py`

This file runs local tests.

You can use it to check whether your functions are implemented correctly.

Run:

```bash
python test.py
```

Passing these tests does not guarantee full correctness, but it is very useful
for debugging.

### Dataset Files

The project also includes data files such as:

- `reviews_train.tsv`
- `reviews_val.tsv`
- `reviews_test.tsv`
- `toy_data.tsv`
- `stopwords.txt`

The `.tsv` files contain text reviews and labels.

## 4. How to Run the Project

First, open a terminal and go to the project folder:

```bash
cd sentiment_analysis
```

Run the tests:

```bash
python test.py
```

Run the main experiment script:

```bash
python main.py
```

If `main.py` gives a `NotImplementedError`, it usually means one function in
`project1.py` has not been completed yet.

## 5. Hinge Loss

Hinge loss is used to measure how well a classifier is doing.

For one data point, the formula is:

```text
loss = max(0, 1 - y * (theta dot x + theta_0))
```

Where:

- `x` is the feature vector
- `y` is the true label, either `+1` or `-1`
- `theta` is the weight vector
- `theta_0` is the bias term

If the classifier is correct and confident, the loss is `0`.

If the classifier is wrong or not confident enough, the loss is positive.

## 6. Perceptron Algorithm

The Perceptron is a simple linear classifier.

It predicts using:

```text
theta dot x + theta_0
```

If the result is greater than zero, the prediction is positive. Otherwise, the
prediction is negative.

The Perceptron updates its parameters when it makes a mistake or when the point
is exactly on the decision boundary.

Update rule:

```text
theta = theta + y * x
theta_0 = theta_0 + y
```

This update only happens when:

```text
y * (theta dot x + theta_0) <= 0
```

## 7. Average Perceptron Algorithm

The Average Perceptron is similar to the normal Perceptron.

The difference is that it averages all parameter values seen during training.

Instead of returning only the final `theta`, it returns:

```text
average_theta = sum of all theta values / number of updates
```

This often works better than normal Perceptron because it makes the model more
stable.

## 8. Pegasos Algorithm

Pegasos is another linear classification algorithm.

It is related to Support Vector Machines and uses hinge loss with
regularization.

Pegasos has two important parameters:

- `eta`: learning rate
- `lambda`: regularization strength

The learning rate changes over time:

```text
eta = 1 / sqrt(t)
```

where `t` is the number of updates so far.

Pegasos update rule:

If:

```text
y * (theta dot x + theta_0) <= 1
```

then:

```text
theta = (1 - eta * lambda) * theta + eta * y * x
theta_0 = theta_0 + eta * y
```

Otherwise:

```text
theta = (1 - eta * lambda) * theta
theta_0 = theta_0
```

Important note: `theta_0` is not regularized.

## 9. Text Features: Bag of Words

Computers cannot directly understand text, so we convert reviews into numbers.

This project uses the Bag of Words representation.

The idea is simple:

1. Build a dictionary of all words in the training reviews.
2. Each word gets a unique index.
3. Each review becomes a vector.
4. Each vector records whether words appear in the review.

Example dictionary:

```python
{
    "great": 0,
    "bad": 1,
    "delicious": 2
}
```

Example review:

```text
great delicious great
```

Binary feature vector:

```text
[1, 0, 1]
```

Count feature vector:

```text
[2, 0, 1]
```

## 10. Classification

After training, the model predicts labels using:

```text
score = theta dot x + theta_0
```

Rule:

```text
if score > 0:
    prediction = +1
else:
    prediction = -1
```

Important detail:

If the score is exactly `0`, the prediction should be `-1`.

## 11. Accuracy

Accuracy measures how many predictions are correct.

Formula:

```text
accuracy = number of correct predictions / total number of predictions
```

For example, if the model gets 80 out of 100 reviews correct:

```text
accuracy = 0.80
```

The project calculates:

- training accuracy
- validation accuracy
- test accuracy

The validation set is used for choosing parameters. The test set is used only
at the end for final evaluation.

## 12. Hyperparameter Tuning

Some algorithms need hyperparameters.

For Perceptron and Average Perceptron, we tune:

```text
T
```

`T` is the number of times the algorithm goes through the dataset.

For Pegasos, we tune:

```text
T
lambda
```

The project tries these values:

```python
Ts = [1, 5, 10, 15, 25, 50]
Ls = [0.001, 0.01, 0.1, 1, 10]
```

The best value is chosen based on validation accuracy.

## 13. Feature Engineering

Feature engineering means changing how the data is represented.

In this project, we try two simple changes.

### Remove Stop Words

Stop words are common words such as:

```text
the, is, to, on
```

These words often do not carry strong sentiment.

The file `stopwords.txt` contains words that can be removed from the
dictionary.

### Count Features

The original Bag of Words representation uses binary features.

Binary feature:

```text
1 if the word appears
0 if the word does not appear
```

Count feature:

```text
number of times the word appears
```

Count features may sometimes help because repeated words can show stronger
sentiment.

## 14. Important Words

After training a linear classifier, each word has a weight in `theta`.

Words with large positive weights are strong signals for positive reviews.

For example, words like these may have large positive weights:

```text
great
delicious
excellent
perfect
```

Words with large negative weights are strong signals for negative reviews.

For example:

```text
bad
awful
terrible
disappointing
```

The function `utils.most_explanatory_word` helps find the most important
positive words.

## 15. Common Errors

### `NotImplementedError`

This means a required function has not been completed yet.

Check `project1.py` and remove the placeholder:

```python
raise NotImplementedError
```

after writing the correct code.

### `NameError`

This means Python is using a variable that has not been defined.

Example:

```text
NameError: name 'train_bow_features' is not defined
```

This usually means the feature matrix was not created before it was used.

### Wrong Classification When Score Is Zero

Remember:

```text
score > 0 means +1
score <= 0 means -1
```

Do not use:

```python
scores >= 0
```

Use:

```python
scores > 0
```

## 16. Suggested Workflow

A good order for completing this project is:

1. Implement hinge loss.
2. Implement Perceptron single update.
3. Implement full Perceptron.
4. Implement Average Perceptron.
5. Implement Pegasos single update.
6. Implement full Pegasos.
7. Test algorithms on toy data.
8. Implement classification.
9. Implement classifier accuracy.
10. Implement Bag of Words.
11. Run baseline accuracy.
12. Tune hyperparameters.
13. Evaluate on the test set.
14. Try feature engineering.
15. Find important words.

After each step, run:

```bash
python test.py
```

This helps catch errors early.

## 17. Final Notes

This project is a good introduction to text classification and linear models.

The most important ideas are:

- text must be converted into numerical features
- linear classifiers use weights to make predictions
- different algorithms update weights in different ways
- validation data is used for tuning
- test data is used for final evaluation
- feature engineering can change performance significantly

Even though the algorithms are simple, they are powerful enough to classify
real product reviews with reasonable accuracy.
