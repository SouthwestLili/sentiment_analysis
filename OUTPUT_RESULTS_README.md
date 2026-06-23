# Output Results Explanation

This file explains the output from running:

```bash
python main.py
```

The results below come from the current project implementation.

## 1. Toy Data Results

The first part of the output shows the parameters learned on the small toy
dataset.

```text
theta for Perceptron is 3.9173999999999918, 4.164000000000001
theta_0 for Perceptron is -8.0

theta for Average Perceptron is 3.47826049999999, 3.611060999999974
theta_0 for Average Perceptron is -6.373

theta for Pegasos is 0.7346463119064065, 0.6300224592973831
theta_0 for Pegasos is -1.2195071848898564
```

These values are the learned linear classifier parameters.

For a two-dimensional feature vector `x`, the classifier predicts using:

```text
theta dot x + theta_0
```

If the result is greater than `0`, the prediction is `+1`. Otherwise, the
prediction is `-1`.

### Values Rounded to 4 Decimal Places

For the Perceptron algorithm:

```text
theta = 3.9174, 4.1640
theta_0 = -8.0000
```

For the Average Perceptron algorithm:

```text
theta = 3.4783, 3.6111
theta_0 = -6.3730
```

For the Pegasos algorithm:

```text
theta = 0.7346, 0.6300
theta_0 = -1.2195
```

## 2. Baseline Accuracy

This part reports the training and validation accuracy for each classifier.

```text
Training accuracy for perceptron:   0.7953
Validation accuracy for perceptron: 0.6900

Training accuracy for average perceptron:   0.9030
Validation accuracy for average perceptron: 0.7520

Training accuracy for Pegasos:    0.8592
Validation accuracy for Pegasos:  0.7480
```

### What These Values Mean

Training accuracy means how well the classifier performs on the data it was
trained on.

Validation accuracy means how well the classifier performs on separate data
that was not used for training.

Validation accuracy is usually more important for choosing a model, because it
better estimates how the model may perform on new reviews.

### Baseline Validation Accuracies

For the baseline question, the validation accuracies are:

```text
Perceptron validation accuracy = 0.6900
Average Perceptron validation accuracy = 0.7520
Pegasos validation accuracy = 0.7480
```

In this baseline run, Average Perceptron has the highest validation accuracy.

## 3. Parameter Tuning Results

The next part tries different values of `T` for Perceptron, Average
Perceptron, and Pegasos.

For Pegasos, it also tries different values of `lambda`.

The candidate values are:

```python
Ts = [1, 5, 10, 15, 25, 50]
Ls = [0.001, 0.01, 0.1, 1, 10]
```

## 4. Perceptron Tuning

Output:

```text
perceptron valid: [(1, 0.64), (5, 0.684), (10, 0.69), (15, 0.726), (25, 0.728), (50, 0.754)]
best = 0.7540, T=50.0000
```

This means the best validation accuracy for Perceptron is:

```text
validation accuracy = 0.7540
T = 50
```

## 5. Average Perceptron Tuning

Output:

```text
avg perceptron valid: [(1, 0.732), (5, 0.746), (10, 0.752), (15, 0.748), (25, 0.744), (50, 0.752)]
best = 0.7520, T=10.0000
```

This means the best validation accuracy for Average Perceptron is:

```text
validation accuracy = 0.7520
T = 10
```

There is also a tie at `T = 50`, because it also gives `0.752`.

The code chooses the first best value, so it reports `T = 10`.

## 6. Pegasos Tuning for T

Output:

```text
Pegasos valid: tune T [(1, 0.686), (5, 0.726), (10, 0.748), (15, 0.746), (25, 0.768), (50, 0.752)]
best = 0.7680, T=25.0000
```

This means the best validation accuracy for Pegasos while tuning `T` is:

```text
validation accuracy = 0.7680
T = 25
```

During this step, `lambda` is fixed at:

```text
lambda = 0.01
```

## 7. Pegasos Tuning for Lambda

Output:

```text
Pegasos valid: tune L [(0.001, 0.768), (0.01, 0.768), (0.1, 0.718), (1, 0.52), (10, 0.51)]
best = 0.7680, L=0.0010
```

This means the best validation accuracy for Pegasos while tuning `lambda` is:

```text
validation accuracy = 0.7680
lambda = 0.001
```

There is a tie between:

```text
lambda = 0.001
lambda = 0.01
```

Both give validation accuracy `0.768`.

The code chooses the first best value, so it reports:

```text
lambda = 0.001
```

## 8. Best Tuned Values to Report

For the tuning section, report:

### Perceptron

```text
T = 50
validation accuracy = 0.7540
```

### Average Perceptron

```text
T = 10
validation accuracy = 0.7520
```

### Pegasos

```text
T = 25
lambda = 0.0010
validation accuracy = 0.7680
```

## 9. Stop Words Removed and Count Features

The final accuracy shown in the output is:

```text
Accuracy with stop words removed and counts features: 0.77
```

This result uses:

- Pegasos
- `T = 25`
- `lambda = 0.01`
- stop words removed from the dictionary
- count features instead of binary features

The test accuracy is:

```text
0.7700
```

## 10. Most Explanatory Positive Words

The output also shows the top words with the largest positive weights:

```text
Most Explanatory Word Features
['delicious', 'loves', 'great', 'perfect', 'best', 'favorite', 'wonderful', 'excellent', 'tasty', 'thank']
```

These words are the strongest positive indicators learned by the classifier.

They should be entered as:

```text
Top 1: delicious
Top 2: loves
Top 3: great
Top 4: perfect
Top 5: best
Top 6: favorite
Top 7: wonderful
Top 8: excellent
Top 9: tasty
Top 10: thank
```

## 11. Quick Summary

Important values from this run:

```text
Baseline Perceptron validation accuracy: 0.6900
Baseline Average Perceptron validation accuracy: 0.7520
Baseline Pegasos validation accuracy: 0.7480

Best Perceptron:
T = 50
validation accuracy = 0.7540

Best Average Perceptron:
T = 10
validation accuracy = 0.7520

Best Pegasos:
T = 25
lambda = 0.0010
validation accuracy = 0.7680

Stop words removed + count features test accuracy:
0.7700

Top positive words:
delicious, loves, great, perfect, best, favorite, wonderful, excellent, tasty, thank
```
