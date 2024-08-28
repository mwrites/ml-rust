# Gaussian Estimation

```bash
❯ cargo run
```

### Anomaly Detection vs Supervised Learning
If you got a very small number of positive examples that you want to find, prefer anomaly detection. If you got a more balanced number of positive vs negative examples, you can go for supervised learning.

If you think there are many 'types' of anomalies; of future anomalies may look nothing like any of the anomalous examples seen so far, prefer anomaly detection.

For supervised learning you need enough positive examples to get a sense of what positive examples are like, future positive examples are also assumed to be similar to ones in the training set (same distribution).

For example:
- Fraud works better for anomaly detection (because there are many ways and even new ways to fraud)
- Spam often works the same or ask u to go to the same kind of websites.

In conclusion:
- In anomaly detection you really want to find outliers, and work well for imbalanced data, and it is future proof for the any kind of anomalies.
- In contrast supervised learning works well for previously seen patterns, it will works if the distribution is similar as the trained one.


### Choosing The Right Features
In supervised learning it's to be more lenient on the choice of columns, but for anomaly detection it's super important to carefully choose the features. We need to find features outside the gaussian distribution, so the first requirement is for the feature to have a gaussian distribution. If the feature is not gaussian, consider applying some transforms such as log + eepislon or exp to make a gaussian distrib. Remember to also apply the transformation during inference.

- Choose features that might take on unusually large or small values

- Combine features



### How To Use In Practice?
```bash
# Estimate the Gaussian parameters
mu, var = estimate_gaussian(train_x)

# Evaluate the probabilites for the training set
p = multivariate_gaussian(train_x, mu, var)

# Evaluate the probabilites for the cross validation set
p_val = multivariate_gaussian(val_x, mu, var)

# Find the best threshold
epsilon, F1 = select_threshold(val_y, p_val)

print('Best epsilon found using cross-validation: %e'% epsilon)
print('Best F1 on Cross Validation Set:  %f'% F1)
print('# Anomalies found: %d'% sum(p < epsilon))
```