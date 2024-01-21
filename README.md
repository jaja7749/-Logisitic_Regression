# Logisitic Regression

Logistic regression is a statistical method used for binary classification problems, where the outcome variable is categorical and has two possible classes. It's a type of regression analysis that is well-suited for predicting the probability of an event occurring, based on one or more predictor variables.

Here are the key components and concepts associated with logistic regression:

1. Binary Outcome:
   - Logistic regression is used when the dependent variable (or outcome) is binary, meaning it has two possible outcomes, often labeled as 0 and 1, representing the two classes.

2. Log-odds Transformation:
   - Unlike linear regression, where the output is continuous, logistic regression models the log-odds of the probability of the event occurring. The logistic function (sigmoid function) is employed to transform the linear combination of predictor variables into a range between 0 and 1.

3. Sigmoid Function:
   - The logistic (or sigmoid) function is defined as
    
     $`\sigma (x) = \frac{1}{1+e^{-x}}`$

     where z is the linear combination of predictor variables and their respective coefficients. This function maps any real-valued number to the range [0, 1].

4. Linear Combination of Predictors:
   - The logistic regression model calculates a linear combination of the predictor variables, each multiplied by its corresponding coefficient. The model can be expressed as

     $`P(x) = \sigma (\omega _i^{T}x_i)`$

     where $`\omega`$ is the coefficients for the predictor variables

5. Maximum Likelihood Estimation (MLE):
   - The logistic regression model parameters (coefficients) are estimated using maximum likelihood estimation. The goal is to maximize the likelihood of observing the given set of outcomes given the predictor variables.

6. Decision Boundary:
   - The decision boundary is a threshold value (typically 0.5) that determines the classification. If the predicted probability is above the threshold, the observation is assigned to one class; otherwise, it is assigned to the other class.
  
     $`sign(x)=\left\{\begin{matrix} 1\ \ \mathrm{if}\ x\geq 0.5 \\ 0\ \ otherwise \end{matrix}\right.`$

7. Loss function:
   - The loss function on logistic regression deals with binary classification problems, where the output is a probability score between 0 and 1. For logistic regression, the more appropriate loss function is the logistic loss (or cross-entropy loss). However, we still can use MSE loss to calculate the loss.

     $`loss = \frac{1}{2}(y-\hat{y})^{2}`$

8. Update weight (Stochastic Gradient Descent):
   - Stochastic Gradient Descent (SGD) is an optimization algorithm commonly used to train machine learning models, including logistic regression. The idea behind SGD is to update the model parameters (weights and biases) iteratively based on the gradient of the loss function with respect to the parameters. The updates are performed for each training sample, making it a stochastic process.

     $`\omega _i^{(n+1)}=\omega _i^{(n)}-\eta (\frac{\partial L}{\partial \omega })`$
     
     $`\frac{\partial L}{\partial \omega }=(y-\sigma (\omega ^{T}x))\cdot (- {\sigma}'(\omega ^{T}x))\cdot x`$

     where $`\eta`$ is learning rate

Logistic regression is widely used in various fields, including medicine, finance, and machine learning, for tasks such as predicting disease occurrence, credit default, and spam detection. It is important to note that logistic regression assumes a linear relationship between the log-odds and the predictor variables, and it works well when the classes are approximately linearly separable in the feature space.

# Logisitic Regression on E-mail Classification
- The email data file is on: https://github.com/justmarkham/DAT5/blob/master/data/SMSSpamCollection.txt

|   spam   |    ham    |
|----------|-----------|
|     1    |     0     |

|  Index  |   Label   |                      Content                       |
|---------|-----------|----------------------------------------------------|
|    0    |     0     |   Go until jurong point, crazy.. Available only ...|
|    1    |     0     |                       Ok lar... Joking wif u oni...|
|    2    |     1     |   Free entry in 2 a wkly comp to win FA Cup fina...|
|    3    |     0     |   U dun say so early hor... U c already then say...|
|    4    |     0     |   Nah I don't think he goes to usf, he lives aro...|

First of all, we create a class to input $`X`$ and $`y(label)`$:
```ruby
class LR:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.w = np.random.randn(1, X.shape[-1])
```

Train the data, we need to build the SGD function to model:
```ruby
    def SGD(self, epoch, batch, learning_rate=0.1):
        loss = np.array([])
        loop = tqdm(range(epoch))
        for i in enumerate(loop):
            g = np.zeros(self.w.shape)
            for j in np.random.choice(len(self.X), batch, replace=False):
                g += (y[j]-self.sigmoid(np.dot(self.w, np.atleast_2d(self.X[j]).T))) * (-self.sigmoid_backward(np.dot(self.w, np.atleast_2d(self.X[j]).T))) * self.X[j]
            g = g/batch
            self.w -= learning_rate * g
            loss = np.append(loss, np.average(self.loss_function(y, self.sigmoid(np.dot(self.w, self.X.T)))))
            loop.set_postfix(train_loss=str(loss[-1]))
        self.loss = loss
```

After we update the weight, the model can detect the e-mail is spam or ham:
```ruby
    def pred(self, X):
        return self.sign(self.sigmoid(np.dot(self.w, X.T))[-1])
```

Now, let's try to vaild the dataset:
```ruby
np.random.seed(1)
model = LR(X, y)
epoch, batch, learning_rate = 10000, 100, 10
model.SGD(epoch, batch, learning_rate)

# 100%|██████████| 10000/10000 [15:07<00:00, 11.02it/s, train_loss=0.0048489542336623]

y_pred = model.pred(X)
acc = accuracy_score(y, y_pred)
print(f"Accuracy: {(np.sum(y==y_pred)/len(y))*100:.3f} %")
cm = confusion_matrix(y, y_pred, labels=[0, 1])
cm_display = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
cm_display.plot()
plt.show()

# Accuracy: 99.049 %
```
![image](https://github.com/jaja7749/-Logisitic_Regression/blob/main/images/LR%20cm.png)

Also we can check the loss to epoch chart:
```ruby
plt.plot(range(epoch), model.loss)

plt.grid()
plt.xlabel("iterations(epoch)")
plt.ylabel("loss")
 
plt.show()
```
![image](https://github.com/jaja7749/-Logisitic_Regression/blob/main/images/LR%20loss.png)

And test the new data:
```ruby
# test1, test2 are [spam, ham]
test1 = "You are a winner U have been specially selected 2 receive £1000 cash or a 4* holiday (flights inc) speak to a live operator 2 claim 0871277810810"
test2 = "I just reached home. I go bathe first. But my sis using net tell u when she finishes k..."
test = np.array([test1, test2])
X_ = dataset.datatransform(data=test, new_data=True)
print("Naive Bayes Predict(1:spam, -1:ham):", model.pred(X_))

# Naive Bayes Predict(1:spam, 0:ham): [1. 0.]
```
