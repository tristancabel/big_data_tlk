
# First classifier 

Our problem is taking  a matrix representing a letter in input and finding the corrsponding letter in output.
First, we take our input X, and using weights W and regularization b, we get scores Y=WX+b = [2.0 , 1.0, 0.1] 

Note that we need to work a bit on our input, in order to correctly train our weights, we will make our inputs **0 mean** and **equal variance**. Then we sill initialize our weights randomly with a relatively small variance.

### softmax
We know that an image can either be an a, a b, or a c but not 2 at the same time. So we want to maximize the probability of correct class and minimize the probability of other classes, for this we will use a softmax on Y. $$S(y_i)= \frac{e^{y_i}}{\sum_j e^{y_j}}$$


### One-Hot encoding
let's use a classifier to find if a letter is either {a,b,c} . 
To represent results, we can use **one-hot encoding**, representing letters by a vector of size 3 such that a= [1,0,0] ; b=[0,1,0] and c=[0,0,1] . In this way, if the vector that comes out of the classifiers is [0.7, 0.2, 0.1] we can easily measure how well we are doing using cross entropy.

It works very well for lots of problem, but for a large number of different outputs, our output vectors become too big. We can use embeddings to solve this.


### Cross entropy 

The cross entropy is $$D(S,L) = - \sum_i{L_i \log(S_i)}$$

 D being distance
 S our solution [0.7, 0.2, 0.1]
 L the correct label [1.0, 0.0, 0.0]
 
 
so X -> linear model ( Y=WX+b ) -> softmax (S(Y)) -> cross entropy (D(S,L)) it's called **multinomial logistic classification**

### Measuring performances
We don't want to have overfitting so we split our input set into:

 - **training set**  train with this
 - **validation set** use output of validation to tune your model
 - **testing set** use this set only to measure real performances

first assignment notebook **notMNIST**


### Stochastic gradient descent  (SGD)
Training logistic regression with gradient descent is great as you are optimizing the error. But, it has problems, the biggest one being it's hard to scale. Indeed, 
the loss function being $$L=\sum D_i$$ , so working on all data it leads to computing a derivative multipled by the learning rate $$\alpha \delta L(w_1, w_2)$$ . And since it's iterative, we have to do this calculation for each step so it's really compute intensive. Si instead of computing the loss, we are going to compute an estimate of it (a bad estimate).
Instead of computing the loss on all the data, we are going to use a tiny random sample of the data (1 to 1000 samples).


### Momentum
At each step, we are taking a small step forward to our solution. Momentum idea is to use knowledge of previous steps direction and speed in order to go to our solution faster.

We will define a running average $$ M = 0.9 M + \delta L$$ then our direction will be $$- \alpha M(w_1, w_2)$$

Next, we will also use a **learning rate decay** (ex exponential decay, or only when there is a plateau, .. )

### Overview

We have seen that quite a few parameters needs to be tuned: 

 - initial learning rate
 - learning rate decay
 - momentum
 - batch size
 - weight initialization
 
 But when things are not working, the first thing to do if often to lower the learning rate! We can also look at **ADAGRAD** which is a tuning of SGD doing momentum and learning rate decay automatically.
