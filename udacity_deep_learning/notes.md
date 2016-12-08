
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

### Linear model complexity
In general, if we have N inputs features and K outputs, linear models will need (N+1)K parameters.


# Deep Neural networks
We want to introduce some non linearity, for this we need non-linear functions. The first of it being **ReLU** .

### Rectified Linar Unit (ReLU)
Its the simplest non linear function, it's $$y=max(0,x)$$

### first network
With this function, we can build our first network! Let's start from our previous model 
```
X -> linear model ( Y=WX+b ) -> softmax (S(Y)) 
```
Our first neural network (2-layers) will be 
```
X -> linear model ( Y=WX+b ) -> ReLU(Y_2=max(0,Y)) -> -> linear model ( Y_3=W_2*Y_2+b_2 ) -> softmax (S(Y_3))
```
The first layer is `Y=WX+b -> ReLU(Y)`
We will need to choose how many ReLU units we want and that's it!

### back propagation
to train our network, we will use back-propagation. Some functions like ReLU don't need to be trained (blank in the graph), for other we will use chain rules: *the derivative of f(g(x)) is g'(f(x)) * f'(x)*

![back_propagation](back_propagation.png)

second assignment notebook **2_fullyconnected**

Then, it's quite straightforward to have a deeper network adding more layers. Usually, we will get better performances with a deeper network than with a larger one.

### regularization

We can do **early termination**, meaning stopping the training when validation set performances start to decrease. It will prevent too much overfitting. Another thing to do is **regularization**: applying artificial constraints on your network that implicitely reduce the number of free parameters while not making it more difficult to optimize. 

#### L2 regularization
L2 regularization is $$L_2 = L + \beta \frac{1}{2} \norm{W}_2^2$$ The idea is to add a term to the loss that minimize large weights. Note that the derivative of the second term is just $$W$$ !

#### dropout
Another recent technique for regularization that works well is **dropout**. It consist of randomly deactivate some activation (boxes) for each training example (50% or less). It ensure that your network cannot rely on any specific activation because they might get squashed at any given moment. So it forces the network to be somewhat redundant.

With dropout, we want to take the consensus over these redundant models aka averaging the output of the different models. $$Y_e ~ E(Y_t)$$ with *e* standing for evaluation and *t* for training

# Convolutional Neural Network (CNN)

## statistical invariance
Sometimes some features we get are not usefull for the learning we want to do! For example, if we want to guess which letter from the alphabet an image is, the color of this letter is not usefull. In the same way, if want to know if a picture represents a cat, the position of the cat in the image (left upper corner or center or .. ) is not usefull! In this later case, we want to tell our network to apply some **translation invariance**.

The way to achieve this in a network is using **weight sharing**: when we know that two inputs can contain the same kind of information, then we share the weights and train the weights jointly ofr these inputs.

## convnets
Convnets of convolutional networks are neural networks that share their parameters across space.
For an image, it's like sliding the same network on small sets of the image (sliding a kernel over patches of the image with a stride of few pixels). Instead of a stack of matrix multiply layers, we'are going to have stacks of convolutions with the general idea of them forming a pyramid. For example taking an RGB image of 256*256 pixels, we could apply the following network:
`Input(256x256x3) -> h1(128x128x16) -> h2(64x64x64) -> h3(32x32x64) -> classifier`

We increase the depth at each steps considering it roughly represents the semantic complexity of our representation.

## advanced convnets
there is a lot of things we can do to improve our convnets. We will talk about **polling**, and **inception**.

### polling 
It's a better way to reduce the feature maps in the convolutional pyramid. Until now, we have used striding to shift the filters by a few pixel each time and reduce the feature map size.
What if we still do some striding with a low number (like 1) but then took all the convolutions in a neighborhood and combined them somehow! It's **polling**. We can do:

 - *max polling* Y = max(X_i) . It's parameter free, often more accurate but it's more expensive and it's need more hyper parameters (pooling size, pooling stride).
 - *average pooling* it's ike taking a blurred biew.
 - *1x1 convolution* adding a 1x1 convolution allows us to have a mini neural network running over the patch instead of a linear classifier: ` Input(KxKx3) -> h_1(ixixL) -> h_2(1x1xM)` . they are relatively cheap 

### inception
The idea is at each layer of your convnet, you can make a choice: have a pooing operation, have a convoution. Then you need to decide is it a 1x1, 3x3, a 5x5. But all of them can be beneficial so why choose, let's use them all and concatenate the result! 

assignment 4

# Deep models for text analysis
Text analysis is a good problem to tackle because it's complex.

A first example of complexity is some words are more important than others: if you find "name" in a text, it doesn't give too much insight of what the text is about. However, if we find "retinopathy", we can guess it's a medical text. But retinopathy appears only in a frequency of 0.0001% in english!  That's very hard to train a network to correcty handle cases for which there is not many examples!

Another example of complexity is multiple words can mean the same or almost the same thing like "cat" or "kitty". So we sould like to share parameters of cases like this, but cat is completely different from kitty! So we have to learn that they are related.

To solve that problem, we are going to turn to **unsupervised learning** ! It's training without any label.
