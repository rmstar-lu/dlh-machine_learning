

# Optimization techniques for neural networks

## Feature Scaling

Features often need to be rescaled to get similar absolute values. Otherwise large numerical differences will dominate squared distances, e.g. body mass will dominate other measurements in the penguin data.

This is a pre-processing/data quality kind of operation which applies to all distance- or similarity-based algorithms and algorithms assuming normally distributed features.

E.g., KNN (k nearest neighbours) depends on Euclidean distance so needs features to be scaled.

PCA (principal component analysis), being variance-based, needs all features to be 0-centered and have equal magnitudes and therefore variances.

Gradient descent in neural networks will converge faster with normalization. Without normalization the step will be too high for low-magnitude features and too small for high-magnitude features. Normalizing will make the loss function topology more spherical when it is an elongated ellipsoid. See batch normalization below for details.

Note that tree-based methods or LDA (linear discriminant analysis) do not need feature scaling as they respect different feature scales by design.

Feature scaling must be done before fitting the model, and of course it needs to be applied to the test data in exactly the same way to get correct predictions.

There are the following ways to do it:

### Standardisation:

When features can be assumed to be normally distributed.

$$x' = \frac{x - \mu}{\sigma}$$
$$x'\sim\cal{N}(0, 1)$$

### Mean normalisation:

$$x' = \frac{x - \bar{x}}{\max(x) - \min(x)}$$
$$x' \in [-1, 1], \bar{x}' = 0$$

### Min-max scaling:

$$x' = \frac{x - \min(x)}{\max(x) - \min(x)}$$
$$x' \in [0, 1]$$

### Robust scaling (IQR scaling):

This is less sensitive to outliers than min-max scaling or mean normalisation.

$$x' =\frac{x - Q_2(x)}{Q_3(x) - Q_1(x)}$$
$$Q_2(x') = 0, Q_3(x') - Q_1(x') = 1$$

### Unit-vector scaling:

$$x' = \frac{x}{\lVert x\rVert}$$
$$\lVert x'\rVert = 1$$

## Batch normalization

This is a technique to make neural networks learn faster because of better loss-function topology and more uniform gradients. It also has a small regularization effect.

During training, normalization is applied in the forward-prop phase to the pre-activations $Z$ of a layer, by computing (element-wise)

$$Z' = \frac{Z - \mu}{\sqrt{\sigma^2+\epsilon}}$$

where $\mu$ and $\sigma^2$ are vectors of dimension $d$ (number of outputs) representing the mean and variance of $Z$ by output dimension over the whole current mini-batch.

Then the $Z'$ are re-scaled by *learned* parameters $\gamma$ and $\beta$ (vectors of dimension $d$, initialized to all-one and all-zero respectively) giving

$$\tilde{Z} = \gamma Z' + \beta$$

which is then fed instead of $Z$ into the activation function.

In return, one can do without the bias parameter in the neural network layer, because it has been subtracted away and replaced by $\beta$.

During inference, it is not possible to compute $\mu$ and $\sigma^2$ since there is usually only one sample available.

Instead, one uses an exponential moving average of $\mu$ and $\sigma^2$ computed during training which is then adjusted online during inference.

## Gradient descent

Gradient descent minimizes a loss function $\cal{L}(\theta)$ parameterized by a model's parameters $\theta\in ℝ^d$ by iteratively updating the parameters in the opposite direction of the gradient $\nabla_\theta\cal{L}(\theta)$.

The learning rate $\eta$ determines the size of the steps taken.

$$\theta_t = \theta_{t-1} - \eta \cdot\nabla_\theta \cal{L}(\theta)$$

When training neural networks, there are the following variants of gradient descent:

### Batch gradient descent

The gradient is computed for the entire training dataset to perform each update. This can be very slow, or even impossible if the dataset does not fit into memory. It also does not allow for online learning, i.e. adding single examples and re-training on the fly.

### Stochastic gradient descent (SGD)

The gradient is updated for each training example separately. Every time the model has seen all the data, this marks an *epoch*. The training examples should be shuffled at the start of each epoch. SGD is usually much faster than batch gradient descent and can be used to learn online. SGD performs updates with a high variance which causes the loss function to fluctuate wildly. Because of this fluctuation, SGD might find better local minima than batch gradient descent.

### Mini-batch gradient descent

This updates the gradient for a batch of size $n$ (`batch_size`) chosen from the shuffled training set. This reduces the variance of the updates, making convergence more stable. It can also be vectorized when using SOA libraries, making it very efficient. 

## Gradient descent optimization algorithms

Vanilla gradient descent struggles to converge in ravines, i.e. areas where the surface curves much more steeply in one dimension than in another (elongated ellipse). Progress toward the bottom will be slow while gradient descent oscillates across the slopes of the ravine.

### Momentum

This is a method that helps to accelerate gradient descent in the relevant direction. It works by computing an exponential moving average of past gradients/past updates.

$$v_t = \beta v_{t-1} + (1- \beta)\nabla_\theta\cal{L}(\theta)$$
$$\theta_t = \theta_{t-1} - \eta \cdot v_t$$

or, similarly (as done by Keras):

$$v_t = \beta v_{t-1} - \eta \cdot \nabla_\theta\cal{L}(\theta)$$
$$\theta_t = \theta_{t-1} + v_t$$

The $\beta$ is typically set to $0.9$ or similar.

This causes the momentum term to increase for dimensions whose gradients point in the direction of the accumulated past updates, and makes it decrease for dimensions where the sign is different. Like rolling a ball down a hill.

One can modify the momentum formula to use "look-ahead" in the calculation of the gradient:

$$v_t = \beta v_{t-1} - \eta \cdot \nabla_\theta\cal{L}(\theta + \beta v_{t-1})$$
$$\theta_t = \theta_{t-1} + v_t$$

This is called *Nesterov* momentum.

### RMSProp

Geoff Hinton, [lecture6 of coursera class](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).

This uses an adaptive learning rate which depends inversely on an EMA of squared gradients (second moment).

$$s_t = \beta s_{t-1} + (1 - \beta) [ \nabla_\theta\cal{L}(\theta)]^2$$
$$\theta_t = \theta_{t-1} - \eta \frac{1}{\sqrt{s_t + \epsilon}}\cdot \nabla_\theta\cal{L}(\theta)$$

The $\beta$ is typically set to $0.9$, and $\eta$ to $0.001$.

### Adam

Adaptive moment estimation (Kingma and Ba, 2015) combines momentum and RMSProp. If momentum is a ball running down a slope, then Adam is like a heavy ball with friction, making it find flat minima in the loss function surface.

It works by computing EMA's of both the first and second moments (mean and uncentered variance) of the gradient.

$$v_t = \beta_1 v_{t-1} + (1- \beta_1)\nabla_\theta\cal{L}(\theta)$$
$$s_t = \beta_2 s_{t-1} + (1- \beta_2)[\nabla_\theta\cal{L}(\theta)]^2$$

Bias correction is applied to both averages:

$$\hat{v}_t = \frac{v_t}{1 - {\beta_1}^t}$$
$$\hat{s}_t = \frac{s_t}{1 - {\beta_2}^t}$$

The update rule is:

$$\theta_t = \theta_{t-1} - \eta \frac{1}{\sqrt{\hat{s}_t}+\epsilon}\hat{v}_t $$

The authors propose $\beta_1=0.9$, $\beta_2 = 0.999$ and $\epsilon=10^{-8}$.

## Learning rate decay

In gradient descent it is best to start with a higher learning rate $\eta$ and then gradually decrease it to allow the algorithm to "zone in" and eventually converge on a minimum.

This is typically done using a learning rate schedule such as inverse time decay

$$\eta_{\mathit{epoch}} = \eta_0\cdot\frac{1}{1 + \mathit{epoch}\cdot \mathit{decay}}$$

or exponential decay

$$\eta_{\mathit{epoch}} = \eta_0\cdot 0.95^{\mathit{epoch}}$$

but it could also be done manually.
