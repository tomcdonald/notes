# Weight Space View of Gaussian Processes

* Notes from [Chapter 2.1 of Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/chapters/RW.pdf)

By considering a Bayesian approach to the standard linear model, we can describe a weight-space view of Gaussian Processes (GPs) as opposed to the typical view in which inference takes place in function-space. We can also project the inputs into high-dimensional feature space and apply a *kernel trick* in order to efficiently carry out computations within the high-D space, which is useful when the dimensionality of the feature space is much larger than the number of datapoints used.

Consider training set of $n$ datapoints, $\left\{\mathbf{x}_i, y_i\right\}^n_{i=1}$ where all the $D$ dimensional inputs $\mathbf{x}_i$ are placed in a design matrix $\mathbf{X} \in \R^{D\times n}$ and the outputs are concatenated into a 1D vector $y \in \R^n$.

## 1. Bayesian Linear Model

The Bayesian analysis of the standard linear model with Gaussian noise takes the form:

$$f(\mathbf{x})=\mathbf{x}^\top \mathbf{w}, \\ y=f(\mathbf{x})+\epsilon$$

where $\mathbf{x}$ is the input, $w$ are weights/parameters of the linear model, $f$ is the function value and the I.I.D Gausian noise is given by $\epsilon \sim \mathcal{N}(0, \sigma_n^2)$. 

This I.I.D assumption allows us to construct the likelihood factored over the datapoints:

$$p(\mathbf{y}|\mathbf{X,w}) = \prod^n_{i=1} p(y_i|\mathbf{x}_i, \mathbf{w}) = \mathcal{N}(\mathbf{X}^\top \mathbf{w}, \ \ \ \sigma^2_n \mathbf{I})$$

We place a zero-mean Gaussian prior over our weights with covariance matrix $\mathbf{\Sigma}_p$, such that $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \mathbf{\Sigma}_p)$. We can then perform inference by computing the posterior distribution over the weights using Bayes' Rule:

$$p(\mathbf{w| y, X}) = \frac{p(\mathbf{y|X,w})p(\mathbf{w})}{p(\mathbf{y|X})}$$

where the normalisation constant (i.e. the marginal likelihood) is given by the sum rule of probability:

$$p(\mathbf{y|X}) = \int p(\mathbf{y|X,w})p(\mathbf{w})d\mathbf{w}$$

By expanding the form of the likelihood, ignoring terms from the likelihood and prior which aren't dependent on the weights and completing the square, we can write the posterior as a Gaussian:

$$p(\mathbf{w|X,y}) \sim \mathcal{N}\left(\frac{1}{\sigma_n^2}\mathbf{A}^{-1}\mathbf{Xy}, \ \ \ \ \mathbf{A}^{-1}\right)$$

where $\mathbf{A}^{-1} = \sigma_n^{-2}\mathbf{XX}^\top + \mathbf{\Sigma_p^{-1}}$. For this and any other Gaussian posterior, the mean of the distribution is also the mode, known as the *maximum a posteriori* (MAP) estimate of $\mathbf{w}$. Non-Bayesian approaches such as *ridge regression* take the negative log-prior to be a penalty term and the MAP estimate to be the penalised MLE of the weights, but in the Bayesian framework, the MAP estimate has no massive significance.

We can use the posterior the make predictions for a test input by averaging over all possible values of the weights, weighted by their posterior probability. Specifically, the predictive distribution for test input $\mathbf{x}_*$ is given by:

$$
\begin{aligned}
p(f_* |\mathbf{x}_*, \mathbf{X, y}) &= \int p(f_*|\mathbf{x}_*, \mathbf{w})p(\mathbf{w|X, y})d\mathbf{w} \\
&= \mathcal{N}(\left(\frac{1}{\sigma_n^2}\mathbf{x}_*^\top\mathbf{A}^{-1}\mathbf{Xy}, \ \ \ \ \mathbf{x}_*^\top\mathbf{A}^{-1}\mathbf{x}_*\right)
\end{aligned}
$$

In other words, to get the mean of the predictive distribution we multiply the mean of the posterior by the test input, and the variance contains a quadratic form of the test input, which makes intuitive sense as the uncertainty in a linear model should grow with input magnitude.

## 2. Projection of Inputs into Feature Space

The Bayesian linear model described above has limited flexibility and ability to express relationships between variables, in much the same way the standard linear model does. We can attempt to overcome this issue by projecting the inputs into a high-D feature space using basis functions, and then perform the linear modeling in this high-D space instead of on the inputs themselves. In order to preserve the linearity of the model, the basis functions must be fixed, independent of the parameters $\mathbf{w}$, which ensures tractability. This is a similar concept to approaches to classification which involve projecting data into high-D spaces to achieve linear separability. Ch.5 of GP for ML discusses the choice of these basis functions further but we assume here the functions are fixed.

If we introduce a basis function $\phi$ which maps the $D$ dimensional input vector into an $N$ dimensional feature space, we can write the model as:

$$f(\mathbf{x})=\phi(\mathbf{x})^\top \mathbf{w}$$

Also, we can collect the projections for each input into a matrix $\mathbf{\Phi(X)}$, so we can basically just replicate the standard linear model from before, but with all instances of $\mathbf{X}$ replaced with $\mathbf{\Phi(X)}$. Our predictive distribution becomes:

$$ p(f_* |\mathbf{x}_*, \mathbf{X, y}) \sim \mathcal{N}\left(\frac{1}{\sigma_n^2}\phi(\mathbf{x}_*)^\top\mathbf{A}^{-1}\mathbf{\Phi y}, \ \ \ \ \phi(\mathbf{x}_*)^\top\mathbf{A}^{-1}\phi(\mathbf{x}_*)\right) $$

where $\mathbf{\Phi} = \mathbf{\Phi(X)}$ and $\mathbf{A}^{-1} = \sigma_n^{-2}\mathbf{\Phi \Phi}^\top + \mathbf{\Sigma_p^{-1}}$. Computing predictions using this equation involves inverting $A$, which is a $N \times N$ matrix; this is very computationally intensive. We can rewrite the equation in the form:

 $$
 \begin{aligned}
p(f_*|\mathbf{x}_*, \mathbf{X, y}) \sim \mathcal{N}&( \mathbf{\phi}_*^\top \mathbf{\Sigma_p} \mathbf{\Phi}(\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1}\mathbf{y}, \\ & \mathbf{\phi}_*^\top \mathbf{\Sigma}_p \mathbf{\phi}_* -  \mathbf{\phi}_*^\top \mathbf{\Sigma}_p \mathbf{\Phi}(\mathbf{K}+ \sigma_n^2 \mathbf{I})^{-1} \mathbf{\Phi}^\top \mathbf{\Sigma}_p \mathbf{\phi}_* )
\end{aligned}
 $$

 where $\mathbf{\phi}_* = \phi(\mathbf{x}_*)$ and $\mathbf{K} = \mathbf{\Phi}^\top \mathbf{\Sigma}_p \mathbf{\Phi}$. By this formulation, now we only need to invert matrices of size $n \times n$, which is useful when $n < N$. The entries of the matrices within which the feature space enters the expression take the form $\phi(\mathbf{x})^\top \mathbf{\Sigma}_p \phi(\mathbf{x^\prime})$ which we now call $k(\mathbf{x, x^\prime})$, the kernel/covariance function, where the inputs are points in the training or test data sets. The kernel is an inner product and thus can be written as a simple dot product representation $k(\mathbf{x, x^\prime}) = \mathbf{\psi(x)} \cdot \mathbf{\psi(x^\prime)}$ where $\mathbf{\psi(x)} = \mathbf{\Sigma}_p^{\frac{1}{2}}\phi(\mathbf{x})$.

 If an algorithm is defined in terms of inner products in input space, we can transfer it into feature space by replacing said inner products with $k(\mathbf{x, x^\prime})$; this is known as the *kernel trick*. Often it is easier to compute the kernel than the feature vectors which gives the approach added significance.