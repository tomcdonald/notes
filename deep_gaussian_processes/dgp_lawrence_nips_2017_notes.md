# Deep Probabilistic Modelling with Gaussian Processes 
## Neil Lawrence - NIPS 2017 Tutorial

*23/06/2020*

Available at [this link](https://www.youtube.com/watch?v=NHTGY8VCinY&list=WL&index=4&t=0s). Notes below definitely aren't exhaustive, worth a rewatch for Part 1 (regular GPs) and discussion of the equations which relate a DNN to a DGP.

### Questions on Part 1 (GPs)

- **Why set the mean to zero in a GP?** People often do fit the mean function for some applications, especially if they have prior knowledge based on a physical system for example. However, it's often best not to for general modeling as by doing so you introduce a non-probabilistic construct whose uncertainty is highly correlated with that of the GP, therefore it will be very difficult to find this uncertainty and keep the nice property of GPs that is the error bars surrounding the mean function. In some statistics appliactions, the mean function is specified but not treated probabilistically, in other words, it is the noise which is treated probabilistically.

### Part 2 - Deep Gaussian Processes (DGP)

#### Sparse GPs

- The primary motivation behind deep GP architectures is the issue with GPs that if you feed nonlinearities into GPs they become non-Gaussian in nature.

- Sparse GPs are generally how GPs are used in practice with lots of data. For a sparse GP we use a low rank approximation to the covariance function by compressing the information from the training set into a smaller number of 'pseudo-observations'/inducing points rather than using all the data. This is still a non-parametric model.

- This is similar in nature to the Random Fourier Feature approximation but is much sparser in nature.

- We optimise with respect to the inducing variable locations and kernel parameters jointly such that the inducing variables are placed at the locations which provide the best model fit to the training data.

- Approximations of this form are necessary to develop a deep GP model.

- See Damianou PhD thesis, ***DGPs and Variational Propagation of Uncertainty* (2015)** for application of this idea to deep architectures.

#### Brief Comparison of DNN vs DGP Architectures

- GPs are mathematically elegant and algorithmically complex, whereas Deep Neural Networks (DNNs) are mathematically inelegant but also algorithmically simple, which makes them easy to work with and thus, very popular.

- A common problem with DNNs is that if the width (i.e. number of a nodes) in two adjacent hidden layers is large, the corresponding weight matrix $\mathbf{W}$ will have a huge number of parameters, leading to possibility of overfitting. This is handled in practice by using dropout, however an alternative approach could be to parameterise $\mathbf{W}$ using its Singular Value Decomposition (SVD):

$$ \mathbf{W} = \mathbf{U\Lambda}\mathbf{V}^T \equiv \mathbf{U}\mathbf{V}^T \\ 
\text{where  } \mathbf{W}\in\R^{\mathcal{k_1\times k_2}}, \mathbf{U}\in\R^{\mathcal{k_1\times q}},\mathbf{V}\in\R^{\mathcal{k_2\times q}}$$

> This equation gives a low rank factorisation of the matrix weights $\mathbf{W}$, where $q< k_1 \text{ and } k_2$. Basically, we're 'bottlenecking' the neural network at each layer, which effectively looks mathematically like stacking a set of single layer neural networks (see 01:15:00 approx. for the equations).

- If we take the number of hidden units to infinity and place a prior/integrate over all the parameters on input and output layer, we have a GP. This is a nice feature, as it allows us to replace each of our DNNs with a GP, leaving us with the Deep Gaussian Process (DGP) model, a 'cascade of GPs':

- We can write a DGP as a composite multivariate function, i.e. a Markov Chain:

$$ p(\mathbf{y}|\mathbf{x}) = p(\mathbf{y}|\mathbf{f_3}) p(\mathbf{f}_3|\mathbf{f}_2) p(\mathbf{f}_2|\mathbf{f}_1) p(\mathbf{f}_1|\mathbf{x}) $$

- GPs also have elegant properties such as the fact that derivatives of GPs are also Gaussian distributed if they do indeed exist. Also, for certain covariance functions, GPs are 'universal approximators', such that all functions have support under the prior. These features make GPs suited for use in a stacked architecture.

#### Summary

- Deep GPs could be very useful for probabilistic numerics, surrogate modeling, emulation and uncertainty quantification, where typical deep models perform very poorly.

- An example is that designing an F1 car needs CFD, aero testing and limited track testing, how do combine this on a limited budget in order to maximise performance?

- As of 2017 (likely improved a little since but still work to do), the modeling framework is powerful but the software just isn't there. Ongoing work using Gaussian Processes underpinned by MXNet and other frameworks (e.g. GPyTorch) but not as widespread and well supported as the wealth of DNN software available.