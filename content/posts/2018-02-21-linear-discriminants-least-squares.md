---
title: "Linear discriminants and the least-squares method"
date: 2018-02-21
---


Let's say we have a classification problem of $K$ classes. That is: for any given data point $x \in \mathbb{R}^D$, we want to decide to which class $k \in \underline{K}$ $x$ belongs. We are given a training set $x_1, \dots, x_N$, $x_i \in \mathbb{R}^D$ along their labels $t_1, \dots, t_N$ where $t_i \in \{0,1\}^K$, $(t_i)_j = 1 \iff x_i \text{ belongs to class } j$.

Obviously, we want to use the examples we have been given to estimate a function $r: \mathbb{R}^D \rightarrow \{0, 1\}$ that classifies any given data point as well as possible. How can we do this?

## Discriminant functions

One approach is to define what is called a discriminant function per possible class. A discriminant function for class $k \in \underline{K}$ is simply a function $y_k: \mathbb{R}^D \rightarrow \mathbb{R}$. To classify a point $x \in \mathbb{R}^D$, we choose the class $k$ whose discriminant function yields the heighest value: $$
    r: \mathbb{R}^D \rightarrow \underline{K}, x \mapsto \underset{k}{\operatorname{argmax}}\{y\_k(x)\}
$$

## Linear discriminant functions

We haven't yet specified what kind of function the discriminants should be. In principle, we could use anything. For the beginning, we focus on simple linear functions:
$$
y\_k(x) = w\_k^T \cdot x + w\_{k,0} = w\_{k,D} \cdot x\_D + \dots + w\_{k,1} \cdot x\_1 + w\_{k,0}
$$
where $w\_k \in \mathbb{R}^D$ is referred to as the weight vector and $w\_{k,0} \in \mathbb{R}$ is referred to as bias term. Note that we could simplify this equation by setting $x\_0 = 1$, thereby increasing the dimension of our $x$ by one and considering $w\_{k,0}$ as another entry in the weight vector.

## Learning linear discriminant functions: the least-squares method

How could we go about learning a set of linear discriminant functions such that the error we make is minimized? Well, first we need to define what we mean by error. Notice that the error we make on a given training set $x\_1, \dots, x\_n$ with target vectors $t\_1, \dots, t\_n$ should be measured in terms of the weights $w\_1, \dots, w\_K$ of the $K$ discriminant functions $y\_k$, as we plan to improve upon the weights and thus need to have a way of telling how "good" a certain set of weight vectors is.

When we use the least-squares-method of approximating the discriminant functions, we sum up the squared errors of classifying any point in the dataset with any discriminant function. Here is the formal definition:
$$
E(w\_1, \dots, w\_k \ | \ x\_1, t\_1, \dots , x\_N, t\_N) = \sum\_{n=1}^N \sum\_{k=1}^K ( y\_k(x\_n) - t\_{n,k} )^2 \\\\\\\\
    = \sum\_{n=1}^N \sum\_{k=1}^K ( w\_k^T \cdot x\_n - t\_{n,k} )^2
$$

Now, how can we minimize this error? The same way we have been minimizing functions in high school. We find the derivative with respect to the weight vectors and set it to zero. From that we can derive a minimizing condition. We could do this by taking the derivative of $E$ with respect to an arbitrary weight vector entry $w\_{k,i}$, but that would take a lot of sum signs and indices. Instead, let us restate the problem as a matrix equation.

Let $W = \begin{pmatrix}w\_1 \dots w\_K\end{pmatrix} \in \mathbb{R}^{(D+1) \times K}$ be the matrix of our adapted weight vectors. Then $y\_k(x) = (W^T \cdot x)\_k$. We can thus define
$$
    y(x) = W^T \cdot x = \begin{pmatrix}y\_1(x) \\\\\\\\ \dots \\\\\\\\ y\_K(x)\end{pmatrix}
$$
If we evaluate $y(x\_i)$ for a vector $x\_i$ in the training set, we can set the goal that it should approximate the corresponding target vector $t\_i$; that is: if $x\_i$ belongs to class $k$, all entries of $y(x\_i)$ that are not the $k$-th entry should be close to zero, and the $k$-th entry should be close to 1.

If we set the matrix $X = \begin{pmatrix} x\_1^T \\\\\\\\ \dots \\\\\\\\ x\_N^T\end{pmatrix} \in \mathbb{R}^{N \times (D+1)}$, we can evaluate $y(x\_n)$ for all $n \in \underline{N}$ simultaneously:
$$
    W^T \cdot X^T = \begin{pmatrix}y\_1(x\_1) \dots y\_1(x\_N)\\\\\\\\ \dots \\\\\\\\ y\_K(x\_1) \dots y\_K(x\_N)\end{pmatrix} \overset{!}{=} \begin{pmatrix}t\_1 \dots t\_N\end{pmatrix}
$$
or, equivalently
$$
    X \cdot W = \begin{pmatrix}y\_1(x\_1) \dots y\_K(x\_1)\\\\\\\\ \dots \\\\\\\\ y\_1(x\_N) \dots y\_K(x\_N)\end{pmatrix} \overset{!}= \begin{pmatrix}t\_1^T \\\\\\\\ \vdots \\\\\\\\ t\_N^T\end{pmatrix} =: T
$$
where $X \cdot W \overset{!}= T$ is supposed to mean that we want $X \cdot W$ to be equal to $T$.

Now let us express the least-squares criterion in matrix notation. We use $\odot$ to denote the elementwise multiplication of two matrices.
$$
    E(W) = \frac{1}{2} \cdot \sum\_{i,j} [(XW - T) \odot (XW - T)]\_{i,j}
$$
We square every entry of the matrix $XW - T$ and sum all the entries of the resulting matrix. Using the identity $\sum\_{i,j} a\_{i,j}^2 = Tr(A^T \cdot A)$, where $Tr$ denotes the matrix trace (sum of the diagonal entries), we arrive at
$$
    E(W) = \frac{1}{2} \cdot Tr[(XW-T)^T \cdot (XW-T)]
$$
Let us now find the derivative with respect to the weights:
$$
    \frac{\partial E(W)}{\partial W} = \frac{1}{2} \frac{\partial}{\partial W} \cdot Tr[(XW-T)^T \cdot (XW-T)] \\\\\\\\
    = \frac{1}{2} \frac{\partial (XW-T)^T \cdot (XW-T)}{\partial W} \cdot \frac{\partial Tr[(XW-T)^T \cdot (XW-T)]}{\partial (XW-T)^T \cdot (XW-T)} \\\\\\\\
    = X^T \cdot (XW - T)
$$

To find a minimum, we set this derivative to zero:
$$
X^T \cdot (XW - T) = 0
$$
As we assume that $X \not= 0$, this is equivalent to
$$
    XW = T \\\\\\\\
    \iff X^T \cdot X \cdot W = X^T \cdot T \\\\\\\\
    \iff W = \underbrace{(X^T \cdot X)^{-1} \cdot X^T}\_{\text{pseudo-inverse } X^\dagger} \cdot T
$$

We have thus found a direct expression for $W$ minimizing the squared error criterion! (Okay, we haven't verified that this is actually an optimum, but we could..)

Let us now try to use this result by applying it to a toy example.

## Experimenting with the least-squares method

We start of by generating two clusters of points in $\mathbb{R}^2$, drawn from independent Gaussian distributions.


```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

sampleSize = 50

meanA = np.array([0, 0])
varianceA = np.array([0.3, 0.5])
samplesA = np.random.normal(meanA, varianceA, (sampleSize, 2))

meanB = np.array([2, -2])
varianceB = np.array([0.2, 0.2])
samplesB = np.random.normal(meanB, varianceB, (sampleSize, 2))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlim([-3, 3])
plt.ylim([-3, 3])

ax.scatter([x for [x, y] in samplesA], [y for [x, y] in samplesA], marker="x", color="b")
ax.scatter([x for [x, y] in samplesB], [y for [x, y] in samplesB], marker="o", color="r")
```




    <matplotlib.collections.PathCollection at 0x7f0e82b9b358>




![png](output_2_1.png)


We now want to find linear discriminants $y\_0$ and $y\_1$ that classify all points in $\mathbb{R}^2$ as belonging to either the blue set of points or the red set of points. How can we visualize these discriminant functions? We can try to find the set of points $\{ x \in \mathbb{R}^2 | y\_0(x) = y\_1(x) \}$! As the discriminant functions are linear, the solution will be a subspace of $\mathbb{R}^2$. In particular, it will be a straight line! Suppose we are given the weight vectors $w, v \in \mathbb{R}^3$ of $y\_0$ and $y\_1$. How can we compute this decision boundary?

$$
    y\_0(x) = w^T \cdot x = v^T \cdot x = y\_1(x) \\\\\\\\
    \iff w\_0 + w\_1 \cdot x\_1 + w\_2 \cdot x\_2 = v\_0 + v\_1 \cdot x\_1 + v\_2 \cdot x\_2 \\\\\\\\
    \iff x\_2 = \frac{w\_0 - v\_0 + x\_1 \cdot (w\_1 - v\_1)}{v\_2 - w\_2}
$$
Let's do this!


```python
# add the 1s to the x vectors so that we can encode the bias as a weight
Xa = np.ones((sampleSize, 3))
Xa[:,1:] = samplesA
Xb = np.ones((sampleSize, 3))
Xb[:,1:] = samplesB

# build the complete data matrix X by concatenating Xa and Xb
X = np.concatenate((Xa, Xb))

# build the according target vectors
Ta = np.tile([1,0], (sampleSize, 1))
Tb = np.tile([0,1], (sampleSize, 1))
# and put them in one matrix
T = np.concatenate((Ta, Tb))

# numpy has a method for computing the pseudo-inverse directly,
# so this way we don't need to type it out ourselves
W = np.linalg.pinv(X) @ T

# extract the weight vectors of y_0 and y_1
w, v = W[:,0], W[:,1]

# plot the decision boundary
x = np.arange(-4, 4, 1)
y = (w[0] - v[0] + x * (w[1] - v[1])) / (v[2] - w[2])
ax.plot(x, y, color="black")
fig
```




![png](output_4_0.png)



Nice! This seems to have worked very well. But let's see what happens if our data isn't that easy to separate by a linear function. Let's add a couple of outliers to the blue data set.


```python
# let us add some outliers
num_outliers = 20

meanC = np.array([-2, 1])
varianceC = np.array([0.2, 0.2])
samplesC = np.random.normal(meanC, varianceC, (num_outliers, 2))

Xc = np.ones((num_outliers, 3))
Xc[:,1:] = samplesC

X = np.concatenate((Xa, Xc, Xb))
Tc = np.tile([1,0], (num_outliers, 1))
T = np.concatenate((Ta, Tc, Tb))

W = np.linalg.pinv(X) @ T


w, v = W[:,0], W[:,1]
# w is the weight vector of y_A, v is the weight vector of y_B

x = np.arange(-4, 4, 1)
y = (w[0] - v[0] + x * (w[1] - v[1])) / (v[2] - w[2])

ax.scatter([x for [x, y] in samplesC], [y for [x, y] in samplesC], marker="x", color="b")
ax.plot(x, y, color="orange")
fig
```




![png](output_6_0.png)



Hm. The black line, which is our old decision boundary, would still be a fine decision boundary, but our new decision boundary (orange) even misclassifies some of our data points. This shows us that the least-squares method is very sensitive to outliers.
