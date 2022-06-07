### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 356000c0-882e-11ec-0fd0-1932daca92e7
begin
    using Optim
    using PlutoUI, PlutoTeachingTools, HypertextLiteral
    using Plots, LaTeXStrings
    using Random
    using Distributions
    using CSV
    using DataFrames
end

# ╔═╡ 38ff28bd-6406-436c-ac0d-faf125febf5b
md"""
We begin our exploration of the world of machine learning with the classic topic of linear regression.  While you have likely seen linear regression previously, linear regression is such a foundational tool that it's worth a quick refresher.  

Often in astrophysics research we'd like to find a relationship between variables. Occasionally, we look for relationships that we anticipate are likely to be linear.  For instance, in the absence of influence from external forces, we would expect a planet's orbit to be periodic.  In this case, linear regression would be the starting point for fitting a model to observed transit times to find the orbital period.  
"""
#Extraneous influence will cause the transit times to vary, allowing observers to infer properties of the other body(s), even if it does not transit.

# ╔═╡ 49d355c8-0894-4a59-b16a-05e79f90d493
md"""
# A Statistican's Perspective
Supppose we suspect a linear relationship between ``N`` observations of a response variable, $y$, and ``K`` regressor variables ($x_{i1}$, $x_{i2}$, ..., $x_{iK}$; also known as *features*) for each observation.  In summation notation, we have
```math
	y_i = \sum_{j = 1}^K \beta_j x_{ij} + \varepsilon_i, \quad \forall i = 1,\, 2,\, \ldots,\, N.
```
where $\mathbf{\beta}$ is an unobservable model parameter  
and $\varepsilon_i$ is the error associated with each observation $i$.

Alternatively, in matrix-vector equation, we can express the same model as
```math
	\mathbf{y} = \mathbf{X}\mathbf{\beta} + \mathbf{\varepsilon},
	\quad
	\{\,
		\mathbf{X} \in \mathbb{R}^{N \times K};\,
		\beta \in \mathbb{R}^K;\,
		\mathbf{y},\,\mathbf{\varepsilon} \in \mathbb{R}^N
	\,\},
```
which when fully expanded gives us a system of ``N`` equations with ``K`` variables,
```math
	\begin{bmatrix}
		y_1\\
		y_2\\
		\vdots\\
		y_N
	\end{bmatrix}
	=
	\begin{bmatrix}
		1 		& x_{12} & x_{13} & \ldots & x_{1K}\\
		1 		& x_{22} & x_{23} & \ldots & x_{2K}\\
		\vdots 	& \vdots & \vdots & \ddots & \vdots\\
		1 		& x_{N2} & x_{N3} & \ldots & x_{NK}
	\end{bmatrix}
	\begin{bmatrix}
		\beta_1\\
		\beta_2\\
		\vdots\\
		\beta_K
	\end{bmatrix}
	+
	\begin{bmatrix}
		\varepsilon_1\\
		\varepsilon_2\\
		\vdots\\
		\varepsilon_N
	\end{bmatrix},
```
Here ``X_{ij}`` is known as the _feature matrix_  or _design matrix_.
"""

# ╔═╡ 3ce217e6-d87b-4767-9abb-3cd9ad365444
Foldable(
	"
	Let us pause for a moment to ponder the implications of a system being overdetermined.  What do the columns in the feature matrix represent?

	(Click the arrow once you're ready to read a possible response.)",

	md"""As a concrete example, consider a ``3 \times 2`` matrix ``M``. The two columns of ``M`` are vectors from ``\mathbb{R}^3`` that span a 2-dimensional plane embedded in 3-dimensional space. Any vector that lies on this plane can be decomposed into a linear combination of the two column vectors of ``M``. This 2D plane is called the _column space_ of ``M``. This picture works for higher dimensions too!
	"""
)

# ╔═╡ d4874f23-4ad4-4cea-9893-f12aff99b878
md"""
The goal of linear regression is to find a statistic $\mathbf{\hat{\beta}}$ to estimate the true value of $\mathbf{\beta}$.   
From a statistican's perspective, one might maximize a likelihood or *a posteriori* probability to obtain an estimate of $\mathbf{\hat{\beta}}$.  Defining a likelihood requires specifying the distribution of $\varepsilon_i$'s.  

A common assumption is that $\mathbf{\varepsilon}$ is drawn from a normal distribution.
If each $\varepsilon_i$ is drawn from a normal distribution with the same variance, then one can estimate $\mathbf{\beta}$ by minimizing the sum (or mean) of the squared errors of the samples.  
We'll start by considering this case, which goes by the name *ordinary least squares*.  
Later, we'll consider the more general case of unequal measurement uncertainties or even correlated measurement uncertainties.  
"""

# ╔═╡ c6ca7cbc-0809-4cd2-a365-0f8fc16e3a3e
md"""
## Ordinary Least Squares

## Estimating β
Here, we'll briefly review how ``\mathbf{\hat{\beta}}`` is computed in the case of ordinary least squares, where the goal is to
```math
	\textrm{minimize}\; ||\,\mathbf{y} - \mathbf{X}\mathbf{\hat{\beta}} \,||^2.
```

We gain some intuition by approaching ordinary least squares from two perspectives: calculus and linear algebra.  In both cases, aim for refreshing your understanding of the approach and becoming familiar with the notation, rather than worrying about the mathematical details of each step of the derivation.

### Calculus
First, we expand the squared-norm of the residual vector:
```math
	||\,\mathbf{y} - \mathbf{X}\mathbf{\hat{\beta}} \,||^2
	=
	\left(\mathbf{y} - \mathbf{X}\mathbf{\hat{\beta}}\right)^\intercal
	\left(\mathbf{y} - \mathbf{X}\mathbf{\hat{\beta}}\right)
	=
	\mathbf{y}^\intercal \mathbf{y}
	- 2\mathbf{\hat{\beta}}^\intercal\mathbf{X}^\intercal \mathbf{y}
	+ \mathbf{\hat{\beta}}^\intercal\mathbf{X}^\intercal \mathbf{X}\mathbf{\hat{\beta}} .
```
(Do you recognize the step we skipped over?)
Since we would like to find a ``\mathbf{\hat{\beta}}`` that minimizes the expression, we take the [matrix-derivative](https://www.gatsby.ucl.ac.uk/teaching/courses/sntn/sntn-2017/resources/Matrix_derivatives_cribsheet.pdf) with respect to ``\mathbf{\hat{\beta}}`` and set the expression to 0, i.e.
```math
	\dfrac{d}{d\mathbf{\hat{\beta}}}\left(\mathbf{y}^\intercal \mathbf{y}
	- 2\mathbf{\hat{\beta}}^\intercal\mathbf{X}^\intercal \mathbf{y}
	+ \mathbf{\hat{\beta}}^\intercal\mathbf{X}^\intercal \mathbf{X}\mathbf{\hat{\beta}}\right)
	=
	-2\mathbf{X}^\intercal \mathbf{y}
	+2\mathbf{X}^\intercal \mathbf{X}\mathbf{\hat{\beta}}
	=
	0
```
which finally yields us an expression for $\mathbf{\hat{\beta}}$:
```math
	\mathbf{X}^\intercal \mathbf{X}\mathbf{\hat{\beta}}
	=
	\mathbf{X}^\intercal \mathbf{y}
	\quad \rightarrow \quad
	\mathbf{\hat{\beta}}
	=
	\left(\mathbf{X}^\intercal \mathbf{X}\right)^{-1}\mathbf{X}^\intercal \mathbf{y}
```
"""

# ╔═╡ 5e971191-3353-428c-b26b-dfc0bcbf1c63
md"""
### Linear Algebra
Another approach which some find more intuitive is to think of the problem geometrically. The fundamental problem is to solve an overdetermined linear system (i.e., there are more constraint equations than unknown) of ``\mathbf{X}\mathbf{\beta} = \mathbf{y}``.  
"""

# ╔═╡ a61518a4-6e42-4d8a-af8d-4d2a9605eef1
md"""
Most of the time, the solution does not exist since ``\mathbf{y}`` is _not_ in the column space of ``\mathbf{X}``. We can, however, find a vector ``\mathbf{\hat{\beta}}`` that best estimates it.

For intuition, we can project ``\mathbf{y}`` onto the nearest vector that does lie in the column space of ``\mathbf{X}``, and solve that problem instead.  The linear system can now be solved but the solution is only the best approximation to the original problem.

The following animation may help to visualize this interpretation idea:"""

# ╔═╡ ee83d05b-4a46-4f62-a0eb-1a70b5f4be8a
html"""
<iframe id="kaltura_player" src="https://cdnapisec.kaltura.com/p/2356971/sp/235697100/embedIframeJs/uiconf_id/41416911/partner_id/2356971?iframeembed=true&playerId=kaltura_player&entry_id=1_3b711nk1&flashvars[streamerType]=auto&amp;flashvars[localizationCode]=en&amp;flashvars[leadWithHTML5]=true&amp;flashvars[sideBarContainer.plugin]=true&amp;flashvars[sideBarContainer.position]=left&amp;flashvars[sideBarContainer.clickToClose]=true&amp;flashvars[chapters.plugin]=true&amp;flashvars[chapters.layout]=vertical&amp;flashvars[chapters.thumbnailRotator]=false&amp;flashvars[streamSelector.plugin]=true&amp;flashvars[EmbedPlayer.SpinnerTarget]=videoHolder&amp;flashvars[dualScreen.plugin]=true&amp;flashvars[hotspots.plugin]=1&amp;flashvars[Kaltura.addCrossoriginToIframe]=true&amp;&wid=1_s6b8e414" width="91%" height="376" allowfullscreen webkitallowfullscreen mozAllowFullScreen allow="autoplay *; fullscreen *; encrypted-media *" sandbox="allow-forms allow-same-origin allow-scripts allow-top-navigation allow-pointer-lock allow-popups allow-modals allow-orientation-lock allow-popups-to-escape-sandbox allow-presentation allow-top-navigation-by-user-activation" frameborder="0" title="Column Space Projection"></iframe>
"""

# ╔═╡ d9be4bad-3292-4d71-a066-fce7dafb8865
md"""
Algebraically, a projection is an inner product and so we have
```math
	\mathbf{X}\mathbf{\beta} = \mathbf{y}
	\quad \rightarrow \quad
	\mathbf{X}^\intercal\mathbf{X}\mathbf{\hat{\beta}} = \mathbf{X}^\intercal\mathbf{y}
```
to which we get the same solution
```math
	\mathbf{X}^\intercal \mathbf{X}\mathbf{\hat{\beta}}
	=
	\mathbf{X}^\intercal \mathbf{y}
	\quad \rightarrow \quad
	\mathbf{\hat{\beta}}
	=
	\left(\mathbf{X}^\intercal \mathbf{X}\right)^{-1}\mathbf{X}^\intercal \mathbf{y}
```
"""

# ╔═╡ 420a9b91-8672-4c73-8a34-9e3f3a9b9cb5
md"""
## Example
Let's try this out in code. For demonstration purposes, we'll generate a simple dataset of 100 points that are generated by a linear model with normally distributed measurement noise.
"""

# ╔═╡ 82e7ca08-4149-4b0e-97bf-77f421d84acd
begin
	# For reproducibility
	Random.seed!(42)

    # We'll use 30 randomly sampled points from a uniform distribution for our x variable
    N = 100
    x = sort(rand(Uniform(0, 30), N))

	# The true unobserved relationship, β
	βₜᵣᵤₑ = [-50; 5]

	# Design matrix
	X = [ones(N) x]
    # The true unobserved relationship, β
    βₜᵣᵤₑ = [-50; 5]

	# Linear model
	f(X, β) = X * β
    # Design matrix
    X = [ones(N) x]

	# Get the true y values, and add "observation" noise that has been sampled from a Gaussian
	yₜᵣᵤₑ = f(X, βₜᵣᵤₑ)
	noise = rand.(Normal.(0, 10), N)
	y = yₜᵣᵤₑ + noise
    # Linear model
    f(X, β) = X * β

    # Get the true y values, and add "observation" noise that has been sampled from a Gaussian
    yₜᵣᵤₑ = f(X, βₜᵣᵤₑ)
    noise = rand.(Normal.(0, 10), N)
    y = yₜᵣᵤₑ + noise
end;

# ╔═╡ 87c75fa1-0a62-4e83-bc8c-db0b6a9befba
begin
    plot(x, y, st=:scatter, label="Observed Data")
    plot!(x, yₜᵣᵤₑ, label="True Function", legend=:bottomright)
    xlabel!(L"x")
    ylabel!(L"y")
end

# ╔═╡ e6c536ef-7202-44b6-aacf-020c152f74de
md"""
And we solve for the least square fit with:
"""

# ╔═╡ 4882f593-d707-46ff-8966-cb7813baad87
β̂ = (X'X) \ (X'y)

# ╔═╡ c511d6b3-96d2-44a8-933e-a01065dbf7bc
md"""
The backslash `\` in the equation above tells Julia to solve the linear system using a robust factorization method as alluded to earlier. For completeness, here's the solution if we approached it the naïve way:
"""

# ╔═╡ 0d1c5003-090d-4f96-8683-f3d6aeb16e62
inv(X'X) * (X'y)

# ╔═╡ b198560f-e55c-473a-bc36-c2f5e2a0fbfc
md"""
Finally, let's plot our least squares fit.
"""

# ╔═╡ 5cade76c-ebcf-4cbd-9707-108799452cf3
begin
    plot(x, y, st=:scatter, label="Observed Data")
    plot!(x, yₜᵣᵤₑ, label="True Function")
    plot!(x, X * β̂, label="Least Squares", legend=:bottomright)
    xlabel!(L"x")
    ylabel!(L"y")
end

# ╔═╡ 704d5b25-cd7e-4486-9ec3-8a8030146ba5
md"""
# The Machine Learning Perspective
Now that we're familiar with the linear algebra approach, we turn our attention to the ML approach. Luckily for us, most of the formulation remains the same, but we will switch notation standards.

``\\[2mm]``
$(Resource("https://imgs.xkcd.com/comics/standards_2x.png"))
Credit: [XKCD: 927](https://xkcd.com/927/)  This cartoon is licensed under under a [Creative Commons Attribution-NonCommercial 2.5 License](https://creativecommons.org/licenses/by-nc/2.5/).
``\\[6mm]``

So far, the notation we've used is one that is common in the land of Statistics. In the world of Machine Learning, the conventions are typically different in that ``\mathbf{\hat{\beta}}`` is written as ``\mathbf{\theta}`` -- a vector of parameters that we tune to *train* a model or *learn* a relationship.
We will adopt this standard in this section.
"""

# ╔═╡ ae885e79-d922-4017-b150-554bd7762c42
md"""
## Objective Function (or Loss Function)
To start tuning the parameters, we must first define an **objective function** or **loss function**, ``J``, that we would like to minimize.  Motivated by the log likelihood for OLS, we will consider the Mean-Squared Error (MSE) loss function:
```math
	J(x_i;\, \theta_0,\, \theta_1) = \dfrac{1}{N} \sum_{i = 0}^N (y_i' - y_i)^2
```

You're encouraged to implement this loss function in the next cell.  There are hint boxes to help.  Or if you'd rather continue on without implementing the loss function yourself, there's a check box below you can check (which will cause the rest of the lab to function properly with a reference implementation).
"""
# = \dfrac{1}{N} \sum_{i = 0}^N (\theta_0 + \theta_1x_i - y_i)^2

# ╔═╡ 7eeebedc-a056-4577-b625-d4fe71798985
function J(y, y′)
    # Replace the following line so it computes the loss function. It should return the mean squared error between the observed and predicted values.  
    return missing
end;

# ╔═╡ 964083ba-cdc1-42b0-a0fa-aa1de8ccee26
md"""
!!! hint "Hover over hint boxes to if/when you'd like some help."
    It may useful to use the functions `sum(z)` and `length(z)`.
    You can subtract two vectors using the `-` operator..
    To compute the square of every element in an array `z`, you can write `z.^2`.  
"""

# ╔═╡ e9a8d694-dbf7-49dc-ae75-fdcb67d93a7f
md"""
!!! hint "Want more help with the syntax?"
    There are many ways you could implement this.  For example,
	```julia
	function J(y, y′)
	    return sum((y - y′).^2) /length(y)
	end
	```
"""

# ╔═╡ c54763c1-d104-443c-895f-5729ea416ab0
md"""
Don't feel like writing the code yourself? Click here to skip ahead: $(@bind skip_loss_function CheckBox())
"""

# ╔═╡ cea9a7b8-e7d2-46e1-92d3-1dce48012693
md"""
## Finding the minima
The loss function chosen above, ``J(x;\, \theta_0,\, \theta_1)`` is differentiable.  This means that we could iteratively optimize the parameters using the gradient descent algorithm. If you would like a quick introduction to gradient descent (or to refresh your intuition behind the idea), then take a break to visit the [lab on Optimization using Gradient Descent](https://github.com/Astroinformatics/Optimization) from "day 0".

It is important to recognize that we are optimizing ``J`` with respect to the parameters ``\theta`` (rather than ``x``).  We can compute the partial derivatives analytically as
```math
\begin{align*}
\dfrac{\partial J}{\partial \theta_0} &= \dfrac{\partial}{\partial \theta_0} \left(\dfrac{1}{N} \sum_{i = 0}^N (\theta_0 + \theta_1x_i - y_i)^2\right)\\[2mm]
&= \dfrac{1}{N} \sum_{i = 0}^N \dfrac{\partial}{\partial \theta_0}\left((\theta_0 + \theta_1x_i - y_i)^2\right), \quad \textrm{by linearity of derivatives}\\[2mm]
&= \dfrac{1}{N}\sum_{i = 0}^N 2(\theta_0  + \theta_1x_i - y_i) = \dfrac{2}{N}\sum_{i = 0}^N (y_i' - y_i)
\end{align*}
```
and
```math
\begin{align*}
\dfrac{\partial J}{\partial \theta_1} &= \dfrac{\partial}{\partial \theta_1} \left(\dfrac{1}{N} \sum_{i = 0}^N (\theta_0 + \theta_1x_i - y_i)^2\right)\\[2mm]
&= \dfrac{1}{N} \sum_{i = 0}^N \dfrac{\partial}{\partial \theta_1}\left((\theta_0 + \theta_1x_i - y_i)^2\right), \quad \textrm{by linearity of derivatives}\\[2mm]
&= \dfrac{1}{N}\sum_{i = 0}^N 2(\theta_0  + \theta_1x_i - y_i)\,x_i = \dfrac{2}{N}\sum_{i = 0}^N (y_i' - y_i)\,x_i
\end{align*}
```
We can then use the computed partial derivatives to walk down the gradient by computing ``\theta`` value at iteration ``i + 1`` from iteration ``i`` via:
```math
\begin{align*}
	\theta_{0,\, i + 1} &= \theta_{0,\, i} - \eta \dfrac{\partial J}{\partial \theta_0}\\[2mm]

  	\theta_{1,\, i + 1} &= \theta_{1,\, i} - \eta \dfrac{\partial J}{\partial \theta_1}
\end{align*}
```
where ``\eta`` is the learning rate.
"""

# ╔═╡ 1e1f47cc-6b61-42f8-aebe-8a78779bd9e3
md"""
!!! note "Question:  Differences in choice of loss function"
	How would the gradient descent algorithm be affected if we had chosen to use mean of the squared errors (MSE) as our objective function instead of the sum of squared errors (SSE)?  What about if we had used the negative log likelihood for ordinary least squares?  How would the differences between these functions affect the local of the minimum and/or the gradient of the loss function?
"""

# ╔═╡ 0f5eb443-2884-4f0e-80b1-6bd0f10356aa
md"""
## Example
Enough talk! Let's see how this works in practice. We will reuse the synthetic dataset from earlier.
"""

# ╔═╡ 8a48af4d-2dd7-43cc-a41b-2be7311874ce
begin
    scatter(x, y, label="Noisy data")
    plot!(x, yₜᵣᵤₑ, label="True function", legend=:bottomright)
    xlabel!(L"x")
    ylabel!(L"y")
end

# ╔═╡ d9617fa2-c49c-4642-b1d7-7fc06faaef60
md"""
Next, we'll choose an initial guess for the model parameters and a number of iterations.  
"""

# ╔═╡ 96fd2bed-43f6-4505-9a56-c6ed0f9dc88d
# Initial guess for intercept and slope.  
theta_init = [-100, -20]      

# ╔═╡ 0f7bf676-15c9-488b-87fa-ae8c9cfbf5dd
num_iterations = 10_000 # Don't decrease below 400 or plots below will fail.

# ╔═╡ c70c6482-dabb-49c4-81ee-1f9dbae43fc9
md"The cell below runs the gradient descent algorithm."

# ╔═╡ 30b6cda7-39ba-4f65-a00f-1f5ba3736c0e
if ismissing(J(y, f(X, [0, 0])))
	md"""
!!! warning
    It appears that you haven't replaced code in the function `J` above *and*        that you haven't checked the box above to skip writing that function yourself.  
	Please do one of those before proceeding.
	"""
end

# ╔═╡ 0bc528e8-c596-43a2-81ac-47a52d51a1e2
md"""
If you're curious to see a simple implementation of gradient descent, you can expand the code cell below (by clicking the grey eyeball to the left of the cell).
"""

# ╔═╡ 71713d50-cac7-430c-9973-acdbe97c6712
md"""
**Question:**  Try changing the initial guess (by modifying the values of `theta_init` in the code cell three up).  Is the final result sensitive to your initial guess?  How does it affect the number of iterations required to find the minimum?
"""

# ╔═╡ dc29a574-cc88-4ed9-b0a7-f29d2107bc3d
md"""
**Question:** Depending on your initial guess, there may have been a plateau where the model barely improved for many iterations.  How could you recognize that continuing to run the gradient descent algorithm was likely to result in finding a significantly better model?  Why might that not be practical when working with a real data set ?
"""

# ╔═╡ a941ed2c-3a3d-4b87-8f43-3276b7fd6ef7
md"""
### Visualizing the loss function
To help visualize the gradient descent process, let us visualize the loss function as a surface and plot the ``i``th theta value at each iteration as the gradient descent algorithm searches for the optimum parameters.
"""

# ╔═╡ 7870356f-f6ce-40be-9c69-6ff7459ded02
md"""
**Question:**  What changes in the trajectory of the $\theta_i$'s near the plateau in the loss function?
"""

# ╔═╡ a022804a-9d68-409a-8a8f-abfa1cc61a10
md"""
How does the final estimate from gradient descent compare to the OLS estimator?  Based on the plot below, it might first look like they've performed equally well.  
"""


# ╔═╡ 501a28d0-3478-4564-a9f8-22deaec158c1
md"""
Upon closer inspection, we see that the estimates of $\hat{\beta}$ from OLS differ slightly from the final state of the gradient descent algorithm.  Based on comparing the loss function, we can see that gradient descent's estimate of the intercept and slope is slightly worse.
"""

# ╔═╡ 676168f5-a61a-4b4d-a8c2-56b0b5cfc3d5
md"""
(Click the arrows to read possible responses.)
"""

# ╔═╡ 135bcbd6-f050-49d6-b12c-512d4a534417
md"""
## Review of key assumptions in OLS
Now is a good time to remind ourselves of the assumptions that went into ordinary least squares:
1. **L**inear. The relationship between the variables are linear.
2. **I**ndependent. The errors ``\varepsilon_i`` are mutually independent
3. **N**ormality. The errors ``\varepsilon_i`` are normally distributed with 0 mean
4. **E**qual variances. The errors ``\varepsilon_i`` have equal variances
The assumptions form a convenient initialism to remember: LINE. Technically, you could contract the assumptions into only two parts:
1. Linear relationship between variables
2. The errors are normally distributed with mean 0, [independent and identically distributed (IID)](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)
But who doesn't love nice abbreviations? 🤷

Given that these assumptions are satisfied, OLS [guarantees](https://en.wikipedia.org/wiki/Gauss%E2%80%93Markov_theorem) a ``\mathbf{\hat{\beta}}`` that is the _best linear unbiased estimator_ (BLUE), and it is given by
```math
	\mathbf{\hat{\beta}} = (\mathbf{X}^\intercal\mathbf{X})^{-1}\mathbf{X}^\intercal\mathbf{y}.
```
We will still find an estimate if we used OLS when these assumptions are violated, but it may not be the best!

In the next section, we'll explore what happens when the assumptions are violated.
	"""

# ╔═╡ c24ae519-cdd6-4544-943e-ee6018da0810
md"""
# Incorporating Measurement Uncertainties

## Weighted Linear Regression
Often in astronomy & astrophysics, we have access to the measurement uncertainties (or at least estimate of the measurement uncertainties).  
Usually, these are not equal, violating one of the assumptions of ordinary least squares.  
If the measurement errors are all independent and uncorrelated, then we can use [weighted linear regression](https://en.wikipedia.org/wiki/Weighted_least_squares).  

If all of the assumptions we made on OLS are valid, then we would expect the errors, $\mathbf{\epsilon}$, to be drawn from a multivariate normal distribution with zero mean, and a  [covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix)
```math
	\mathbf{\Sigma} = \sigma^2 \mathbf{I}
	=
	\sigma^2
	\begin{bmatrix}
	1 		& 0 		& \ldots & 0\\
	0 		& 1 		& \ldots & 0\\
	\vdots 	& \vdots 	& \ddots & 0\\
	0 		& 0 		& \ldots & 1\\
	\end{bmatrix}.
```
However, when the errors have non-constant variance, the diagonal entries in the covariance matrix change. To address heteroscedasticity, we use _weighted linear regression_. Since the covariance matrix is still diagonal, it is trivial to invert it to obtain a diagonal weight matrix $\mathbf{W}$.  
```math
	\mathbf{\Sigma} = \begin{bmatrix}
		\sigma^2_1 		& 0 		& \ldots & 0\\
		0 		& \sigma^2_2 		& \ldots & 0\\
		\vdots 	& \vdots 	& \ddots & 0\\
		0 		& 0 		& \ldots & \sigma^2_N\\
	\end{bmatrix}
	\quad \rightarrow \quad
	\mathbf{W} = \mathbf{\Sigma}^{-1} =
	\begin{bmatrix}
		\sigma^{-2}_1 		& 0 		& \ldots & 0\\
		0 		& \sigma^{-2}_2 		& \ldots & 0\\
		\vdots 	& \vdots 	& \ddots & 0\\
		0 		& 0 		& \ldots & \sigma^{-2}_N\\
	\end{bmatrix}
```
This means that the objective function is changed to reflect the weight of each observation -- smaller variances have more importance than larger variances i.e.
```math
	J_\textrm{WLS}(x_i;\, \theta_0,\, \theta_1) = \dfrac{1}{N} \sum_{i = 0}^N \dfrac{1}{\sigma_i^2}(y_i' - y_i)^2
```
Thankfully, it's a simple upgrade and we can still perform weighted linear regression without incurring a significant performance penalty.  
"""

# ╔═╡ 75311f29-da64-44fe-9623-5e92da348f13
md"""
## Correlated Measurement Noise & General Linear Regression (optional)
If the measurement uncertainties are correlated, then we often model the errors as a multivariate normal distribution.
```math
\mathbf{\epsilon} \sim N(\mathbf{0}, \mathbf{\Sigma}),
```
where $\mathbf{\Sigma}$ is a covariance matrix.
In this case, we can make use of [generalized linear regression](https://en.wikipedia.org/wiki/Generalized_least_squares) and the best linear unbiased estimator becomes
```math
	\mathbf{\hat{\beta}} = (\mathbf{X}^\intercal \mathbf{\Sigma}^{-1} \mathbf{X})^{-1}
                           (\mathbf{X}^\intercal\mathbf{\Sigma}^{-1} \mathbf{y}).
```

For small problem sizes (matrix with less than $\sim10^3$ rows or columns), one can still compute $\mathbf{\hat{\beta}}$ efficiently via linear algebra.  
As the size of the system increases, the memory and computational costs rapidly increase. Sometimes there are special properties of the covariance matrix (e.g., banded, block diagonal) that allow it to be factorized efficiently.  Otherwise, you may be driven to taking the machine learning approach of estimating $\mathbf{\hat{\beta}}$ via an iterative algorithm.

"""

# ╔═╡ 03157430-e8a8-4daf-9689-802ccd1195f8
md"""
## General Linear Regression & Non-linear Models (optional)
Often in astronomy & astrophysics we need to work with models that are non-linear (at least as originally written).  However, in some cases, we can perform a change of variables that allows us to rewrite the problem as a form of linear regression on the transformed variable.  

One example would be replacing fluxes with log fluxes (or magnitudes) to fit a [log-linear model](https://en.wikipedia.org/wiki/Log-linear_model).  Sometimes this requires making some sacrifices (e.g., the measurement uncertainties probably aren't normally distributed in the log flux).  

Another example of applying general linear regression to solve a non-linear problem is fitting a sinusoidal model to a timeseries of observations.  On first inspection, the model
```math
v(t) = K \cos\left(\frac{2\pi (t-t₀)}{P}\right) + C,
```
for the radial velocity ($v$) of a star or planet at time $t$ with amplitude $K$ and orbital period $P$ appears to be non-linear.  
But making use of trigonometric relations, we can rewrite this as
```math
v(t) = A \cos\left(\frac{2\pi t}{P}\right) + B \cos\left(\frac{2\pi t}{P}\right) + C,
```
where the coefficients $A$ and $B$ can be related to $K$ and $t₀$.  If we knew the time of each observation and orbital period, $P$, then the model is linear in the parameters $A$, $B$ and $C$.  The trigonometric functions become the regressors and two columns of the design matrix.  In practices, we often don't know $P$ *a priori*.  But linear regression is so efficient, it's often computationally efficient to perform a brute force search over a grid of $P$ values, computing the best fit values of $A$ and $B$ for each value of $P$.  
This is the basis for the Lomb-Scargle periodogram (and many other periodograms) that is widely used for searching for periodicities in unevenly sampled time-series.


In both of the examples above the careful choice of regressors (or *features* in the machine learning parlance), the problem can be reframed as a problem of linear regression (or many linear regression problems).  In addition to accelerating the fitting process, rewriting a linear models in terms of linear regression can be very beneficial since the resulting models have a single global mode, are readily interpretable and are computationally efficient.  
In the next lab, we'll consider logistic regression, one example of a family of *generalized linear models* which makes use of a link function to help with classification problems.
	"""

# ╔═╡ 422aa4a5-5642-4e64-a20f-ec1727efcafc
md"""
## Generalized Linear Regression in the astronomical literature

Linear regression (and the numerous variants) are very common in the astronomical literature.  For some examples of general linear regression being applied to astronomical problems and comparisons how other more "advanced" methods performed on the same data, see:
- [Elliott et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016ASSP...42...91E/abstract)
- [Rafieferantsoa et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.4509R/abstract)
- [de Beurs et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020arXiv201100003D/abstract)

"""

# ╔═╡ dbbea6dd-2f02-420c-9ee6-9dc5bd9ebb30
md"# Setup & Helper Code"

# ╔═╡ 8806e08e-0af8-48dc-a580-314a6a0cf917
begin
    function reference_J(y, y′)
        return sum((y - y′) .^ 2) / length(y)
    end

    # Chooses which function to use based on the checkbox above.
    loss_function = skip_loss_function ? reference_J : J
end

# ╔═╡ 7752f520-80c2-4822-9acb-19afa24b13e5
function gradient_descent(X, y, θ; η=1e-3, num_iteration=15_000)
	@assert size(X,1) == length(y)
	@assert size(X,2) == length(θ)
	N = length(y)
	J_history = zeros(num_iteration + 1)
	θ_history = zeros(num_iteration + 1, size(X,2))

	# Manually prepare store the starting values and associated loss function
	y′ = f(X, θ)                         # predictions for starting parameters
	θ_history[1, :] .= θ
	J_history[1] = loss_function(y, y′)

    for iter in 2:num_iteration+1

        # Matrix-Vector form of gradient descent
        θ = θ - η * 2 / N * X' * (y′ - y)

		y′ = f(X, θ)                            # predictions  
		J_history[iter] = loss_function(y, y′)
		θ_history[iter, :] = θ[:]

	end

	return (;θ, J_history, θ_history)
end

# ╔═╡ a378cfdd-13fd-4bf9-ab5a-3d87e14413a2
θ_gd, j_gd_history, θ_gd_history = gradient_descent(X, y, theta_init, num_iteration=num_iterations);

# ╔═╡ 55a64cf8-73f8-42b9-9b84-05ac6f956024
md"""
Move the slider below to see how the model and loss function ($J$) change with increasing iterations.
Iteration: $(@bind n Slider([1:29; 30:10:400; range(400, size(j_gd_history, 1); step=400)], default=1))
"""

# ╔═╡ d2b4fe71-c159-46d3-8aaa-481598e7b75f
begin
    n_int = round(Int, n)

    p1 = scatter(x, y, label="Observed data")
    plot!(
        x,
        f(X, θ_gd_history[n_int, :]),
        xlabel=L"x",
        ylabel=L"y",
        label=L"$\theta = [%$(round(θ_gd_history[n_int, 1], digits=1)), %$(round(θ_gd_history[n_int, 2], digits=1))]^\intercal$",
        legend=:bottomright,
        foreground_color_legend=nothing
    )
    xlims!(-1, 17)
    ylims!(-80, 50)

    p2 = plot(
        1:n_int,
        j_gd_history[1:n_int],
        xlabel=L"$\log_{10}(\mathrm{Iteration})$",
        ylabel=L"$\log_{10}(J)$",
        xscale=:log10,
        yscale=:log10,
        legend=false
    )
    scatter!(
        [n_int],
        [j_gd_history[n_int]],
    )

    xlims!(0.5, 2e4)
    ylims!(10, 5e5)

    plot(p1, p2)
end

# ╔═╡ fa353f9c-2adb-473e-80db-f5664b5e1008
begin
    plot(x, y, label="Observed Data", st=:scatter,)
    plot!(x, yₜᵣᵤₑ, label="True Function")
    plot!(x, f(X, β̂), label="OLS")
    plot!(x, f(X, θ_gd), label="Gradient Descent", legend=:bottomright)
    xlabel!(L"x")
    ylabel!(L"y")
end

# ╔═╡ 3609cae3-263a-454a-8929-ea79e2021710
begin
	println("OLS Results")
	println("β̂: ", β̂)
	println("Loss: ", loss_function(y, f(X, β̂)), "\n")

	println("GD Results")
	println("θ_", num_iterations, ": ", θ_gd)
	println("Loss: ", loss_function(y, f(X, θ_gd)))
end

# ╔═╡ 244234eb-eb61-4333-817f-f7cec47fd111
TableOfContents()

# ╔═╡ d556b1d3-ef78-45a3-9d60-63645d1ceb48
function aside(x; v_offset=0)
    @htl("""
     <style>


     @media (min-width: calc(700px + 30px + 300px)) {
     	aside.plutoui-aside-wrapper {
     		position: absolute;
     		right: -11px;
     		width: 0px;
     	}
     	aside.plutoui-aside-wrapper > div {
     		width: 300px;
     	}
     }
     </style>

     <aside class="plutoui-aside-wrapper" style="top: $(v_offset)px">
     	<div>
     	$(x)
     	</div>
     </aside>

     """)
end

# ╔═╡ bd4bb1f9-00d4-418a-bd2a-04bb0ba07c92
aside(
    md"""
    !!! note "Notation"
    	The given notation sets up a **linear** system with ``K`` predictors. For the standard linear relationship between two variables, we recover the familiar equation of a straight line in 2D:
    	```math
    		y_i = \beta_1 + \beta_2 x_i + \varepsilon_i
    	```
    """,
    v_offset=150
)

# ╔═╡ 0176e0f9-36a6-4c9e-8221-10f6e3a74f19
aside(
    md"""
    !!! tip "Tip:  Numerical Linear Algebra"
    	Sovling linear systems numerically introduces some additional challenges, particularly for large systems of equations.  Computing the inverse of a matrix is computationally costly and can be numerically unstable.
        Therefore, one tries to avoid computing `inv(A)` whenever possible.
        Instead, one solves the matrix-vector equation using one of several algorithms that make use of matrix factorization (QR, LU, etc.).  
        In the case of ordinary least squares, one would solve the system:
    	```math
    		(\mathbf{X}^\intercal\mathbf{X})\mathbf{\hat{\beta}} = \mathbf{X}^\intercal\mathbf{y}
    	```
    """,
    v_offset=-320
)

# ╔═╡ c0edc83e-2cfa-4542-8f08-08a35cd4a178
aside(
    md"""
    !!! tip "Common Notations"
    	It is also common in the machine learning world to represent the predicted value (values generated from our model) with a primed variable (and to some extent hatted variables) in contrast to unprimed variables for the actual values!
    """,
    v_offset=550
)

# ╔═╡ 1b607612-80be-453a-970d-cc449bce6c64
aside(
    md"""
    !!! note "Unicode variables"
        Since Julia supports unicode variables, you can create the primed variable by 	typing out 'y\prime' and hitting `tab`. You can even use emojis as variable 	names! 🚀
    """,
    v_offset=-200
)

# ╔═╡ 817ef46e-4706-4743-854c-9bdb64df5243
aside(md"""
!!! tip "Pro Tip:  Autodifferentiation"
    In this example, we worked out the derivatives analytically.  For more complex problems, computing the derivatives analytically rapidly becomes tedious and error prone.  Fortunately, modern machine learning toolkits include packages that can calculate derivatives analytically.  Often there is a significant computational cost to working out the derivatives once, but subsequent calls can be only a little more expensive than calculating the loss function once.  There are multiple algorithms for automatic differentiation (or autodifferentiation).  Many packages provide multiple implementations, so you can choose the most efficient algorithm for your problem.  
""",
v_offset=-680)

# ╔═╡ ddfb06e8-04d4-47ac-aed1-396b744e95b9
aside(tip(md"In general, one would generally try to initialize an iterative algorithm with a reasonable guess for the initial values of the parameters.  Here, we've purposefully set the values to be very far from the minimum for illustration purposes. 😁"))


# ╔═╡ 6d7e2378-afe7-469c-a4e7-3522a0c76f76
aside(md"""
!!! tip "Pluto notebooks are reactive"
    All the code in the computational notebooks used for lab tutorials are editable.  We try to provide specific instructions for the main questions.  But you're always able to explore more on your own.  
    Pluto notebooks are *reactive* meaning that when you change any code cell, Pluto will figure out what other cells are effected and redo those calculations.  
    In contrast, with Jupyter notebooks, you'll have to manually choose which code cell(s) you want to rerun.  (This is a common source of bugs in Jupyter notebooks.)
""", v_offset=-120)

# ╔═╡ df992e3a-8d0c-44d7-824a-2a1d3f022f06
aside(
    md"""
    !!! tip "Geometric Interpretation"
    	The weight matrix ``\mathbf{W}`` changes the linear algebra for the solution in a simple way. The new solution is given by:
    	```math
    		(\mathbf{X}^\intercal \mathbf{W} \mathbf{X})\mathbf{\hat{\beta}}
    		=
    		\mathbf{X}^\intercal \mathbf{W} \mathbf{y}
    	```
    	If we examine the formula, the idea of projecting the best regression estimate ``\mathbf{\beta}`` onto the column space of ``\mathbf{X}`` still holds. What has changed, is what it means to "project a vector" since the space is now imbued with a weighted inner product
    	```math
    		\left\langle \mathbf{x},\,\mathbf{y} \right\rangle = \mathbf{x}^\intercal\mathbf{W}\mathbf{y}.
    	```
    	Geometrically, this means our notion of "distance between two points" has changed!
    """,
    v_offset=-550
)

# ╔═╡ 29dc1f06-64cd-4dd4-8ac2-6fdac00bcf3f
begin  # For embedded html into markdown
	nbsp = html"&nbsp;"
	br = html"<br />"
end;

# ╔═╡ c0df473f-6264-4a1a-b83a-e360c9cafe9c
md"""
# Lab 2: Revisiting Linear Regression $br From a Machine Learning Perspective
#### [Penn State Astroinformatics Summer School 2022](https://sites.psu.edu/astrostatistics/astroinfo-su22-program/)
#### Kadri Nizam & [Eric Ford](https://www.personal.psu.edu/ebf11)
"""

# ╔═╡ ddc378dd-537a-4247-a2ea-4a5be1924289
md"""
Viewing angles: $nbsp  Azimuth $(@bind camera1 Slider(0:1:90, default=83))
$nbsp  $nbsp
Altitude: $(@bind camera2 Slider(0:1:90, default=20))
"""

# ╔═╡ e2518430-ff7a-4142-9334-f5278be2b2f6
# 0a4ea37a-cbf1-4647-b206-79a5c281f2f6
let
	mesh_size = 31
	θ₀ = range(-150, stop=150, length=mesh_size)
	θ₁ = range(-25, stop=25, length=mesh_size)

	J_mesh = zeros(mesh_size, mesh_size)

	for (idx₀, t₀) ∈ enumerate(θ₀), (idx₁, t₁) ∈ enumerate(θ₁),
		J_mesh[ idx₀,idx₁ ] = loss_function(y, f(X, [t₀, t₁]))  
	end
	plt = plot(xlabel="θ₀", ylabel="θ₁", zlabel="J", zlims=(0,2e5), camera=(camera1, camera2))
	surface!(plt,θ₀, θ₁, J_mesh', color=:viridis )

	plot!(plt,
		view(θ_gd_history,1:n, 1),
		view(θ_gd_history,1:n, 2),
		view(j_gd_history,1:n),
		linestyle=:dash, color=:white,
		markershape=:circle, markerstrokewidth=0.1,
		legend=false
	)

end

# ╔═╡ 9b65b8b8-ac9a-4d28-b87b-8a75cdfbaee2
let

    q1 = Foldable(
        "Why is the loss function on GD worse than OLS?",
        md"""OLS is an exact methodology whilst GD is an iterative scheme. GD will traverse down the gradient to a local/global minima but can miss the **exact** minima depending on the geometry of the objective function, step size, etc. $(br)"""
    )

    q2 = Foldable(
        "What's the point of using GD if OLS yields the best estimate?",
        md"""The examples in this lab are small for practical reasons. Can you imagine constructing a design matrix for large, multi-dimensional datasets? You will be certain to run out of computer memory before you can solve the problem. GD does not suffer from this. $(br) The gist of it is use OLS for small to moderately large datasets; GD otherwise."""
    )

    md"""
    Some questions to ponder:
    1. $(q1)
    2. $(q2)
    """
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
CSV = "~0.10.4"
DataFrames = "~1.3.4"
Distributions = "~0.25.59"
HypertextLiteral = "~0.9.4"
LaTeXStrings = "~1.3.0"
Optim = "~1.7.0"
Plots = "~1.29.0"
PlutoTeachingTools = "~0.1.4"
PlutoUI = "~0.7.39"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.0"
manifest_format = "2.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "8a9c02f9d323d4dd8a47245abb106355bf7b45e6"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "873fb188a4b9d76549b81465b1f75c82aaf59238"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.4"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9489214b993cd42d17f44c36e359bf6a7c919abf"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "1e315e3f4b0b7ce40feded39c73049692126cf53"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.3"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "7297381ccb5df764549818d9a7d57e45f1057d30"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.18.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "0f4e115f6f34bbe43c19751c90a38b2f380637b9"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.3"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "87e84b2293559571802f97dd9c94cfd6be52c5e5"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.44.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "daa21eb85147f72e41f6352a57fccea377e310a9"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.4"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "28d605d9a0ac17118fe2c5e9ce0fbb76c3ceb120"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.11.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "d29d8faf1a0ca59167f04edd4d0eb971a6ae009c"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.59"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "129b104185df66e408edd6625d480b7f9e9823a0"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.18"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[deps.FiniteDiff]]
deps = ["ArrayInterfaceCore", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "4fc79c0f63ddfdcdc623a8ce36623346a7ce9ae4"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.12.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "2f18915445b248731ec5db4e4a17e451020bf21e"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.30"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "b316fd18f5bc025fedcb708332aecb3e13b9b453"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.3"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1e5490a51b4e9d07e8b04836f6008f46b48aaa87"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.3+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "SpecialFunctions", "Test"]
git-tree-sha1 = "cb7099a0109939f16a4d3b572ba8396b1f6c7c31"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.10"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "61feba885fac3a407465726d0c330b3055df897f"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "336cc738f03e069ef2cac55a104eb823455dca75"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.4"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "46a39b9c58749eefb5f2dc1178cb8fab5332b1ab"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.15"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "f27132e551e959b3667d8c93eae90973225032dd"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.1.1"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "09e4b894ce6a976c354a69041a04748180d43637"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.15"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "50310f934e55e5ca3912fb941dec199b49ca9b68"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.2"

[[deps.NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "7a28efc8e34d5df89fc87343318b0a8add2c4021"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "027185efff6be268abbaf30cfd53ca9b59e3c857"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.10"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "1285416549ccfcdf0c50d4997a94331e88d68413"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "d457f881ea56bbfa18222642de51e0abf67b9027"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.29.0"

[[deps.PlutoTeachingTools]]
deps = ["LaTeXStrings", "Markdown", "PlutoUI", "Random"]
git-tree-sha1 = "e2b63ee022e0b20f43fcd15cda3a9047f449e3b4"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.1.4"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "6a2f7d70512d205ca8c7ee31bfa9f142fe74310c"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.12"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "bc40f042cfcc56230f781d92db71f0e21496dffd"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.5"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "cd56bf18ed715e8b09f06ef8c6b781e6cdc49911"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c82aaa13b44ea00134f8c9c89819477bd3986ecd"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.3.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "9abba8f8fb8458e9adf07c8a2377a070674a24f1"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.8"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─c0df473f-6264-4a1a-b83a-e360c9cafe9c
# ╟─38ff28bd-6406-436c-ac0d-faf125febf5b
# ╟─49d355c8-0894-4a59-b16a-05e79f90d493
# ╟─bd4bb1f9-00d4-418a-bd2a-04bb0ba07c92
# ╟─3ce217e6-d87b-4767-9abb-3cd9ad365444
# ╟─d4874f23-4ad4-4cea-9893-f12aff99b878
# ╟─c6ca7cbc-0809-4cd2-a365-0f8fc16e3a3e
# ╟─0176e0f9-36a6-4c9e-8221-10f6e3a74f19
# ╟─5e971191-3353-428c-b26b-dfc0bcbf1c63
# ╟─a61518a4-6e42-4d8a-af8d-4d2a9605eef1
# ╟─ee83d05b-4a46-4f62-a0eb-1a70b5f4be8a
# ╟─d9be4bad-3292-4d71-a066-fce7dafb8865
# ╟─420a9b91-8672-4c73-8a34-9e3f3a9b9cb5
# ╟─82e7ca08-4149-4b0e-97bf-77f421d84acd
# ╟─87c75fa1-0a62-4e83-bc8c-db0b6a9befba
# ╟─e6c536ef-7202-44b6-aacf-020c152f74de
# ╠═4882f593-d707-46ff-8966-cb7813baad87
# ╟─c511d6b3-96d2-44a8-933e-a01065dbf7bc
# ╠═0d1c5003-090d-4f96-8683-f3d6aeb16e62
# ╟─b198560f-e55c-473a-bc36-c2f5e2a0fbfc
# ╟─5cade76c-ebcf-4cbd-9707-108799452cf3
# ╟─c0edc83e-2cfa-4542-8f08-08a35cd4a178
# ╟─704d5b25-cd7e-4486-9ec3-8a8030146ba5
# ╟─ae885e79-d922-4017-b150-554bd7762c42
# ╠═7eeebedc-a056-4577-b625-d4fe71798985
# ╟─964083ba-cdc1-42b0-a0fa-aa1de8ccee26
# ╟─e9a8d694-dbf7-49dc-ae75-fdcb67d93a7f
# ╟─1b607612-80be-453a-970d-cc449bce6c64
# ╟─c54763c1-d104-443c-895f-5729ea416ab0
# ╟─cea9a7b8-e7d2-46e1-92d3-1dce48012693
# ╟─817ef46e-4706-4743-854c-9bdb64df5243
# ╟─1e1f47cc-6b61-42f8-aebe-8a78779bd9e3
# ╟─0f5eb443-2884-4f0e-80b1-6bd0f10356aa
# ╟─8a48af4d-2dd7-43cc-a41b-2be7311874ce
# ╟─ddfb06e8-04d4-47ac-aed1-396b744e95b9
# ╟─d9617fa2-c49c-4642-b1d7-7fc06faaef60
# ╠═96fd2bed-43f6-4505-9a56-c6ed0f9dc88d
# ╠═0f7bf676-15c9-488b-87fa-ae8c9cfbf5dd
# ╟─c70c6482-dabb-49c4-81ee-1f9dbae43fc9
# ╠═a378cfdd-13fd-4bf9-ab5a-3d87e14413a2
# ╟─30b6cda7-39ba-4f65-a00f-1f5ba3736c0e
# ╟─0bc528e8-c596-43a2-81ac-47a52d51a1e2
# ╟─7752f520-80c2-4822-9acb-19afa24b13e5
# ╟─d2b4fe71-c159-46d3-8aaa-481598e7b75f
# ╟─55a64cf8-73f8-42b9-9b84-05ac6f956024
# ╟─71713d50-cac7-430c-9973-acdbe97c6712
# ╟─6d7e2378-afe7-469c-a4e7-3522a0c76f76
# ╟─dc29a574-cc88-4ed9-b0a7-f29d2107bc3d
# ╟─a941ed2c-3a3d-4b87-8f43-3276b7fd6ef7
# ╟─e2518430-ff7a-4142-9334-f5278be2b2f6
# ╟─ddc378dd-537a-4247-a2ea-4a5be1924289
# ╟─7870356f-f6ce-40be-9c69-6ff7459ded02
# ╟─a022804a-9d68-409a-8a8f-abfa1cc61a10
# ╟─fa353f9c-2adb-473e-80db-f5664b5e1008
# ╟─501a28d0-3478-4564-a9f8-22deaec158c1
# ╟─3609cae3-263a-454a-8929-ea79e2021710
# ╟─9b65b8b8-ac9a-4d28-b87b-8a75cdfbaee2
# ╟─676168f5-a61a-4b4d-a8c2-56b0b5cfc3d5
# ╟─135bcbd6-f050-49d6-b12c-512d4a534417
# ╟─c24ae519-cdd6-4544-943e-ee6018da0810
# ╟─df992e3a-8d0c-44d7-824a-2a1d3f022f06
# ╟─75311f29-da64-44fe-9623-5e92da348f13
# ╟─03157430-e8a8-4daf-9689-802ccd1195f8
# ╟─422aa4a5-5642-4e64-a20f-ec1727efcafc
# ╟─dbbea6dd-2f02-420c-9ee6-9dc5bd9ebb30
# ╠═356000c0-882e-11ec-0fd0-1932daca92e7
# ╠═8806e08e-0af8-48dc-a580-314a6a0cf917
# ╠═244234eb-eb61-4333-817f-f7cec47fd111
# ╟─d556b1d3-ef78-45a3-9d60-63645d1ceb48
# ╠═29dc1f06-64cd-4dd4-8ac2-6fdac00bcf3f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
