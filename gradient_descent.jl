### A Pluto.jl notebook ###
# v0.19.5

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

# ╔═╡ a79db603-a35d-4807-bb70-0d493c743ca6
begin  # Import packages to be used by notebook
	#using Pkg
	#Pkg.activate(".")
	using Optim
	using ForwardDiff

	using Plots, LaTeXStrings

	using PlutoUI
	using Markdown, HypertextLiteral
end

# ╔═╡ d40cc238-e015-4ee0-ba2f-451f077218ca
md"""
# Lab 1:  Introduction to Optimization Algorithms
#### [Penn State Astroinformatics Summer School 2022](https://sites.psu.edu/astrostatistics/astroinfo-su22-program/)
#### Kadri Nizam & [Eric Ford](https://www.personal.psu.edu/ebf11)
"""

# ╔═╡ 7503ff50-1706-4f2e-8b07-86d06051fd09
md"""
# Function optimization

In research, we frequently encounter tasks that can be formulated as an optimization problem, i.e., finding the values that minimize (or maximize) a function.  Some astrophysical examples would be:
- fitting light curves of exoplanets, finding a star formation rate that leads to an observed stellar population,
- reconstructing the geometry of an AGN to match observed broad spectral line profiles,
- finding stellar atmospheric parameters that can reproduce observed spectral lines,
- building models of circumstellar disk structure that can match ALMA observations,
- etc.
As we learn more about modern statistical methods and Machine Learning (ML), we will see that a recurring theme when implementing an ML model is optimizing (potentially very complex) functions.  
"""

# ╔═╡ d1dea7ba-306e-4b90-9e9f-9eaddc13c1f5
md"""
!!! question "Your turn!"
    Can you think of a procedure in your research that could be reformed into an optimization problem?
"""

# ╔═╡ f6906ac2-eb06-464c-8e99-38f86e264595
md"""
Let's get started!

#### How can we minimize or maximize a function?
There are many algorithms for finding the minimum (or maximum) of a function.  
In this lab, we will explore one foundational (and surprisingly useful) algorithm for function optimization, [__Gradient Descent__](https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html) (sometimes abbreviated GD).
This method was proposed before modern computers existed.  Later we will see how to use more sophisticated algorithms that are often implemented in optimization libraries.
While there are important differences in the details, most of the improved methods use similar ideas.
"""

# ╔═╡ 10113bbe-b86d-4dc5-b2d4-e83085098bba
md"""
#### Optimizing the objective function
"""

# ╔═╡ 89c4616a-84ee-4df3-b304-3acb1e6ccb6e
md"""
The goal of an optimization problem is usually to find the global mode (e.g., best solution).  There are mathematical proofs about when algorithms such as gradient descent will converge to the global mode.  Generally, you are only guaranteed to find the global minimum (or maximum) if the __objective function__ is [__convex__](https://en.wikipedia.org/wiki/Convex_function).
"""

# ╔═╡ d5fc4a87-c311-47f6-bb8b-f6b984991fc4
md"""
In many real world problems, you can't be sure that your objective function is convex.  Even if your objective function is __non-convex__, you can still apply algorithms such as gradient descent to find the lowest possible value within a neighborhood, i.e. a local minimum or maximum (e.g., best solution within a neighborhood).  In these cases, the result will depend on how you initialize the algorithm.  
"""

# ╔═╡ 4a6f4998-2e25-485b-9dcd-26ea63b3eead
md"""
## Example 1: Convex 1-d Function
To build our intuition for how gradient descent works, let us begin with a simple quadratic function. Consider:

```math
	f(x) = 3x^2 + x - 2, \quad x \in [-10, 10]
```
"""

# ╔═╡ 10e281e5-8990-4c06-a4ea-f36054b134f8
md"In Julia, we can implement that easily with the following code"

# ╔═╡ 94dfadf9-aeb7-404c-980a-84b617cebda7
f(x) = 3x^2 + x - 2;

# ╔═╡ f47a762a-017a-4e0b-92a8-1687250aeb57
let
	x_plt = -10:0.1:10
	plot(x_plt, f.(x_plt), xlabel="x", ylabel="f(x)", color="black", legend=false)
end

# ╔═╡ 053b15d0-600d-411a-8224-6affc138214f
md"""
We can use the gradient descent algorithm to find the minimum by following an iterative algorithm to compute an improved guess for the minimum, ``x_{n+1}`` from the a given point ``x_n`` using the following steps:

1. Choose a random initial starting point ``x_o``.  
2. Compute ``\nabla f(x_i)``, the gradient of ``f(x)`` at the current position, ``x_i``
3. Scale the gradient with [_learning rate_](https://en.wikipedia.org/wiki/Learning_rate), ``\eta``
4. Take a step from ``x_i`` towards the minimum, i.e.,  set ``x_{i+1} = x_i - \eta \nabla f(x_i)`` .
5. Increase ``i`` and repeat 2 through 4 until a stopping criterion is met (e.g., step is smaller than a desired tolerance)

Mathematically,

```math
    x_1 = x_o
```
"""

# ╔═╡ b8b331b3-0e31-4909-94b1-3c5e4ebe2720
md"""
```math
	x_{n+1} = x_n  - \eta \nabla f(x_n), \quad n \in \mathbb{Z}^+
```
In our univariate example, the gradient of ``f(x)`` is just its derivative.

```math
	 \nabla f(x) = f_x(x) = \dfrac{df}{dx}(x) = 6x + 1
```
In this example, we'll implement fₓ manually:
"""

# ╔═╡ cd51ec28-c061-4b1b-bf9d-9e57f9afbddb
fₓ(x) = 6x + 1;

# ╔═╡ 3c84eb23-0c0c-4ed2-92c6-7d4c9abf508a
#	x_2 = x_1 - \eta f_x(x_1) = 8 - 0.1(49) = 3.1
md"""
Let's compute the first two iterations by hand.   

Let's say our initial random point is ``x_1 = 8`` and ``\eta = 0.1``.   
Then

``x_2 = x_1 - \eta f_x(x_1) = ``
 $(@bind response_x2 NumberField(-10:10,default=0))
"""

# ╔═╡ 6777d6fe-2235-4832-9455-a12a6ef1f947
# TODO Pretty up
if response_x2 == 0
	md""" Enter response above."""
elseif  response_x2 ≈ 3.1
	#=```math
	x_2 = x_1 - \eta f_x(x_1) = 8 - 0.1(49) = 3.1
	```=#
	md""" Correct.  
	The next iteration would be
	```math
	x_3 = x_2 - \eta f_x(x_2) = 3.1 - 0.1(19.6) = 1.14
	```
	"""
else
	md""" Check your work above."""
end

# ╔═╡ 90ec3591-d797-43af-a990-e8001a6d8e09
md"""
Let's examine how this translates to code.  
The key step is simply:
"""

# ╔═╡ 9bdd5e48-c0da-4fcc-936e-f1dac567d035
compute_step_gd(fₓ, xₒ, η) = η * fₓ(xₒ);

# ╔═╡ cc261c81-1180-43d5-a27c-20c7ea749cd3
md"""
We also need a wrapper function to initialize ``x_i``, repeatedly compute the step using the function above, and decide when we're happy and can stop improving our estimate of the minimum.  In this case, we also save the state at each step of the algorithm, so we can easily visualize how the algorithm is working below.
"""

# ╔═╡ c630a893-a6c7-4818-b99c-29fd5afaf85d
function univariate_gd(fₓ, xₒ, η; tol=1e-6, max_iter=500)
	history = zeros(max_iter)
	history[1] = x = xₒ                        # Log initial guess
	for i in 2:max_iter
		step = compute_step_gd(fₓ,x,η)         # compute step to take
		x = x - step             			   # take step
		history[i] = x  
		if abs(step) < tol                     # step was smaller than tolerance
			resize!(history,i)				   # keep only iterations taken
			break                              # so we can quit early
		end
	end
	return history  
end;		

# ╔═╡ 4447b740-0e17-4f45-9fc3-e4f9514e28f9
md"""
We have replotted ``f(x)`` below with our initial point ``x_1`` indicated.
We also list the learning rate, initial guess, difference between the current iteration and the true minimum, and ``\Delta f(x)``, the difference in the objective function evaluated at the location of the current iteration and the objective function evaluated at the true minimum.
"""

# ╔═╡ 47bef150-c19b-45d5-96d3-1a90f47ca582
md"""
### Learning Rate (``\eta``) for convex optimization
Below are two controls to adjust η, the learning rate, and the location of the starting point.  Try setting a smaller learning rate.  Predict how the trajectory will differ.  Then drag the Iteration slider above forward to see if your predictions are correct.  

What do you expect will happen if you set a larger learning rate?  Try it!
"""

# ╔═╡ 3d3792b2-3538-4ae6-8127-11ff6a9391c2
md"""
log η: $(@bind logη Slider(-1.5:0.2:-0.5, default=-1))
Starting point: $(@bind x_1 Slider(-10.0:0.2:10.0, default=8.0))
"""

# ╔═╡ 3a10ac44-a6b2-4996-ade2-7aa393a03e53
begin # logη and x_1 are set below
	η = 10^logη                             
	history = univariate_gd(fₓ, x_1, η, tol=1e-3)
end;

# ╔═╡ 17dea57e-f645-4467-9769-91d87f403771
md"""
Iteration: $(@bind n_plt Slider(1:length(history), default=1))

As you drag the slider from left to right, you'll reveal the value ``x_i`` for additional iterations of the gradient descent algorithm.  Where do you expect the  next two guesses to be?  

Drag the slider above to reveal the trajectory that gradient descent takes.  The initial guess will become blue, and the final estimate will be in red.  
"""

# ╔═╡ 0bbd734d-4e2a-4152-80fd-43bd8b42c13c
let
	x_plt = -10:0.1:10
	plot(x_plt, f.(x_plt), xlabel="x", ylabel="f(x)", color="black", legend=:none)
	plot!(history[1:n_plt], f.(history[1:n_plt]), markershape = :circle, color="grey")
	plot!(history[1:1], f.(history[1:1]), markershape = :circle, color="blue")
	plot!(history[n_plt:n_plt], f.(history[n_plt:n_plt]), markershape = :circle, color="red")
	true_x_min = -1/6
	Δx = history[n_plt]-true_x_min
	Δf = f(history[n_plt])-f(true_x_min)
	annotate!([(0,300,L"\eta = " * string(round(η,digits=3)))])
	annotate!([(0,250,latexstring("x_{" * string(n_plt) * "} = " * string(round(history[n_plt],digits=3)) ))])
	annotate!([(0,200,L"\Delta x = " * string(round(Δx,digits=4)))])
	annotate!([(0,150,latexstring("\\Delta f(x_{" * string(n_plt) * "} ) = " * string(round(Δf,digits=5))))])
end

# ╔═╡ d1881047-bd5c-4c4e-bb41-5ca36deac6f7
md"""
We mentioned that ``\eta`` scales the gradient effectively controlling how big steps are taken in the direction of the gradient at each iteration. This value influences the performance of your model (potentially strongly).
- Too small of a learning rate and it will take many iterations to optimize the objective function.  For a simple problem like this, that may not be a big deal.  But if the function you are attempting to optimize is more complex, then each model evaluation may be quite expensive.  Taking too many small steps may make it impractical to converge to the minimum in a reasonable amount of time.
- Too large of a learning rate will result in overshooting the minimum.  For a convex function like this one, the consequence will be slow convergence.  But for non-convex functions, the algorithm might meander to a different minimum or even worse diverge!

How do we pick the best ``\eta`` value? There are heuristics and libraries that suggested values, but it is a good idea to get a feel of how ``\eta`` changes the behavior of the gradient descent algorithm. We'll explore this more in the next example.

"""

# ╔═╡ 2415ffde-93d2-4ae8-9724-a8c703dbd862
md"""
## Example 2:  Non-convex 1-d Function
"""

# ╔═╡ 09f20cb7-a1f6-4b89-a121-8e4b78e8e5ca
md"""Below is a quartic function and its derivative."""

# ╔═╡ 843b960f-1198-48d7-b18f-a046b0f56711
begin
	g(x) = 0.05x^4 - 3.5x^2 + 6x
	gₓ(x) = 0.2x^3 - 7x + 6
end;

# ╔═╡ 816cafa8-412e-4323-8d75-a526391c2a24
md"""
We will specify two different starting points (green and brown) for applying the gradient descent algorithm.
"""

# ╔═╡ ee0c466c-2765-4ff6-b6de-d7d8ed747036
x_left, x_right = -9.5, 9.5;

# ╔═╡ 103c05e7-0985-4522-b225-43352631aff6
md"""
The starting points are shown as large triangles and the stopping points are shown as large octagons.  The smaller points show the trajectory that the gradient descent algorithm took.  As you may have already intuited by now, GD is the process of tracing the gradient to a minima akin to always following the steepest path down a hill to the valley for dinner after a long day of hiking. 🍕

### Learning rate for non-convex objective function
What do you predict will happen as you increase the learning rate, ``\eta``?

Move the slider around to see what changing ``\eta`` does and test your prediction.

"""

# ╔═╡ 8678297f-b6cb-43aa-b278-e253fc9767cc

md"``\eta``: $(@bind η_g Slider(0.005:0.005:0.18))"

# ╔═╡ acbe741d-a695-422a-a6c5-21377c5dbdec
begin
	h_g_l = univariate_gd(gₓ, x_left, η_g)
	h_g_r = univariate_gd(gₓ, x_right, η_g)
end;

# ╔═╡ 76cf7e14-ef2f-401d-a12d-f42e1a6af8ea
begin
	x_plt = -10.0:0.1:10.0
	plot(x_plt, g.(x_plt), color="black", legend=false)

	min_g_l = last(h_g_l)
	min_g_r = last(h_g_r)
	# Starting points
	scatter!([x_left], [g.(x_left)], markershape=:utriangle, markersize=8, color="forestgreen")
	scatter!([x_right], [g.(x_right)], markershape=:utriangle, markersize=8, color="chocolate2")
	# Stopping points
	scatter!([h_g_l[end]], [g.(h_g_l[end])], markershape=:octagon, markersize=8, color="forestgreen")
	scatter!([h_g_r[end]], [g.(h_g_r[end])], markershape=:octagon, markersize=8, color="chocolate2")
	# Trajectories
	plot!(h_g_l, g.(h_g_l), markershape=:circle, markersize=3, color="forestgreen")
	plot!(h_g_r, g.(h_g_r), markershape=:utriangle, markersize=3, markeredgesize=0, color="chocolate2")

	xlabel!("x")
	ylabel!("g(x)")

	xlims!(-11, 11)
	ylims!(-110, 200)
end

# ╔═╡ 0d85e4dc-4251-473b-9029-d6ade20c4fcc
md"""
As you move the slider to the right, notice that the first step becomes larger.  At first, this reduces the number of iterations before convergence.  However, if you keep increasing ``\eta`` further, it will overshoot and cause the algorithm to convergence at different minimum.

If we keep going to even larger ``\eta``'s, then we see that the hexagon marked lines have trouble finding the minima and eventually both solution fails to converge (the brown one even diverges to infinity!).
"""

# ╔═╡ 0c157ab2-05b8-4ad6-ac0c-a0373baf5bd6
md"""
!!! tip "Pro tip"
    In real world applications, **most** optimization problems involve functions that are non-convex.  Therefore, it is important to realize that the solution returned by an iterative algorithm may not necessarily be the global minimum.
    For a univariate problem, it's often practical to scan the domain of your function for a global minima.  However, this gets prohibitively expense in higher dimensions - imagine manually figuring this out for a 50-dimensional multivariate function!
"""

# ╔═╡ 7b791afc-045f-4b2b-8ad3-abf5c3a5afb9
md"""
## Gradient Descent in Higher Dimensions

One of the great things about the gradient descent algorithm is that the idea of following the steepest path down a hill generalizes to higher dimensional cases.

Now it's your turn.  We've provided a shell function, `my_multivariate_GD` below.  But it's missing two critical steps.  Your task is to implement the logic for multivariate gradient descent in the following cell by replacing the code for `step` and updating `x` on lines 8 and 9.
"""

# ╔═╡ ed2581f6-21c3-465f-8578-4c6782fc929e
function my_multivariate_GD(∇f::Function, xₒ::Vector, η::Real;
					 		tol::Real=1e-6, max_iter::Integer=500)
	history = zeros(length(xₒ), max_iter) # 2-d array to store history
	history[:,1] .= xₒ                    # Log initial state
	x = xₒ                                # Initialize x
	for iter in 2:max_iter
		# YOUR TASK:  Replace the following two lines
		step = zeros(length(xₒ))   
		x = x
		history[:,iter] .= x             # Log current state
		if sqrt(sum(step.^2)) < tol      # Check if step was smaller than tolerance
			history = history[:,1:iter]  # Trim history to number of itterations used
			break                        # We can exit early
		end
	end
	return history
end

# ╔═╡ bade00b7-4e71-4375-8bc3-64e5985e35e4
md"""
Click here to skip ahead: $(@bind skip_multivariate CheckBox())
"""

# ╔═╡ fc3c3227-5680-43cd-9779-919c90690f1c
md"""
### Convex function
Now, we'll demonstrate our multi-dimensional gradient descent on a simple convex function, `f_multi`.
"""

# ╔═╡ c2dd09df-a967-46ac-9c8b-c350b2cea7cd
begin
	f_multi(x) = 0.5x[1]^2 .+ x[2]^2
	∇f_multi(x) = [x[1], 2*x[2]]
end;

# ╔═╡ 5ca5dcff-6e19-4926-96b8-ed63d443a042
md"For the visualization below, we'll use the following initial conditions and learning rate."

# ╔═╡ c213ff64-8eba-4e00-acc8-dcfdd1a623a7
begin
	x_start_2d = [38, -25]
	η_2d = 0.25
end;

# ╔═╡ 1c2438b5-2d85-4753-aef6-740f583fe814
md"""The large triangle marker is the starting point.  Each circle shows another iteration.  The algorithm manages to find the minima quite easily since we have a convex function.

### Non-convex function
Now we'll try again, but using a more complicated objective function that is not convex.
"""


# ╔═╡ aece4898-9a1e-43ee-80a8-6956542d1719
g_multi(x) = 16sin(0.1x[1]) + 8cos(0.15x[2]) + 0.02x[1]*x[2] + 0.1x[1] + 0.05x[2]

# ╔═╡ b943c9df-ba25-4bf5-baf5-6836499645a4
md"""
In this case, we _could_ calculate the gradient ourselves.  But as functions become more complex, that will become more and more time consuming and error prone.  Fortunately, there are now many packages to compute gradients _automatically_.  
Here's we'll use the `ForwardDiff` package to compute the gradient of `g_multi(x)` for us.
"""

# ╔═╡ 2c4af529-1e32-4b7f-be00-9e8643e45650
begin
	∇g_multi(x) = ForwardDiff.gradient(g_multi,x)
	# We could have written out
	∇g_multi_by_hand(x) = [1.6cos(0.1x[1]) + 0.02x[2] + 0.1, -1.2sin(0.15x[2]) + 0.02x[1] + 0.05]
	# But why risk it?
end;

# ╔═╡ 4d9a6c2e-1b1c-43d8-9041-3f2779a793ea
md"""
We can check that the gradients computed by hand and automatically agree.
"""

# ╔═╡ b573b310-c967-422c-81c9-037105c9b7fb
md"Choose the starting point and learning rate with the sliders below."

# ╔═╡ f98d868f-89d5-4b47-8b30-32226275508b
md"""
x = $(@bind x_g_2d Slider(-13π:0.5:13π)) ``\quad`` y = $(@bind y_g_2d Slider(-13π:0.5:13π)) ``\quad`` η value = $(@bind η_g_2d Slider(0.5:0.5:8, default=1))
"""

# ╔═╡ 198f7007-dbb4-4a98-9a24-890862888706
∇g_multi([x_g_2d, y_g_2d]) ≈ ∇g_multi_by_hand([x_g_2d, y_g_2d])

# ╔═╡ 3973978f-5a12-4cae-ba33-e48e5e8daf66
begin	# Make contour plot
	g_range = -20π:0.25:20π
	z_g = [g_multi([x,y]) for y in g_range, x in g_range]
	plt_2d_nc = contour(g_range, g_range, z_g)
	xlims!(plt_2d_nc,-20π, 20π)
	ylims!(plt_2d_nc,-20π, 20π)
end;

# ╔═╡ d3849cc2-f0fd-49b4-aa9c-12fedfb0600a
md"""
Not so simple anymore, eh? The contour plot above has many local minima and saddle points. Depending on your starting condition, you might end up in one of the numerous local minima or at a saddle point. This is how you can get some varying results at different iterations when optimizing a function with machine learning.
"""

# ╔═╡ fad5bfc8-1b5e-4ec9-92c4-84f499dfa8ca
md"""
!!! question "Check your understanding"
    If you look carefully, you might notice that the step size gets smaller as we
    converge to a minima. Why does this happen even when ``\eta`` remains constant?
"""

# ╔═╡ 1a24d1cf-f8f6-4264-9dc2-d0026dedc752
md"""
## Optimization Libraries
While the standard gradient descent algorithm can be easily implemented by hand, there are several variations.  Conjugate gradient descent can be more efficient for many simple problems.  For converging more rapidly, it can be useful to use "acceleration".  For high dimensional problems, it can be helpful to add "momentum".  For maximizing the chance of convergence, it's useful to make use of ["Nesterov acceleration"](https://jlmelville.github.io/mize/nesterov.html).  For example, when fitting models to large data sets, "stochastic" and "mini-batch" gradient descent can be quite useful.  With so many choices, you may want to try a few different algorithms and compare their performance.  It would be nice if you didn't have to implement each yourself.  Fortunately, most modern high-level programming langauges have packages or libraries that implement gradient descent and several variants.  

For Julia users, [Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/) is a good starting point.  In addition to gradient descent, it provide conjugate gradient descent, [Nelder-Mead](https://julianlsolvers.github.io/Optim.jl/stable/#algo/nelder_mead/) for problems where computing the gradient is impractical and [BFGS](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm) (and its lower memory version L-BFGS) that are very often good choices when one can compute the gradient.  For high-dimensional problems, the [Flux.jl](https://fluxml.ai/Flux.jl/stable/training/optimisers/) package provides several optimization algorithms more commonly used with neural networks (which we'll discuss later in the week).  
"""

# ╔═╡ caf5e2ea-90e7-4bc1-b251-0108814a1bcc
md"""
!!! tip "Pro tip: Optimization algorithsm in other languages"
    Python users may be interested in [Keras](https://keras.io/), [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/).  Python packages can also be called from Julia via the [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) package.
"""


# ╔═╡ 787c28c5-ad8f-4156-b915-d81ee0f007d0
md"""
### [Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/#)
Let's go through a short example on how use the Optim library.
We will be using the same univariate and multivariate functions in our examples above for this. We've already imported the Optim package into this notebook.  If you were starting a fresh Julia session, then you'd do that via `using Optim`.
"""

# ╔═╡ f97dd99b-ef70-43bd-b81e-a8b75a701728
md"""
For comparison's sake, we will use the function and gradient from one of our examples above.
"""

# ╔═╡ 65b8c072-3f08-44eb-9ea1-b18bbf0d1f9c
f_optim(x) = 0.5x[1]^2 + x[2]^2;

# ╔═╡ 1ca00c51-2005-4b17-a508-79b86344a477
md"""
Next, we set the initial coordinates.
"""

# ╔═╡ 07fae632-64a4-4913-bfa8-bfb6b81fabde
start_optim_1 = [10.0, 10.0];

# ╔═╡ 1f0d94fb-240b-43a1-b60d-6f6dd14bb66b
md"""We'll call the `optimize` function provided by `Optim`, passing the objective function, the starting point and a variable specifying the optimization algorithm to be used.  We'll start with gradient descent.
"""

# ╔═╡ 529f442f-182d-45f8-840c-690196ddb52a
optimize(f_optim, start_optim_1, GradientDescent())

# ╔═╡ 8767a160-e521-4074-955d-d86d56184921
md"""
We can easily compare the number of function and gradient evaluations required using different optimization algorithms.  For example, [Conjugate Gradient Descent](https://en.wikipedia.org/wiki/Conjugate_gradient_method) or [BFGS](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm).
"""

# ╔═╡ 6e793c7f-b4ed-4c8c-ae3c-1a424b68b350
optimize(f_optim, start_optim_1, ConjugateGradient())

# ╔═╡ 652f3b91-d61c-425e-b322-f326ea6893e5
md"""
How did the number of objective function evaluations and gradient evaluations compare for the standard gradient descent and conjugate gradient descent?  
"""

# ╔═╡ a33d53fe-7430-41f5-9808-d86df9dffbeb
md"""
### Automatic differentiation
In the examples above, we did not provide a function to compute the gradient above, so Optim attempted to approximate the gradient by using central finite differencing.  While convenient, it is important to remember that methods like finite differences are approximations and subject to numerical error!  

Alternatively, you can specify that it should use automatic differentiation by passing the optional parameter `autodiff = :forward`.  (There are also algorithms for "reverse-mode automatic differentiation", but those aren't fully integrated into Optim.jl yet.)
"""

# ╔═╡ 698433ba-9672-422c-abbd-522c6d5147f9
optimize(f_optim, start_optim_1, ConjugateGradient(), autodiff=:forward)

# ╔═╡ 8b093c35-67ae-4052-9f03-80de0fe4c2c7
md"""
How did the accuracy of the solutions compare when using numerical differences and automatic differentiation?
"""

# ╔═╡ e5515428-514f-425b-96cb-14b9b6a5c8cd
md"## Optional Advanced Topics"

# ╔═╡ 8d9bd819-7c49-4118-8c11-d67995aea5ee
md"""
### BFGS algorithm
Another workhorse algorithm that is much more complex to implement yourself is known as [`BFGS`](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm).  There's also a version that uses less memory, known as `LBFGS`.
"""

# ╔═╡ 1b546127-6f1b-4267-b997-80ffc4d6afc5
optimize(f_optim, start_optim_1, BFGS(), autodiff=:forward)

# ╔═╡ 8442db0c-eb1c-4fee-b242-2810458f09f3
optimize(f_optim, start_optim_1, LBFGS(), autodiff=:forward)

# ╔═╡ 9db36156-fccd-4402-9bb1-50f1c2fe9bee
md"""
How does their performance compare to the gradient descent and conjugate gradient descent algorithms?
"""

# ╔═╡ b7d94cea-068d-49a4-bbc7-993c5599f493
md"""
### Improving computational efficiency
In some cases, you may prefer to provide both the objective function and your own function to compute the gradient of the objective function more efficiently.  For example, by pre-allocating memory and repeatedly writing into the same memory, you can minimize unnecessary memory allocations and improve the performance.  

In order for your gradient function to work with Optim, it takes two parameters, the first called `storage`  is a pre-allocated array where the result will be written.
The second argument is the location to evaluate the gradient.
Since the`storage` parameter will be mutated (i.e., altered) inside the function, we will append an exclamation mark to the function name to remind ourself (and anyone else who reads the code) of this.
"""

# ╔═╡ b5520943-9698-4eb1-9900-79aa753dc431
function ∇f_optim!(storage, x)
	# If we wanted to use automatic differentiation
	storage .= ForwardDiff.gradient(f_optim,x)
	# If we wanted to compute the gradient by hand
	storage[1] = x[1]
	storage[2] = 2x[2]
	return storage
end

# ╔═╡ f97de562-fdc2-4bd8-abde-c212d947a036
optimize(f_optim, ∇f_optim!, start_optim_1, BFGS())

# ╔═╡ 229f86e4-9059-419d-8be0-ab4ee3b2f6b4
md"""
### Additional reading
Vanilla gradient descent is a good starting point to understand some of the more advanced variants commonly used in machine learning. If you're interested in learning more about optimization algorithms, you're invited to read (or skim) [An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/), and particularly the sections on [Stochastic Gradient Descent](https://ruder.io/optimizing-gradient-descent/index.html#stochasticgradientdescent) and [Mini-Batch Gradient Descent](https://ruder.io/optimizing-gradient-descent/index.html#minibatchgradientdescent).
"""

# ╔═╡ e8500d66-f1ab-43f2-beee-88e6b3f8f31d
md"""
# Reference Implementations
Want to see the reference version of the functions we asked you to write?  
Their code is shown below."""

# ╔═╡ 56f8dec7-adf5-4012-87ad-5257f6bef5c1
function reference_multivariate_GD(∇f::Function, xₒ::Vector, η::Real;
					tol::Real=1e-6, max_iter::Integer=500)
	history = zeros(length(xₒ), max_iter)
	history[:,1] .= xₒ
	x = xₒ
	step = zeros(length(xₒ))
	for iter in 2:max_iter
		step .=  η * ∇f(x)				
		x = x - step
		history[:,iter] .= x
		if sqrt(sum(step.^2)) < tol
			return history[:,1:iter]
		end
	end
	return history
end;

# ╔═╡ 24b69a6f-9854-41b6-ac48-ddd5812d63e6
# TODO ADD tests, prettify
if my_multivariate_GD(f_multi, x_start_2d, 0.1 ) == reference_multivariate_GD(f_multi, x_start_2d, 0.1 )
	md""" Good job!"""
else
	md""" Hmm, your function (`my_multivariate_GD`) isn't returning the same values as the reference function.  Either try again or click the box below to continue using the reference implementation (which you can view near the bottom of the notebook).
	"""
end

# ╔═╡ 2efb8cd0-159a-4d78-97f7-5fb3bc2280f8
# Chooses which function to use based on the checkbox above.
multivariate_GD = skip_multivariate ? reference_multivariate_GD : my_multivariate_GD;

# ╔═╡ a10d0c34-74ee-4a76-bcfe-102869cc67c7
begin
	history_2d = multivariate_GD(∇f_multi, x_start_2d, η_2d)
	num_iter_2d = size(history_2d,2)
end;

# ╔═╡ d5258020-0e95-45a2-8add-0ee9dcc54e46

md"""
You can see GD gradient descent action by moving the slider below.

Iteration = $(@bind n2 Slider(1:num_iter_2d))
"""

# ╔═╡ 089fd6a3-894e-46ce-b621-77442d5e8c74
begin

	axis_range = -40:0.5:40
	z = [f_multi([x, y]) for y in axis_range, x in axis_range]

	contour(axis_range, axis_range, z, legend=false)
	plot!(history_2d[1,1:n2], history_2d[2,1:n2], linestyle=:dash, markershape=:circle, color="red")
	scatter!(history_2d[1,1:1], history_2d[2,1:1], linestyle=:dash, 	markersize=6,markershape=:utriangle, color="red")
	xlabel!(L"x_1")
	ylabel!(L"x_2")
end

# ╔═╡ a98651fc-be42-46fd-9fc0-2982d1a52603
begin
	start_g = [x_g_2d, y_g_2d]
	h_g_2d = multivariate_GD(∇g_multi, start_g, η_g_2d)

	plot!(plt_2d_nc,h_g_2d[1,:], h_g_2d[2,:], markershape=:circle, color="red")
	scatter!([h_g_2d[1,1]], [h_g_2d[2,1]], markershape=:utriangle, markersize=8, color="red", legend = false)
end

# ╔═╡ 2f0459f9-e44e-448c-ac53-fb84f144e259
md"# Setup & Helper Code"

# ╔═╡ 7db41766-d8eb-404c-9cee-66c426c79559
TableOfContents()

# ╔═╡ 695ac71a-2bf9-4032-9db0-cff937787d6b
hint(;hint_title = "Hint", text = "") = Markdown.MD(Markdown.Admonition("hint", hint_title, [text]))

# ╔═╡ 8ae295aa-1bc6-42ea-a9b5-ca81deabf624
hint(hint_title = "Hint 1", text = md"The approach is similar to the univariate case except the gradient is now subracted from each dimension independently.  In Julia, you can multiple scalars, vectors and matrices using the ``*`` operator.")

# ╔═╡ 79df6f0b-e1b3-48a4-b315-4aec7a1aed77
hint(hint_title = "Hint:  How to type a ∇?", text = md"To get the ∇ symbol, type ''\\nabla'' then press the tab key.")

# ╔═╡ 86bb1bf1-eea8-49e1-b9f7-6f741dc9550c
hint(hint_title = "Hint:  What do '.=' and ':' mean?", text = md"In Julia, if we write, `a = b` then the symbol `a` would point to the same data as stored in `b`.   
Alternatively, we can write data into an existing array with two arrays `a .= b`.
 The colon (':') operator allows us to specify what range of array it self, allowing us to avoid to unnecessary copies of the whole array.  Therefore,  the syntax `history[:,1] .= xₒ` stores the contents of the vector `xₒ` into the first column of the 2-d array `history`.")

# ╔═╡ dc92ba3f-3195-4941-8066-b5e1151ba3f5
function aside(x)
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

		<aside class="plutoui-aside-wrapper">
		<div>
		$(x)
		</div>
		</aside>

		""")
end

# ╔═╡ e432cc65-694e-48fb-b7dc-1a1c44ac5144
aside(md"""
!!! terminology "Terminology: Objective Function"
	There are many names for the function to be optimized.  Each field (and subfield) has their preferred terminology.  Here, we'll refer to the function that we want to optimize the __objective function__.  You might also hear __loss function__, or __target function__, or a name specific to a particular problem at hand, e.g., a log likelihood or energy.  
""")

# ╔═╡ c2d926eb-10b5-4b18-9c0e-d15da5e26d2d
aside(md"""
!!! terminology "Terminology: Convexity"
	A function is convex if a line segment connecting any two points in the graph does not lie below the graph between the two points. In simpler terms, a function that is cupped (and ready to hold my midnight cuppa) is probably convex. ☕
""")

# ╔═╡ 28e06e61-48e4-4dda-a70b-f89c52b8bc11
aside(md"""
!!! terminology "Notation"
    The notation ``n \in \mathbb{Z}^+`` is just a fancy way of saying ``n`` is a positive integer.
""")

# ╔═╡ d28da7f2-82e0-499a-9892-57356fce50a9
aside(md"""
!!! tip
    Rather than using different variables to represent the different dimensions (``x``, ``y``), it is much more common to create a function that takes in vector inputs instead (``x_1``, ``x_2``).  This also has the advantage of allowing your function to be used for an arbitrary number of dimensions.

""")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
ForwardDiff = "~0.10.25"
HypertextLiteral = "~0.9.3"
LaTeXStrings = "~1.3.0"
Optim = "~1.6.0"
Plots = "~1.25.7"
PlutoUI = "~0.7.34"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
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

[[deps.ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "1bdcc02836402d104a46f7843b6e6730b1948264"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "4.0.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f9982ef575e19b0e5c7a98c6e75ee496c0f73a93"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.12.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "6b6f04f93710c71550ec7e16b650c1b9a612d0b6"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.16.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

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
git-tree-sha1 = "44c37b4636bc54afac5c574d2d02b625349d6582"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.41.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

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

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "84083a5136b6abf426174a58325ffd159dd6d94f"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.9.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ae13fcbc7ab8f16b0856729b050ef0c446aa3492"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.4+0"

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

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "6eae72e9943d8992d14359c32aed5f892bda1569"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.10.0"

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
git-tree-sha1 = "1bd6fc0c344fc0cbee1f42f8d2e7ec8253dda2d2"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.25"

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

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "4a740db447aae0fbeb3ee730de1afbb14ac798a1"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.63.1"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "aa22e1ee9e722f1da183eb33370df4c1aeb6c2cd"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.63.1+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

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

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

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
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

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
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

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
git-tree-sha1 = "e5718a00af0ab9756305a0392832c8952c7426c1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.6"

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
git-tree-sha1 = "648107615c15d4e09f7eca16307bc821c1f718d8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.13+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "916077e0f0f8966eb0dc98a5c39921fdb8f49eb4"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.6.0"

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

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0b5cfbb704034b5b4c1869e36634438a047df065"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "6f1b25e8ea06279b5689263cc538f51331d7ca17"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.1.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "7e4920a7d4323b8ffc3db184580598450bde8a8e"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.25.7"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8979e9802b4ac3d58c503a20f2824ad67f9074dd"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.34"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "2cf929d64681236a2e074ffafb8d568733d2e6af"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

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
git-tree-sha1 = "37c1631cb3cc36a535105e6d5557864c82cd8c2b"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.0"

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

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

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
git-tree-sha1 = "a4116accb1c84f0a8e1b9932d873654942b2364b"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.1"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "d4da8b728580709d736704764e55d6ef38cb7c87"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.5.3"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "a635a9333989a094bddc9f940c04c549cd66afcf"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.3.4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
git-tree-sha1 = "d88665adc9bcf45903013af0982e2fd05ae3d0a6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "51383f2d367eb3b444c961d485c565e4c0cf4ba0"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.14"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "d21f2c564b21a202f4677c0fba5b5ee431058544"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.4"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "bb1064c9a84c52e277f1096cf41434b675cd368b"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

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
git-tree-sha1 = "66d72dc6fcc86352f01676e8f0f698562e60510f"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.23.0+0"

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
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

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
# ╟─d40cc238-e015-4ee0-ba2f-451f077218ca
# ╟─7503ff50-1706-4f2e-8b07-86d06051fd09
# ╟─d1dea7ba-306e-4b90-9e9f-9eaddc13c1f5
# ╟─f6906ac2-eb06-464c-8e99-38f86e264595
# ╟─e432cc65-694e-48fb-b7dc-1a1c44ac5144
# ╟─10113bbe-b86d-4dc5-b2d4-e83085098bba
# ╟─89c4616a-84ee-4df3-b304-3acb1e6ccb6e
# ╟─d5fc4a87-c311-47f6-bb8b-f6b984991fc4
# ╟─c2d926eb-10b5-4b18-9c0e-d15da5e26d2d
# ╟─4a6f4998-2e25-485b-9dcd-26ea63b3eead
# ╟─f47a762a-017a-4e0b-92a8-1687250aeb57
# ╟─10e281e5-8990-4c06-a4ea-f36054b134f8
# ╠═94dfadf9-aeb7-404c-980a-84b617cebda7
# ╟─053b15d0-600d-411a-8224-6affc138214f
# ╟─28e06e61-48e4-4dda-a70b-f89c52b8bc11
# ╟─b8b331b3-0e31-4909-94b1-3c5e4ebe2720
# ╠═cd51ec28-c061-4b1b-bf9d-9e57f9afbddb
# ╟─3c84eb23-0c0c-4ed2-92c6-7d4c9abf508a
# ╠═6777d6fe-2235-4832-9455-a12a6ef1f947
# ╟─90ec3591-d797-43af-a990-e8001a6d8e09
# ╠═9bdd5e48-c0da-4fcc-936e-f1dac567d035
# ╟─cc261c81-1180-43d5-a27c-20c7ea749cd3
# ╠═c630a893-a6c7-4818-b99c-29fd5afaf85d
# ╟─3a10ac44-a6b2-4996-ade2-7aa393a03e53
# ╟─4447b740-0e17-4f45-9fc3-e4f9514e28f9
# ╟─0bbd734d-4e2a-4152-80fd-43bd8b42c13c
# ╟─17dea57e-f645-4467-9769-91d87f403771
# ╟─47bef150-c19b-45d5-96d3-1a90f47ca582
# ╟─3d3792b2-3538-4ae6-8127-11ff6a9391c2
# ╟─d1881047-bd5c-4c4e-bb41-5ca36deac6f7
# ╟─2415ffde-93d2-4ae8-9724-a8c703dbd862
# ╟─09f20cb7-a1f6-4b89-a121-8e4b78e8e5ca
# ╠═843b960f-1198-48d7-b18f-a046b0f56711
# ╟─816cafa8-412e-4323-8d75-a526391c2a24
# ╠═ee0c466c-2765-4ff6-b6de-d7d8ed747036
# ╟─acbe741d-a695-422a-a6c5-21377c5dbdec
# ╟─76cf7e14-ef2f-401d-a12d-f42e1a6af8ea
# ╟─103c05e7-0985-4522-b225-43352631aff6
# ╟─8678297f-b6cb-43aa-b278-e253fc9767cc
# ╟─0d85e4dc-4251-473b-9029-d6ade20c4fcc
# ╟─0c157ab2-05b8-4ad6-ac0c-a0373baf5bd6
# ╟─7b791afc-045f-4b2b-8ad3-abf5c3a5afb9
# ╟─d28da7f2-82e0-499a-9892-57356fce50a9
# ╠═ed2581f6-21c3-465f-8578-4c6782fc929e
# ╟─8ae295aa-1bc6-42ea-a9b5-ca81deabf624
# ╟─79df6f0b-e1b3-48a4-b315-4aec7a1aed77
# ╟─86bb1bf1-eea8-49e1-b9f7-6f741dc9550c
# ╟─24b69a6f-9854-41b6-ac48-ddd5812d63e6
# ╟─bade00b7-4e71-4375-8bc3-64e5985e35e4
# ╟─fc3c3227-5680-43cd-9779-919c90690f1c
# ╠═c2dd09df-a967-46ac-9c8b-c350b2cea7cd
# ╟─5ca5dcff-6e19-4926-96b8-ed63d443a042
# ╠═c213ff64-8eba-4e00-acc8-dcfdd1a623a7
# ╠═a10d0c34-74ee-4a76-bcfe-102869cc67c7
# ╟─d5258020-0e95-45a2-8add-0ee9dcc54e46
# ╟─089fd6a3-894e-46ce-b621-77442d5e8c74
# ╟─1c2438b5-2d85-4753-aef6-740f583fe814
# ╠═aece4898-9a1e-43ee-80a8-6956542d1719
# ╟─b943c9df-ba25-4bf5-baf5-6836499645a4
# ╠═2c4af529-1e32-4b7f-be00-9e8643e45650
# ╟─4d9a6c2e-1b1c-43d8-9041-3f2779a793ea
# ╠═198f7007-dbb4-4a98-9a24-890862888706
# ╟─b573b310-c967-422c-81c9-037105c9b7fb
# ╟─f98d868f-89d5-4b47-8b30-32226275508b
# ╟─3973978f-5a12-4cae-ba33-e48e5e8daf66
# ╟─a98651fc-be42-46fd-9fc0-2982d1a52603
# ╟─d3849cc2-f0fd-49b4-aa9c-12fedfb0600a
# ╟─fad5bfc8-1b5e-4ec9-92c4-84f499dfa8ca
# ╟─1a24d1cf-f8f6-4264-9dc2-d0026dedc752
# ╟─caf5e2ea-90e7-4bc1-b251-0108814a1bcc
# ╟─787c28c5-ad8f-4156-b915-d81ee0f007d0
# ╟─f97dd99b-ef70-43bd-b81e-a8b75a701728
# ╠═65b8c072-3f08-44eb-9ea1-b18bbf0d1f9c
# ╟─1ca00c51-2005-4b17-a508-79b86344a477
# ╠═07fae632-64a4-4913-bfa8-bfb6b81fabde
# ╟─1f0d94fb-240b-43a1-b60d-6f6dd14bb66b
# ╠═529f442f-182d-45f8-840c-690196ddb52a
# ╟─8767a160-e521-4074-955d-d86d56184921
# ╠═6e793c7f-b4ed-4c8c-ae3c-1a424b68b350
# ╟─652f3b91-d61c-425e-b322-f326ea6893e5
# ╟─a33d53fe-7430-41f5-9808-d86df9dffbeb
# ╠═698433ba-9672-422c-abbd-522c6d5147f9
# ╟─8b093c35-67ae-4052-9f03-80de0fe4c2c7
# ╟─e5515428-514f-425b-96cb-14b9b6a5c8cd
# ╟─8d9bd819-7c49-4118-8c11-d67995aea5ee
# ╠═1b546127-6f1b-4267-b997-80ffc4d6afc5
# ╠═8442db0c-eb1c-4fee-b242-2810458f09f3
# ╟─9db36156-fccd-4402-9bb1-50f1c2fe9bee
# ╟─b7d94cea-068d-49a4-bbc7-993c5599f493
# ╠═b5520943-9698-4eb1-9900-79aa753dc431
# ╠═f97de562-fdc2-4bd8-abde-c212d947a036
# ╟─229f86e4-9059-419d-8be0-ab4ee3b2f6b4
# ╟─e8500d66-f1ab-43f2-beee-88e6b3f8f31d
# ╠═2efb8cd0-159a-4d78-97f7-5fb3bc2280f8
# ╠═56f8dec7-adf5-4012-87ad-5257f6bef5c1
# ╟─2f0459f9-e44e-448c-ac53-fb84f144e259
# ╠═a79db603-a35d-4807-bb70-0d493c743ca6
# ╠═7db41766-d8eb-404c-9cee-66c426c79559
# ╟─695ac71a-2bf9-4032-9db0-cff937787d6b
# ╟─dc92ba3f-3195-4941-8066-b5e1151ba3f5
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
