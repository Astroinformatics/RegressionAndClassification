### A Pluto.jl notebook ###
# v0.19.6

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

# ╔═╡ 6cdc71be-cbc7-11ec-3c09-afe22a13530f
begin
	using CSV, DataFrames
	using StatsBase, StatsModels
	using Distributions
	using EvalMetrics
	using GLM
	using Plots, ColorSchemes, LaTeXStrings
	using PlutoUI
	using Downloads
end

# ╔═╡ b4006b03-d1c3-4868-bf86-fe1c6577fed3
md"""
# Lab 3: Logistic Regression & Classification
#### [Penn State Astroinformatics Summer School 2022](https://sites.psu.edu/astrostatistics/astroinfo-su22-program/)
#### [Prof. Hyungsuk Tak](https://science.psu.edu/stat/people/hvt5139)
"""

# ╔═╡ 232af762-1bff-4b01-8d5c-911e660a3c48
md"""
In this lesson, we'll apply logistic regression as we attempt to build a classifier for high-${z}$ quasars.  
"""

# ╔═╡ f47998cc-0b90-436f-8aa7-02d8e6376652
md"""
## 1. Loading the Data

Our data set is composed of 649,439 objects, among which 20,640 are high-${z}$ quasars and 628,799 are non-high-${z}$ quasars.  In the next cell we'll load the data.
"""

# ╔═╡ cd73c9b4-803e-4fde-8df4-c57fe382478e
md"""
Take a quick glance at the first few rows of the DataFrame above to make sure that they are loaded correctly.
The first six columns (labeled ug , gr, ri, iz, zs1, and s1s2) contain six color features defined as magnitude differences (u-g , g-r, r-i, i-z, z-s1, and s1-s2, where u, g, r, i, z and Sloan band magnitudes and s1 and s2 are magnitudes in two Spitzer bands).  The last column contains binary values indicating whether the object is a high-$z$ quasars or something else.  We've flipped the label column, so 1 corresponds to a high-$z$ quasar, and 0 indicates anything else.
"""

# ╔═╡ c5c9e630-f854-4076-8856-be63defd670a
md"""
First, let's check what fraction of the objects are labeled as high-${z}$ quasars.
"""

# ╔═╡ 05c0a3c1-24b3-46d3-ab72-57238327e4bb
md"""
**Question 1a:** For this problem, are we looking for common or relatively rare objects?
"""

# ╔═╡ 15d19d64-86b6-44c9-84e1-784ea151b18b
md"""
## 2. Fitting a logistic regression model

The logistic regression model assumes that each binary response variable $Y_i$ is an independent (but not identically distributed) Bernoulli random variable with probability of being a high-$z$ quasar equal to $\theta_i$, i.e.,


$Y_i\stackrel{\textrm{ind.}}{\sim} \textrm{Bernoulli}(\theta_i)$


The mean of the response variable,  $\theta_i~(=E(Y_i))$, is connected to a linear function of regression coefficients via a logit  function.

$\textrm{logit}(\theta)=\log\left(\frac{\theta_i}{1-\theta_i}\right)=\beta_0+\beta_1x_{i1}+\beta_2x_{i2}+\cdots+\beta_6x_{i6}+\epsilon_i=x_i^{\top}\beta,$

where $x_{ij}$ is the $j$-th color feature of the $i$-th object.

The expression above shows the matrix notation: $x_i=(1, x_{i1}, x_{i2}, \ldots, x_{i6})^{\top}$ and $\beta=(\beta_0, \beta_1, \beta_2, \ldots, \beta_6)^{\top}$. We note that $x_i$'s are basically column vectors (default in mathematics). In this model, $\beta_0, \beta_1, \beta_2, \ldots$ are unknown parameters.

The main reason for  transforming $\theta_i$ to $\textrm{logit}(\theta_i)$ is to correct the mismatch between the range of $\theta_i\in(0, 1)$ and that of the linear function $x_i^{\top}\beta\in \mathbb{R}$. The range of $\textrm{logit}(\theta_i)\in \mathbb{R}$ on the left-hand side is now matched to the range of the linear function on the right-hand side, as shown in the plot of $\textrm{logit}(\theta)$ below.
"""

# ╔═╡ 598e765e-7af2-4d3c-97e7-3d9950da4e27
let
	x = range(0,stop=1,length=100)
	y = log.(x./(1 .- x))
	plot(x,y, label=:none)
	xlabel!(L"\theta")
	ylabel!(L"\mathrm{logit}(\theta)")
end

# ╔═╡ fbadcb50-efd8-45a3-97fb-6fa0b01370f3
md"""
The cell below defines a formula that specifies which variables are to be used as regressors ($x_i$'s) and that the `label` variable is to be the response variable ($Y_i$'s).
"""

# ╔═╡ 601bfbd5-9f85-4fb3-a75b-9953ef6e9d8d
fm_all = @formula(label ~ 1+ ug + gr + ri + iz + zs1 + s1s2 )

# ╔═╡ 192469ed-a067-4fc2-a997-36b68afadb1e
md"""
The next cell fits a logistic regression model to the data using that formula and saves the outcome of the logistic regression model using all the potential regressors.
A table summarizing the maximum likelihood coefficient values for each regressor (along with some related statistics) is displayed.
"""

# ╔═╡ 81cce0d1-6a09-4a5a-8934-db9cf33f8a98
md"""
It is important to know that the fit of the logistic regression also produces an estimated probability of being a high-$z$ quasar for each object. This information is useful for making a prediction (or classification), as we'll see below.

We can inspect the histogram of the predicted probabilities for the first several objects.
"""

# ╔═╡ 674be2fe-dcf9-4c6e-95a8-71897d86ff61
md"""
We can also inspect the histogram of the $\theta_i$'s for each of the high-${z}$ quasars (top) and other objects (bottom).
"""

# ╔═╡ 9a72f8b3-fd1a-49fd-abbf-18719316f5c6
md"""
The distributions for the predicted values ($\theta_i$'s) are clearly different for the two types of objects.  That's an encouraging sign that we could build a simple classifier by asking whether the predicted value is greater than a threshold.   
Before we do that, let's make a simpler logistic regression model with only two regressors, so it's easier to visualize.
"""

# ╔═╡ 6d702611-0560-4c2a-9a47-90fd7dd19d96
md"""
The generalized regression model also returned an estimate for the uncertainty of each $\beta_i$.  This allows one to perform a quick test of how important each regressor is in the model.  In this case, all of the regression coefficients except  $\beta_5$ are significantly different from zero at the significance level $\alpha=0.05$.
We can see this because their asymptotic $z$-tests (not $t$-tests) based on the asymptotic property of the maximum likelihood estimation show that their $p$-values are very small (much smaller than $\alpha=0.05$).
"""

# ╔═╡ 86fc2a81-49d6-4f71-9573-3d84e3f54f65
md"""
**Question 2a:** Looking at the $p$-values for each regressor, which variable could we consider excluding from the model?
"""

# ╔═╡ 3e63bdd0-cbc8-4941-a511-72ce6164ea2a
#A: zs1, as its $p$-value 0.0743 is much greater than the $p$-values of other regressors.  

md"""
If one were using the common $p > 0.05$ criterion for statistical significance, then one could consider that this color likely has little effect on the odds of being a high-$z$ quasar when the other colors are held constants in this model.  Note that this is a highly heuristic method and that there are more sophisticated methods for selecting which variable to include in a model (e.g., AIC scores, cross-validation scores, L1-regularization).  But we'll use the $p$-values for now, so as not to get distracted by learning about these other methods.
"""

# ╔═╡ 98541794-5a2c-4e6c-9829-a06181f9a22f
md"""
## 3.  Visualizing a logistic regression model
It's much easier to visualize the predictions of a logistic regression model in two dimensions.  Based on our model above, select two regressors with the extreme $z$-scores.  Below, we'll try fitting a new logistic regression model with just those two regressors.
"""
# most extreme: ug and zs1

# ╔═╡ 0a3d72bd-dfa2-4256-add2-39c1ce430e4f
@bind lrm_2d_vars PlutoUI.combine() do Child
	md"""
	Colors to use for 2-d logistic regression model:  
	x_1: $(Child(Select([:ug => "u-g", :gr => "g-r", :ri => "r-i", :iz => "i-z", :zs1 => "z-s1", :s1s2 => "s1-s2"], default=:ug)))    
	x_2: $(Child(Select([:ug => "u-g", :gr => "g-r", :ri => "r-i", :iz => "i-z", :zs1 => "z-s1", :s1s2 => "s1-s2"], default=:iz)))
	"""
end

# ╔═╡ 99c787e3-8cc8-4533-8c9e-2e7125a5dbd3
md"""
As before, the distribution of $\theta_i$'s for high-${z}$ quaesars are distinct from that of the other objects.  Since we've only used two regressors, we can plot the contours $\theta_i$ and overplot a subset of the points.  
"""

# ╔═╡ 669dc84f-b9e3-4b05-9810-da75c7b22b91
md"""
**Question 3a:**  Based on the plot above, do you anticipate that the logistic regression model based on the two colors you choose will be a good choice for identifying high-${z}$ quasars?  Why or why not?

**Question 3b:**  Try a few other combinations of colors.  Which do you think will be good for separating high-${z}$ quasars and other objects?
"""

# ╔═╡ ac56b744-4b0a-41c2-b38f-2f772bb3aad2
md"## 4.  Evaluating the model"

# ╔═╡ d3577973-c758-47e0-9806-4fe35a087664
md"""
Before using a model for doing science, it's important to investigate the quality of the model.  
When we fit a linear model to data, we will always get a set of best fit coefficients.  However, that does not mean that it is a good model.  

For example, is our logistic model using 6 colors as inputs better than or statistically equivalent to our model using only 2 colors?

The default outputs above did not test whether all features are meaningless (i.e., our model is statistically equivalent to $\beta_1=\beta_2=\cdots=\beta_6=0$) or whether *at least one* feature is meaningful (non-zero) for the purpose of predicting the odds of being a high-$z$ quasar.

We can perform tests by making use of the following insights.  
The **deviance** is a measure of the quality of fit and is defined as
$\mathrm{Deviance} = -2 \ln(\mathrm{likelihood}).$
The test statistic $T$ is the difference between $\texttt{Residual}$ $\texttt{deviance}$ for a model and $\texttt{Null}$ $\texttt{deviance}$ (i.e., deviance for a model with no regressors).  
"""

# ╔═╡ 35821fb4-0c02-4c9a-882d-f1993204aca5
md"""
This test statistic asymptotically follows the $\chi^2_m$ distribution by the asymptotic property of the generalized likelihood ratio test statistic, where $m$ is the number of features (not counting the intercept term). The observed test statistic for our full model is
"""
#$t = 182987 - 85498 = 97489.$

# ╔═╡ cbc2364d-308d-4ca9-bfca-4bc82637e6ef
md"""
The resulting $p$-value for comparing our full model to the null model is defined as

$P(T\ge t\mid H_0)$

For our full model, $T$ is a $\chi^2_6$ random variable, since the full model includes 6 extra coefficients. We can compute the $p$-value with the following code.  (Actually, we'll compute the log of the $p$-value, since it can be quite small.)
"""

# ╔═╡ b56eaea5-d140-4b9a-8181-0cc0c7b603cf
function calc_log_pvalue_two_logistic_regression_models(model_big, model_small)
	dof = length(coef(model_big)) - length(coef(model_small))
	dist = Chisq( dof )
	t = deviance(model_small) - deviance(model_big)
	logccdf( dist, t )
end

# ╔═╡ 77b3381a-ec6b-4550-84e9-f2bc09341353
md"""
Since this $p$-value is much, much smaller than $\alpha=0.05$, we reject the null hypothesis, concluding that the proposed model with 6 colors is at least meaningful.  
Of course, merely passing this goodness-of-fit test does not always mean that the model is optimal or good for the intended scientific application.  

Now, we can perform similar tests comparing your logistic regression model with just 2 regressors to the null model.
"""

# ╔═╡ deff64f2-eda5-4902-a9de-fa778fd71ab6
md"""
**Question 4a:**  Precisely state the conclusion that you can draw from the $p$-value above.
"""

# ╔═╡ 400a98d3-35f2-4761-9a64-222376c794ef
md"### Model comparison"

# ╔═╡ d280ee8b-31f2-45f5-838b-cac47024c1cc
md"""
We can compare two logistic model in a heuristic fashion based on one of the information criteria.  In this case, we will use the Akaike Information Criterion (AIC) rather than the Bayesian Information Criterion (BIC), because BIC may put too heavy of a penalty on the more complex model when the data size is large.  

$\mathrm{AIC} = 2k - 2 \ln(\mathrm{likelihood})$

$\mathrm{BIC} = 2k \ln(n) - 2 \ln(\mathrm{likelihood})$

**Question 4b:** Compare the size of the penalties applied by AIC and BIC when comparing the 6 and 2 component logistic regression models for this data set.
"""

# ╔═╡ b3f5129e-fc15-4733-b583-603b19c9efac
md"""
**Question 4c:**  Based on the calculations above, is the logistic regression model with all 6 regressors a meaningful improvement over the logistic regression model with just 2 regressors?  
"""

# ╔═╡ bea8e902-d1d0-4210-87ad-74c36d6bcc02
md"## 5.  Classification using Logistic Regression"

# ╔═╡ 5d32866d-246d-43f5-9461-0bd63d8127be
md"""
Now, let us turn the logistic regression model into a classifier.  According to the statistical model, a natural choice for the threshold would be 0.5.  Let's see how well such a classifier based on either of our two logistic regression models would perform.
"""

# ╔═╡ c1705b81-749c-4abc-b196-1c1d42559d56
classify(model; threshold = 0.5 ) = predict(model) .>= threshold;

# ╔═╡ 55ee38d0-0bec-4a41-be22-9a66433d1301
md"""
**Question 5a:**  98% accuracy sounds pretty good.  In the context of this dataset, are you impressed with 98% accuracy?  

**Question 5b:**  Can you think of a way to build a ridiculously simple "classifier" that would have an accuracy nearly as good?
"""

# ╔═╡ c874b8e3-6637-474a-8043-af4e0852f9b8
md"### Contingency tables"

# ╔═╡ 395dd6cc-efcc-46bb-b824-d9029aa29fe7
md"Instead of looking only at the accuracy averaged over all cases, let us break down each of the possibilities with a contingency table.  Rows correspond to actual positives (i.e., objects labeled as high-${z}$ quasars) and for actual negatives (i.e., objects labeled as something else).  Columns correspond to predicted positives (i.e., objects predicted to be high-${z}$ quasars) and predicted negatives (i.e., objects predicted to be something else).
(If you're already familiar with some of the other diagnostics, you can add a cell and inspect the elements of `diag_all` computed below.)

First, we'll inspect the contingency table for the full model."

# ╔═╡ b82d8dc7-7654-426b-ad45-851a65cb9033
md"""
**Question 5c:** How does the rate of true negatives (i.e., an object is both an actual negative and a predicted negative) compare to the total accuracy?  Why?

**Question 5d:** If you knew an object were not a high-${z}$ quasar, then would it be wise for you trust the results of the classifier?  Why?

**Question 5e:** If the classifier predicts an object is a high-${z}$ quasar, then would it be wise for you to trust the results of the classifier?   Why?

"""

# ╔═╡ 5c1ad658-2506-4dd8-8ff1-007e60eedb42
md"""
We can compare to the contingency table for the model using only 2 regressors.
While the accuracies of the two models are very similar, we see that there is a more substantial difference in the false discovery rate.
"""

# ╔═╡ d18eadc8-03e7-4571-97aa-df4e0e8975b1
md"""
**Question 5f:**  Based on the histograms and contour plots above, how do you expect that the behavior of a classifier would differ if we were to vary the threshold used by the classifier?  

Once you've made your predictions, continue on to see the results below.
"""

# ╔═╡ f082e555-4c0d-4b03-bd04-8b99f4eae9ea
md"""
Which logistic regression model to plot diagnostics for:  
$(@bind lrm_to_plot Select([:lrm_all=>"6 regressors", :lrm_5d=>"5 regressors", :lrm_2d=>"2 regressors"]))
"""

# ╔═╡ 8fa5489e-9405-4460-8e75-eff384db6554
md"""
We can see that choosing a threshold less than 0.5 can significantly reduce the false negative rate, while causing only a small increase in the false positive rate.  One tool that's often used to evaluate the trade-off when set tin a threshold is the [Receiver operating characteristic curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) (often abbreviated ROC curve), shown below.
"""

# ╔═╡ 2b191451-5c29-4ee1-997d-c885b0928cdd
md"### Selecting a detection threshold"

# ╔═╡ 0d484210-3a9d-48a9-8350-b554c325e5bb
md"""
**Question 5g:** If your goal were to identify a sample of a few hundred objects very likely to be high-${z}$ quasars, what threshold would be a good choice?

**Question 5h:** If your goal were to identify objects which could be studied further in order to identify a rare subclass of high-${z}$ quaesars, what threshold would be a good choice?

Test your reasoning by selecting the 2 regressor model above, choosing a threshold, and varying the threshold in the box below.
"""

# ╔═╡ be3c9982-0b47-4d65-b275-6deef8639cf1
md"""
Threshold for 2 regressor model: $(@bind thresh_2d NumberField(0.0:0.01:1; default=0.5))
"""

# ╔═╡ a2fb8b6e-8919-4b1d-931f-ec8ef837a593
md"### 5-Regressor model"

# ╔═╡ a8dcdad1-a35f-4a4e-b32c-6542b6f21f37
md"""
We could also consider a logistic regression classifier based on 5 of the regressors.
"""

# ╔═╡ fff26dda-6c92-4135-831e-ab7dc422dcc6
md"""
Color to exclude from 5-regressor model: $(@bind lrm_excl_1var Select([:ug => "u-g", :gr => "g-r", :ri => "r-i", :iz => "i-z", :zs1 => "z-s1", :s1s2 => "s1-s2"], default=:zs1))
"""

# ╔═╡ b654b8b5-fbcc-4119-ae6c-13ccd55398fb
md"""
**Question 5i:** Compare the effectiveness of the logistic classier depending on the number of regressors used based on the diagnostic plots between questions 5f and 5g.
"""

# ╔═╡ cd331dad-7fe9-4f1e-94db-3d103bc7a350
md"""
## Next steps
If you're interested in seeing how logistic regression has been applied the astronomical literature, some recent examples are:
- [Rowlinson et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019A%26C....27..111R/abstract)
- [Cheng et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.4209C/abstract)
"""

# ╔═╡ 194c07d8-5d78-4051-a647-090c94de7fb4
md"""
## Helper functions
"""

# ╔═╡ dc28f942-9ba8-45ee-badb-84326b077d62
function fit_logistic_regression( fm::FormulaTerm, data::DataFrame )
	# Confirm that data contains response variable
	@assert fm_all.lhs ∈ term.(names(data))   
	# Confirm that data contains regressor variables
	@assert all(map(t->t ∈ term.(names(data)) || t isa ConstantTerm, fm_all.rhs))  
	# Fit a model to the data using Logistic Regression using GLM.jl's glm function
	glm(fm, data, Binomial(), LogitLink() )
end

# ╔═╡ ad86f7c4-cce8-446c-bed1-eae4800f1bda
function make_threshold_list(;num_small=16, num_mid=10, num_large=8)
	vcat(10 .^ range(-8,stop=-1, length=num_small),
		range(0.1, stop=0.9, length=num_mid+2)[2:end-1],
		reverse(1 .- 8 .^ range(-10,stop=-1, length=num_large))
		)
end

# ╔═╡ c9a2eb7f-5189-4aab-bcb7-6b3e7f50689c
accuracy(model, data::DataFrame; threshold::Real = 0.5) = sum( classify(model, threshold=threshold) .== data.label ) / length(data.label)

# ╔═╡ a6338962-ef11-479d-8e3c-1975703b1058
misclassified(model, data::DataFrame; threshold::Real=0.5) = classify(model, threshold=threshold) .!= data.label

# ╔═╡ 29f1f1a6-a542-4792-8825-cd37aee7b7d7
function plot_predictions_2d( data::DataFrame, model, xcol::Symbol, ycol::Symbol ;
     plot_misclassified::Bool = true, threshold::Real=0.5)		
	label_mask         = data.label .== 1
	misclassified_mask =  misclassified(model,data, threshold=threshold)

	plt = plot(palette = palette(:RdBu_4))
	scatter!(plt, data[label_mask,xcol],data[label_mask,ycol],ms=0,mc=:2, label="True Positive")
	scatter!(plt, data[.!label_mask,xcol],data[.!label_mask,ycol],ms=0,mc=3, label="True Negative")
	if plot_misclassified
		scatter!(plt, data[misclassified_mask .& .!label_mask,xcol],data[misclassified_mask .& .! label_mask,ycol],ms=2,mc=4, markerstrokewidth=0, label="False Positive")
		scatter!(plt, data[misclassified_mask .& label_mask,xcol],data[misclassified_mask .& label_mask,ycol],ms=2,mc=1, markerstrokewidth=0, label="False Negative")
	end
	xlabel!(plt,string(xcol))
	ylabel!(plt,string(ycol))

	return plt
end

# ╔═╡ 5e97c7fe-38c1-4a4c-a3f0-f985f981b8aa
function calc_classification_diagnostics(model, data; threshold = 0.5)
	pred = classify(model; threshold=threshold)
	num_true_positives  = sum(  data.label.==1 .&&   pred)
	num_true_negatives  = sum(  data.label.==0 .&& .!pred)
	num_false_negatives = sum(  data.label.==1 .&& .!pred)
	num_false_positives = sum(  data.label.==0 .&&   pred)

	num_condition_positives = num_true_positives + num_false_negatives
	num_condition_negatives = num_true_negatives + num_false_positives
	num_total = num_condition_positives + num_condition_negatives
	num_predicted_positives = num_true_positives + num_false_positives
	num_predicted_negatives = num_true_negatives + num_false_negatives
	true_positive_rate  = num_true_positives/num_condition_positives
	true_negative_rate  = num_true_negatives/num_condition_negatives
	false_positive_rate = num_false_positives/num_condition_negatives
	false_negative_rate = num_false_negatives/num_condition_positives
	accuracy = (num_true_positives+num_true_negatives)/num_total
	false_omission_rate = num_false_negatives / num_predicted_negatives
	false_discovery_rate = num_false_positives / num_predicted_positives
	F1_score = 2*num_true_positives/(2*num_true_positives+num_false_positives+num_false_negatives)
	prevalence = (num_true_positives+num_false_negatives)/num_total
	return (;threshold, accuracy, false_discovery_rate, false_omission_rate, F1_score,
		false_positive_rate, false_negative_rate, true_positive_rate, true_negative_rate,
		num_true_positives, num_true_negatives, num_false_positives, num_false_negatives,   
		num_condition_positives, num_condition_negatives, num_predicted_positives, num_predicted_negatives,
		num_total, prevalence )
end

# ╔═╡ c429f200-24c0-41b3-8660-a0ebff7cd5a9
function binary_classification_table(diag; digits::Integer=2)
	apppr = round(diag.num_true_positives/diag.num_total*100, digits=digits)
	anppr = round(diag.num_false_positives/diag.num_total*100, digits=digits)
	anpnr = round(diag.num_true_negatives/diag.num_total*100, digits=digits)
	appnr = round(diag.num_false_negatives/diag.num_total*100, digits=digits)
	ncp = diag.num_condition_positives
	ncf = diag.num_condition_negatives
	npp = diag.num_predicted_positives
	npn = diag.num_predicted_negatives
	nt = diag.num_total
    md"""
|                     | Predicted Positive | Predicted Negative| Count |
|:--------------------|--------------------|-------------------|-------|
| **Actual Positive** | $apppr%            | $appnr%           | $ncp  |
| **Actual Negative** | $anppr%            | $anpnr%           | $ncf  |          
| **Count**           | $npp               | $npn              | $nt   | """        
end

# ╔═╡ 0f391345-5b71-45a7-b826-bc98c1feef9c
md"""
## Setup
"""

# ╔═╡ 663e62c3-480c-4bc3-a11b-f887754f5a5a
TableOfContents()

# ╔═╡ 475c352a-5f0d-40c8-ad74-4a4c81e8d5e1
function find_or_download_data(data_filename::String, url::String)
	if contains(gethostname(),"ec2.internal")
		data_path = joinpath(homedir(),"data")
		isdir(data_path) || mkdir(data_path)
	elseif contains(gethostname(),"aci.ics.psu.edu")
		data_path = joinpath("/gpfs/scratch",ENV["USER"],"Astroinformatics")
		isdir(data_path) || mkdir(data_path)
		data_path = joinpath(data_path,"data")
		isdir(data_path) || mkdir(data_path)
	else
		data_path = joinpath(homedir(),"Astroinformatics")
		isdir(data_path) || mkdir(data_path)
		data_path = joinpath(data_path,"data")
		isdir(data_path) || mkdir(data_path)
	end
	data_path = joinpath(data_path,data_filename)
	if !(filesize(data_path) > 0)
		Downloads.download(url, data_path)
	end
	return data_path		
end

# ╔═╡ fef9e716-e7f5-48c0-a82e-1a950b8bf893
begin
	data_filename = "quasar2.csv"
	url = "https://scholarsphere.psu.edu/resources/edc61b33-550d-471d-8e86-1ff5cc8d8f4d/downloads/19732"
	data_path = find_or_download_data(data_filename,url)
end

# ╔═╡ 9e59eece-1853-468b-84f6-3a28f3852f9c
begin
	data = CSV.read(data_path,DataFrame) #,limit=40000)
	data[:,:label] .= 1 .- data[:,:label]  # Make label=1 for high-z quasars
	data
end

# ╔═╡ 50e27175-3261-4889-a4f6-4d4267b19a19
begin
	label_mask = data.label .== 1   # high-z quasars indicated with a label of 1
	frac_label_1 = sum(label_mask)/length(data.label)
end

# ╔═╡ 177df112-4b58-488e-868e-19fea591e3cb
lrm_all = fit_logistic_regression(fm_all, data)

# ╔═╡ aec4b19f-dea6-4534-b23a-88ea4f8c240f
begin
	β_ug_all = round(coef(lrm_all)[2],digits=5)
	exp_β_ug_all = round(exp(coef(lrm_all)[2]),digits=5)
end;

# ╔═╡ d4b671b8-f9ed-4f02-a00f-23f59b4f7883
md"""
The maximum likelihood estimate for ``\beta_1`` (for ug) can be interpreted as follows. "When all the other color features are held constants (i.e., when values of gr, ri, iz, zs1, and s1s2 are fixed at some constants), a unit increase in ug changes the odds of being a high-``z`` quasar by a factor of exp($(β_ug_all)) = $(exp_β_ug_all)." Note that the odds of being a high-``z`` quasar are defined as the probability of being a high-``z`` quasar divided by the probability of not being a high-``z`` quasar, i.e., ``\frac{\theta_i}{1 - \theta_i}``.
"""

# ╔═╡ b4ac8465-a1ea-4faf-8f1e-4ba803efe8fe
predict(lrm_all)

# ╔═╡ 42f4b383-b5db-47ca-9e18-8f26fa8eae3b
let
	plt1 = histogram(predict(lrm_all)[label_mask], nbins=20, legend=:none, label="1", fc=:red, title="high-z quasars")
	xlabel!(plt1,L"\theta_i")
	plt2 = histogram(predict(lrm_all)[.!label_mask], nbins=20, legend=:none, label="0", fc=:blue, title="other objects")
	xlabel!(plt2,L"\theta_i")

	l = @layout [a; b]
	plt = plot(plt1,plt2, layout=l )
	ylabel!(plt,"Number")
end

# ╔═╡ 1541d99c-e7af-4504-99e2-987d522c7ed1
begin
	lrm_2d_ready =  lrm_2d_vars[1] != lrm_2d_vars[2]
	if lrm_2d_ready
		formula_2d = Term(:label) ~ term(1) + term.(lrm_2d_vars);
		lrm_2d = fit_logistic_regression(formula_2d, data)
	end
end

# ╔═╡ c4824594-56a5-448c-87a4-0d84272a61e7
if lrm_2d_ready
	plt1 = histogram(predict(lrm_2d)[label_mask], nbins=20, legend=:none, label="1", fc=:red, title="high-z quasars")
	xlabel!(plt1,L"\theta_i")
	plt2 = histogram(predict(lrm_2d)[.!label_mask], nbins=20, legend=:none, label="0", fc=:blue, title="other objects")
	xlabel!(plt2,L"\theta_i")

	l = @layout [a; b]
	plt = plot(plt1,plt2, layout=l )
	ylabel!(plt,"Number")
end

# ╔═╡ 945bfab4-0dfa-487f-a2bc-631acdaea482
deviance(lrm_2d)

# ╔═╡ b4dae5ce-16e2-4aa8-8979-6b81345eadf3
(;ΔAIC = aic(lrm_all) - aic(lrm_2d), ΔBIC = bic(lrm_all) - bic(lrm_2d) )

# ╔═╡ 1a6460f9-d7df-42dd-95ca-527233bf2f67
function plot_logistic_contours_scatter(model, data, x_col::Symbol, y_col::Symbol; 	num_pts_plt = 4000, grid_size = 50)
	@assert( string(x_col) ∈ names(data) )
	@assert( string(y_col) ∈ names(data) )
	@assert( "label" ∈ names(data) )
	#@assert( term(x_col) ∈ model.mf.f.rhs.terms)  # issue with unknown vs continuous  term
	#@assert( term(y_col) ∈ model.mf.f.rhs.terms)

	extreme_x = extrema(data[!,x_col])
	extreme_y = extrema(data[!,y_col])
	x_grid = range(extreme_x[1], stop=extreme_x[2], length=grid_size)
	y_grid = range(extreme_y[1], stop=extreme_y[2], length=grid_size)
	z_grid = [ first(predict(lrm_2d,DataFrame([[x],[y]],[x_col,y_col]))) for x in x_grid, y in y_grid]
	plt = plot( legend=:topleft)
	contour!(plt, x_grid,y_grid,z_grid, colorbar_title=L"\theta_i")
	xlabel!(string(x_col))
	ylabel!(string(y_col))

	idx_sample = sample(1:size(data,1),num_pts_plt,replace=false,ordered=true)
	x_pts = data[idx_sample, x_col]
	y_pts = data[idx_sample, y_col]
	z_pts = predict(lrm_2d)
	mask = data.label[idx_sample] .== 1

	scatter!(plt, x_pts[.!mask],   y_pts[.!mask], ms=1.5, markerstrokewidth=0, mc=:blue, label="Other" )
	scatter!(plt, x_pts[mask], y_pts[mask],   ms=1.5, markerstrokewidth=0, mc=:red, label="High-z quaesar" )
end

# ╔═╡ c7168e82-863b-4004-8527-9b033bd0c5d0
plot_logistic_contours_scatter(lrm_2d, data, lrm_2d_vars[1], lrm_2d_vars[2])

# ╔═╡ 2c50f0a5-bebf-438c-a80c-8df00ccba601
lrm_null =  fit_logistic_regression( @formula(label ~ 1 ) , data)

# ╔═╡ 6cdeeb3f-8276-4e5e-9870-cfbca1676673
md"""
Deviance (6 colors): $(round(deviance(lrm_all),digits=1))

Deviance (0 colors): $(round(deviance(lrm_null),digits=1))
"""

# ╔═╡ 6438f9db-5d36-45ee-bb73-9405d1029a6b
t_all_vs_null = deviance(lrm_all) - deviance(lrm_null)

# ╔═╡ 45e6135e-a6b6-40a5-b10a-f483f09893c0
calc_log_pvalue_two_logistic_regression_models( lrm_all, lrm_null )

# ╔═╡ d9a809b2-de1b-4346-9c66-af4681b6e974
calc_log_pvalue_two_logistic_regression_models( lrm_2d, lrm_null )

# ╔═╡ 93015a3f-55ed-4ace-8b01-c1830e104c89
# accuracy (and other helper functions) are defined at the bottom of the notebook
accuracy(lrm_all, data)   

# ╔═╡ 10155825-92d6-4e1f-b388-edee647e0f63
accuracy(lrm_2d, data)   

# ╔═╡ fc691365-0edc-41da-8f60-49eb92528479
begin
	diag_all = calc_classification_diagnostics(lrm_all, data)
	binary_classification_table(diag_all)
end

# ╔═╡ 2f734a61-e940-40e5-84c3-4bc0d3b4687f
begin
	Δk = length(lrm_all.mf.f.rhs.terms) - length(lrm_2d.mf.f.rhs.terms)
	Δklogn = Δk*log(diag_all.num_total)
	(;Δk , Δklogn)
end

# ╔═╡ d3105131-57af-4f55-8f68-76d49086aac0
begin
	diag_2 = calc_classification_diagnostics(lrm_2d, data)
	binary_classification_table(diag_2)
end

# ╔═╡ dc104c01-85fa-4c3c-857f-7eb28c01931f
begin
	threshold_list = make_threshold_list()
	diagnostics_vs_threshold = DataFrame(map(t->calc_classification_diagnostics(eval(lrm_to_plot),data,threshold=t), threshold_list))
end;

# ╔═╡ 98c554e1-694f-4df4-b34e-3a0b3921f773
let
	df = diagnostics_vs_threshold
	plt1 =
	plot(df.threshold,df.false_negative_rate, xscale=:log10, lc=:red, label=:none)#"False ", legend=:bottomright)
	scatter!(plt1, df.threshold,df.false_negative_rate, mc=:red, label=:none)
	xlabel!(plt1,"Threshold")
	ylabel!(plt1,"False Negative Rate")
	xlims!(plt1,1e-4,1)
	ylims!(plt1,0,1)

	plt2 =
	plot(df.threshold,df.false_positive_rate, xscale=:log10, lc=:red, label=:none)#"False ", legend=:bottomright)
	scatter!(plt2, df.threshold,df.false_positive_rate, mc=:red, label=:none)
	xlabel!(plt2,"Threshold")
	ylabel!(plt2,"False Positive Rate")
	xlims!(plt2,1e-4,1)
	ylims!(plt2,0,1)

	plt3 = plot(df.threshold,df.false_discovery_rate, xscale=:log10, lc=:red, label=:none)#"False ", legend=:bottomright)
	scatter!(plt3, df.threshold,df.false_discovery_rate, mc=:red, label=:none)
	xlabel!(plt3,"Threshold")
	ylabel!(plt3,"False Discovery Rate")
	xlims!(plt3,1e-4,1)
	ylims!(plt3,0,1)

	plt4 = plot(df.threshold,df.false_omission_rate, xscale=:log10, lc=:red, label=:none)#"False ", legend=:bottomright)
	scatter!(plt4, df.threshold,df.false_omission_rate, mc=:red, label=:none)
	xlabel!(plt4,"Threshold")
	ylabel!(plt4,"False Omission Rate")
	xlims!(plt4,1e-4,1)
	ylims!(plt4,0,1)


	plt5 = plot(df.false_positive_rate,df.true_positive_rate, lc=:red, label="ROC Curve", legend=:bottomright)
	scatter!(plt5,df.false_positive_rate,df.true_positive_rate, label=:none, mc=:red)
	xlabel!(plt5,"False Positive Rate")
	ylabel!(plt5,"True Positive Rate")
	xlims!(plt5,0,1)
	ylims!(plt5,0,1)

	l = @layout [a b;c d] #; e]
	plt = plot(plt1,plt2,plt3,plt4, layout=l)
end

# ╔═╡ 6628f3ec-2b7a-477c-83f8-6393dd86a2a5
let
	df = diagnostics_vs_threshold
	plt5 = plot(df.false_positive_rate,df.true_positive_rate, lc=:red, label="ROC Curve", legend=:bottomright)
	scatter!(plt5,df.false_positive_rate,df.true_positive_rate, label=:none, mc=:red)
	xlabel!(plt5,"False Positive Rate")
	ylabel!(plt5,"True Positive Rate")
	xlims!(plt5,0,0.1)
	ylims!(plt5,0,1)
end

# ╔═╡ 7515d1f5-0490-417e-808b-28b5902580b5
plot_predictions_2d(data, lrm_2d, lrm_2d_vars[1], lrm_2d_vars[2], threshold=thresh_2d )

# ╔═╡ 32c9945d-a30e-4a95-9065-e867c59ebf06
begin
	candidate_terms = fm_all.rhs
	fm_5d = Term(:label) ~ term(1) + sum(setdiff(candidate_terms,[term(lrm_excl_1var)]))
	lrm_5d = fit_logistic_regression(fm_5d, data)
end

# ╔═╡ 4f2df587-aba9-42e3-b4d0-bf6ef218c0ae
ΔAIC_6_5 = aic(lrm_all)-aic(lrm_5d)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Downloads = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
EvalMetrics = "251d5f9e-10c1-4699-ba24-e0ad168fa3e4"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsModels = "3eaba693-59b7-5ba5-a881-562e759f1c8d"

[compat]
CSV = "~0.10.4"
ColorSchemes = "~3.17.1"
DataFrames = "~1.3.4"
Distributions = "~0.25.57"
EvalMetrics = "~0.2.1"
GLM = "~1.7.0"
LaTeXStrings = "~1.3.0"
Plots = "~1.28.1"
PlutoUI = "~0.7.38"
StatsBase = "~0.33.16"
StatsModels = "~0.6.29"
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

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9950387274246d08af38f6eef8cb5480862a435f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.14.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "12fc73e5e0af68ad3137b886e3f7c1eacfca2640"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.17.1"

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

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "b153278a25dd42c65abbf4e62344f9d22e59191b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.43.0"

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

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "f206814c860c2a909d2a467af0484d08edd05ee7"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.57"

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

[[deps.EvalMetrics]]
deps = ["DocStringExtensions", "RecipesBase", "Reexport", "Statistics", "StatsBase"]
git-tree-sha1 = "08edf9183b70a0a8c075e6bb855561967c84abac"
uuid = "251d5f9e-10c1-4699-ba24-e0ad168fa3e4"
version = "0.2.1"

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

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "92b8d38886445d6d06e5f13201e57d018c4ff880"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.7.0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "af237c08bda486b74318c8070adb96efa6952530"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.2"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "cd6efcf9dc746b06709df14e462f0a3fe0786b1e"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.2+0"

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
git-tree-sha1 = "91b5dcf362c5add98049e6c29ee756910b03051d"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.3"

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

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "76c987446e8d555677f064aaac1145c4c17662f8"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.14"

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

[[deps.NaNMath]]
git-tree-sha1 = "737a5957f387b17e74d4ad2f440eb330b39a62c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.0"

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
git-tree-sha1 = "3114946c67ef9925204cc024a73c9e679cebe0d7"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.8"

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
git-tree-sha1 = "d05baca9ec540de3d8b12ef660c7353aae9f9477"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.28.1"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "670e559e5c8e191ded66fa9ea89c97f10376bb4c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.38"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

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

[[deps.ShiftedArrays]]
git-tree-sha1 = "22395afdcf37d6709a5a0766cc4a5ca52cb85ea0"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "1.0.0"

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
git-tree-sha1 = "5ba658aeecaaf96923dce0da9e703bd1fe7666f9"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.4"

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
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5950925ff997ed6fb3e985dcce8eb1ba42a0bbe7"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.18"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "03c99c7ef267c8526953cafe3c4239656693b8ab"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.29"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "8f705dd141733d79aa2932143af6c6e0b6cea8df"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.6"

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

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

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
# ╟─b4006b03-d1c3-4868-bf86-fe1c6577fed3
# ╟─232af762-1bff-4b01-8d5c-911e660a3c48
# ╟─f47998cc-0b90-436f-8aa7-02d8e6376652
# ╠═fef9e716-e7f5-48c0-a82e-1a950b8bf893
# ╠═9e59eece-1853-468b-84f6-3a28f3852f9c
# ╟─cd73c9b4-803e-4fde-8df4-c57fe382478e
# ╟─c5c9e630-f854-4076-8856-be63defd670a
# ╠═50e27175-3261-4889-a4f6-4d4267b19a19
# ╟─05c0a3c1-24b3-46d3-ab72-57238327e4bb
# ╟─15d19d64-86b6-44c9-84e1-784ea151b18b
# ╟─598e765e-7af2-4d3c-97e7-3d9950da4e27
# ╟─fbadcb50-efd8-45a3-97fb-6fa0b01370f3
# ╠═601bfbd5-9f85-4fb3-a75b-9953ef6e9d8d
# ╟─192469ed-a067-4fc2-a997-36b68afadb1e
# ╠═177df112-4b58-488e-868e-19fea591e3cb
# ╟─aec4b19f-dea6-4534-b23a-88ea4f8c240f
# ╟─d4b671b8-f9ed-4f02-a00f-23f59b4f7883
# ╟─81cce0d1-6a09-4a5a-8934-db9cf33f8a98
# ╠═b4ac8465-a1ea-4faf-8f1e-4ba803efe8fe
# ╟─674be2fe-dcf9-4c6e-95a8-71897d86ff61
# ╟─42f4b383-b5db-47ca-9e18-8f26fa8eae3b
# ╟─9a72f8b3-fd1a-49fd-abbf-18719316f5c6
# ╟─6d702611-0560-4c2a-9a47-90fd7dd19d96
# ╟─86fc2a81-49d6-4f71-9573-3d84e3f54f65
# ╟─3e63bdd0-cbc8-4941-a511-72ce6164ea2a
# ╟─98541794-5a2c-4e6c-9829-a06181f9a22f
# ╟─0a3d72bd-dfa2-4256-add2-39c1ce430e4f
# ╟─1541d99c-e7af-4504-99e2-987d522c7ed1
# ╟─c4824594-56a5-448c-87a4-0d84272a61e7
# ╟─99c787e3-8cc8-4533-8c9e-2e7125a5dbd3
# ╠═c7168e82-863b-4004-8527-9b033bd0c5d0
# ╟─669dc84f-b9e3-4b05-9810-da75c7b22b91
# ╟─ac56b744-4b0a-41c2-b38f-2f772bb3aad2
# ╟─d3577973-c758-47e0-9806-4fe35a087664
# ╟─2c50f0a5-bebf-438c-a80c-8df00ccba601
# ╟─6cdeeb3f-8276-4e5e-9870-cfbca1676673
# ╟─35821fb4-0c02-4c9a-882d-f1993204aca5
# ╠═6438f9db-5d36-45ee-bb73-9405d1029a6b
# ╟─cbc2364d-308d-4ca9-bfca-4bc82637e6ef
# ╠═b56eaea5-d140-4b9a-8181-0cc0c7b603cf
# ╠═45e6135e-a6b6-40a5-b10a-f483f09893c0
# ╟─77b3381a-ec6b-4550-84e9-f2bc09341353
# ╠═945bfab4-0dfa-487f-a2bc-631acdaea482
# ╠═d9a809b2-de1b-4346-9c66-af4681b6e974
# ╟─deff64f2-eda5-4902-a9de-fa778fd71ab6
# ╟─400a98d3-35f2-4761-9a64-222376c794ef
# ╟─d280ee8b-31f2-45f5-838b-cac47024c1cc
# ╠═2f734a61-e940-40e5-84c3-4bc0d3b4687f
# ╠═b4dae5ce-16e2-4aa8-8979-6b81345eadf3
# ╟─b3f5129e-fc15-4733-b583-603b19c9efac
# ╟─bea8e902-d1d0-4210-87ad-74c36d6bcc02
# ╟─5d32866d-246d-43f5-9461-0bd63d8127be
# ╠═c1705b81-749c-4abc-b196-1c1d42559d56
# ╠═93015a3f-55ed-4ace-8b01-c1830e104c89
# ╠═10155825-92d6-4e1f-b388-edee647e0f63
# ╟─55ee38d0-0bec-4a41-be22-9a66433d1301
# ╟─c874b8e3-6637-474a-8043-af4e0852f9b8
# ╟─395dd6cc-efcc-46bb-b824-d9029aa29fe7
# ╟─fc691365-0edc-41da-8f60-49eb92528479
# ╟─b82d8dc7-7654-426b-ad45-851a65cb9033
# ╟─5c1ad658-2506-4dd8-8ff1-007e60eedb42
# ╟─d3105131-57af-4f55-8f68-76d49086aac0
# ╟─d18eadc8-03e7-4571-97aa-df4e0e8975b1
# ╟─f082e555-4c0d-4b03-bd04-8b99f4eae9ea
# ╟─dc104c01-85fa-4c3c-857f-7eb28c01931f
# ╟─98c554e1-694f-4df4-b34e-3a0b3921f773
# ╟─8fa5489e-9405-4460-8e75-eff384db6554
# ╟─6628f3ec-2b7a-477c-83f8-6393dd86a2a5
# ╟─2b191451-5c29-4ee1-997d-c885b0928cdd
# ╟─0d484210-3a9d-48a9-8350-b554c325e5bb
# ╟─be3c9982-0b47-4d65-b275-6deef8639cf1
# ╟─7515d1f5-0490-417e-808b-28b5902580b5
# ╟─a2fb8b6e-8919-4b1d-931f-ec8ef837a593
# ╟─a8dcdad1-a35f-4a4e-b32c-6542b6f21f37
# ╟─fff26dda-6c92-4135-831e-ab7dc422dcc6
# ╟─32c9945d-a30e-4a95-9065-e867c59ebf06
# ╠═4f2df587-aba9-42e3-b4d0-bf6ef218c0ae
# ╟─b654b8b5-fbcc-4119-ae6c-13ccd55398fb
# ╟─cd331dad-7fe9-4f1e-94db-3d103bc7a350
# ╟─194c07d8-5d78-4051-a647-090c94de7fb4
# ╟─dc28f942-9ba8-45ee-badb-84326b077d62
# ╟─1a6460f9-d7df-42dd-95ca-527233bf2f67
# ╟─ad86f7c4-cce8-446c-bed1-eae4800f1bda
# ╟─29f1f1a6-a542-4792-8825-cd37aee7b7d7
# ╟─c9a2eb7f-5189-4aab-bcb7-6b3e7f50689c
# ╟─a6338962-ef11-479d-8e3c-1975703b1058
# ╟─5e97c7fe-38c1-4a4c-a3f0-f985f981b8aa
# ╟─c429f200-24c0-41b3-8660-a0ebff7cd5a9
# ╟─0f391345-5b71-45a7-b826-bc98c1feef9c
# ╠═6cdc71be-cbc7-11ec-3c09-afe22a13530f
# ╠═663e62c3-480c-4bc3-a11b-f887754f5a5a
# ╟─475c352a-5f0d-40c8-ad74-4a4c81e8d5e1
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
