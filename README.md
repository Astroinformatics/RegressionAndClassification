# Regression & Classification Labs

### Astroinformatics Summer School 2022
### Organized by [Penn State Center for Astrostatistics](https://sites.psu.edu/astrostatistics/)
-----

Suggested Order:
- linear_regression.jl ([Pluto notebook](https://astroinformatics.github.io/RegressionAndClassification/linear_regression.jl.html)):  Introduciton/review of linear regression from a machine learning perspective
- logistics_regression.jl ([Pluto notebook](https://astroinformatics.github.io/RegressionAndClassification/logistics_regression.jl.html)):  Apply Logistic regression to classify High-redshift Quasars

-----
## Running Labs
Instructions will be provided for students to run labs on AWS severs during the summer school.  Below are instruction for running them outside of the summer school.

### Running Pluto notebooks on your local computer
Summer School participants will be provided instructions for accessing a Pluto server.  Others may install Julia and Pluto on their local computer with the following steps:
1.  Download and install current version of Julia from [julialang.org](https://julialang.org/downloads/).
2.  Run julia
3.  From the Julia REPL (command line), type
```julia
julia> using Pkg
julia> Pkg.add("Pluto")
```
Steps 1 & 3 only need to be done once per computer.
4.  Start Pluto
```julia
julia> using Pluto
julia> Pluto.run()
```
5.  Open the Pluto notebook for your lab

-----
## Additional Links
- [GitHub respository](https://github.com/Astroinformatics/SummerSchool2022) for all of Astroinformatics Summer school
- Astroinformatics Summer school [Website & registration](https://sites.psu.edu/astrostatistics/astroinfo-su22/)

## Contributing
We welcome people filing issues and/or pull requests to improve these labs for future summer schools.

