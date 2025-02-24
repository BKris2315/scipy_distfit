# Distribution fitter with scipy

## Motivation and description

While there are several great options when it comes to data fitting, for example [distfit package](https://erdogant.github.io/distfit/pages/html/index.html), which can handle best-fit selection and are automatized, they often have shortcomings that the functions that can be fitted are hidden.

`distfit` excels in the task of binning and fitting with the best distribution from `scipy.stats` a given data series, but the objective functions are frozen objects from `scipy` and there is no possibility to:
- give a custom function for the distribution
- give custom binning
- fit already binned data

There is always the possibility of using `scipy.optimize.curve_fit` to overcome this, but it is annoying to implement the fitting pipeline every time. Here I implemented in `numpy` the 80 well-known distributions from `distfit`'s `distr=full` option and created a very minimalistic pipeline to call a `curve_fit`. I also included a script for `normal`, `logspace`, and `custom logarithmic` data binning. The benefit of this minimalistic approach is that the list of functions is easy to expand and the existing distribution functions can be accessed and modified at will.

I still encourage the use of `distfit` whenever it suits your problem as it is for sure a more complex and neater approach, but for custom logic, this small repo can be useful.

## Example usage

```
Some exapmle code will bee added here Soon TM
```

### Creating and installing it as a package

```
Walkthrough will be added Soon TM
```
