# Large-Samples-Testing
Helper functions to facilitate large-samples hypothesis testing.

Student's t-test, Welch's t-test, chi2 contingency and ranksums have been implemented.

Everything else can be added easily by defining custom plot_func and test_func.

If you wish to contribute by implementing further tests and functionalities hit me up! 😊

## Usage
After cloning the repo, first import and define the object: 
```{python}
from tester import Tester
tt = Tester()
```

Then, given an array `X` representing a variable and a binary array `y` representing the controls vs. tested units, create CI_chart of Wilcoxon ranksums tests with:
```{python}
fig, ax = tt.CI_chart(X, y, test_func='ranksums')
```
The function has the following arguments:
  * `X` : `np.ndarray`
      Containing the samples
  * `y` : `np.1darray` (optional)
      For test comparing two samples, binary vector
      differentiating each sample (control vs tested)
  * `sample_size` : `float` in interval (0,1)
      Percentage of instances to be added to the sample at each
      iteration
  * `sample_max` : `float`
      Percentage of instances at last iteration
  * `balance` : `bool`
      Set equal number of samples in controls and tests before 
      testing
  * `test_func` : `str` or Callable
      Function to be specified for testing. Can be 
      [`chi2`, `ranksums`, `ttest`, `welch`] or a user defined function
      (see examples code)
  * `plot_func` : `str` or Callable
      Function to be specified for plotting. Can be user-defined or None. 
      If `test_func` is a string, `plot_func` is automatically overridden.
  * `**plotkwargs` : arguments used for plotting. In the default plot functions,
  they are:
    * `p-limit` : `float`
      The chosen threshold for p-value
    * `labels` : `2-ple`
      Contains the identifiers of controls and tests. Default is `(0, 1)` but an alternative might be `("controls", "tests")`.
    * `title` : `str`
      Plot suptitle

The plot is further modifiable via `fig` and `ax`.

## Todo
Improve `Tester.bootstrap` flexibility.

# References
Lin, Mingfeng & Lucas, Henry & Shmueli, Galit. (2013). Too Big to Fail: Large Samples and the p-Value Problem. Information Systems Research. 24. 906-917. 10.1287/isre.2013.0480. 

Available [here](https://www.researchgate.net/publication/270504262_Too_Big_to_Fail_Large_Samples_and_the_p-Value_Problem).
