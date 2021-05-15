# 1. Scipy.Stats https://www.scipy.org/ - ideal for probability function study
# python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose

# Continuous distributions
# Multivariate distributions
# Discrete distributions
# Summary statistics
# Frequency statistics
# Correlation functions
# Statistical tests
# Transformations
# Statistical distances
# Random variate generation
# Circular statistical functions
# Contingency table functions
# Plot-tests
# Masked statistics functions
# Univariate and multivariate kernel density estimation


# Import statistical package from Scipy
from scipy import stats

# Import the normal distribution class
from scipy.stats import norm
print(stats.norm.__doc__)
import matplotlib.pyplot as plt
# Produce 1000 Random Variable following normal distribution
r = norm.rvs(size=1000)
# Plotting the distribution
fig, ax = plt.subplots(1, 1)
ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()


# 2. Pingouin - ideal for simple yet exhaustive stats functions, APIs function you could use for statistical testing

# ANOVA and T-test
# Bayesian
# Circular
# Contingency
# Correlation and regression
# Distribution
# Effect sizes
# Multiple comparisons and post-hoc tests
# Multivariate tests
# Non-parametric
# Others
# Plotting
# Power analysis
# Reliability and consistency

# pip install pingouin
# Import necessary package
import seaborn as sns
import pingouin as pg

mpg = sns.load_dataset('mpg')
pg.anova(data=mpg, dv='mpg', between='origin')

# 3. Statsmodel - to understand statistical modeling in greater depth
# focuses on the statistical estimation based on the data
# statsmodels.api which provide many Cross-sectional models and methods, including Regression and GLM.
# statsmodels.tsa.api Which provide Time-series models and methods.
# statsmodels.formula.api Which provide an interface for specifying
# models using formula strings and DataFrames — in simpler term, you could create your own model.

# endogenous: caused by factors within the system
# exogenous: caused by factors outside the system

# pip install statsmodels
# Importing the necessary package
from sklearn.datasets import load_boston
import statsmodels.api as sm
from statsmodels.api import OLS

# import the data
boston = load_boston()
data = pd.DataFrame(data=boston['data'], columns=boston['feature_names'])
target = pd.Series(boston['target'])
# Develop the model (Ordinary Least Square)
sm_lm = OLS(target, sm.add_constant(data))
result = sm_lm.fit()
result.summary()
# https://towardsdatascience.com/3-top-python-packages-to-learn-statistic-for-data-scientist-d753b76e6099