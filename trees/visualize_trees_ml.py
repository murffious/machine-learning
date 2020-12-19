# export_graphviz function converts decision tree classifier into dot file and pydotplus convert this dot file to png or displayable form on Jupyter.
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())


# Pros
# Decision trees are easy to interpret and visualize.
# It can easily capture Non-linear patterns.
# It requires fewer data preprocessing from the user, for example, there is no need to normalize columns.
# It can be used for feature engineering such as predicting missing values, suitable for variable selection.
# The decision tree has no assumptions about distribution because of the non-parametric nature of the algorithm. (Source)
# Cons
# Sensitive to noisy data. It can overfit noisy data.
# The small variation(or variance) in data can result in the different decision tree. This can be reduced by bagging and boosting algorithms.
# Decision trees are biased with imbalance dataset, so it is recommended that balance out the dataset before creating the decision tree.
