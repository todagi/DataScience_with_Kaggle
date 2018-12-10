# basic RandomFroestRegressor from sklearn


```python

# 'feature_1', 'feature_2', 'feature_3'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('./train.csv', parse_dates=['first_active_month'])
test = pd.read_csv("./test.csv", parse_dates=['first_active_month'])
submission = pd.read_csv("./sample_submission.csv", index_col = "card_id")


feature_names = [ 'feature_1', 'feature_2', 'feature_3']
x_train = train[feature_names]
y_train = train['target']
x_test = test[feature_names]

RF_model = RandomForestRegressor( max_features = 'auto', max_depth = None,
                                  n_estimators = 750, 
                                  random_state = 50,
                                  n_jobs = 8)
RF_model.fit(x_train, y_train)
predictions = RF_model.predict(x_test)
predictions = predictions
submission['target'] = predictions
submission.to_csv("RF_model1.csv")
print(submission.shape)
submission.head()


# 3.930
```

## one-hot encoding

```python

# 'feature_1', 'feature_2', 'feature_3' one-hot encoding

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('./data/train.csv', parse_dates=['first_active_month'])
test = pd.read_csv("./data/test.csv", parse_dates=['first_active_month'])
submission = pd.read_csv("./data/sample_submission.csv", index_col = "card_id")

train["feature_11"] = train["feature_1"] == 1
test["feature_11"] = test["feature_1"] == 1
train["feature_12"] = train["feature_1"] == 2
test["feature_12"] = test["feature_1"] == 2
train["feature_13"] = train["feature_1"] == 3
test["feature_13"] = test["feature_1"] == 3
train["feature_14"] = train["feature_1"] == 4
test["feature_14"] = test["feature_1"] == 4
train["feature_15"] = train["feature_1"] == 5
test["feature_15"] = test["feature_1"] == 5

train["feature_21"] = train["feature_2"] == 1
test["feature_21"] = test["feature_2"] == 1
train["feature_22"] = train["feature_2"] == 2
test["feature_22"] = test["feature_2"] == 2
train["feature_23"] = train["feature_2"] == 3
test["feature_23"] = test["feature_2"] == 3

train["feature_31"] = train["feature_3"] == 0
test["feature_31"] = test["feature_3"] == 0
train["feature_32"] = train["feature_3"] == 1
test["feature_32"] = test["feature_3"] == 1

feature_names = [ 'feature_11', 'feature_12', 'feature_13', 'feature_14', 'feature_15',
                  'feature_21', 'feature_22', 'feature_23', 
                  'feature_31','feature_32']

x_train = train[feature_names]
y_train = train['target']
x_test = test[feature_names]

RF_model = RandomForestRegressor( max_features = 'auto', max_depth = None,
                                  n_estimators = 750, 
                                  random_state = 50,
                                  n_jobs = 8)
RF_model.fit(x_train, y_train)
predictions = RF_model.predict(x_test)
predictions = predictions
submission['target'] = predictions
submission.to_csv("RF_model2.csv")
print(submission.shape)
submission.head()

# 3.930
```
