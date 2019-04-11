import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'SibSp', 'Parch']

# fetching training Data
data = pd.read_csv("train.csv")
y = data.Survived
X = data[features]

# getting test data
test_data = pd.read_csv("test.csv")
# create test_X which comes from test_data but includes only the columns you used for prediction.
test_X = test_data[features]


X.Sex.replace(to_replace=dict(female=1, male=0), inplace=True)
X.Embarked.replace(to_replace=dict(S=1, C=0, Q=2), inplace=True)

test_X.Sex.replace(to_replace=dict(female=1, male=0), inplace=True)
test_X.Embarked.replace(to_replace=dict(S=1, C=0, Q=2), inplace=True)


# Find some columns with missing values in your dataset.
imputed_X_train_plus = X.copy()
imputed_X_test_plus = test_X.copy()

cols_with_missing = (col for col in X.columns
                                 if X[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()


cols_with_missing_test = (col for col in test_X.columns
                                      if test_X[col].isnull().any())
for col in cols_with_missing_test:
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()



# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

model = GaussianNB()
model.fit(imputed_X_train_plus,y)


# make predictions which we will submit.
test_preds = model.predict(imputed_X_test_plus)

# The lines below shows how to save predictions in format used for competition scoring
# Just uncomment them.

output = pd.DataFrame({'PassengerId': test_data.PassengerId,
                      'Survived': test_preds})
output.to_csv('submission.csv', index=False)

