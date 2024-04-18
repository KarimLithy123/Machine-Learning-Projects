from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
df=pd.read_csv('data.csv')
X=df.drop(columns=['FIRST NAME','LAST NAME','CURRENT DATE','DOJ','LEAVES REMAINING','SALARY'])
y=df['SALARY'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)
numeric_features=X.select_dtypes(include=['int64','int32','float64']).columns
cat_features=X.select_dtypes(include=['object']).columns
numeric_transformation = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))])

cat_transformation = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessing = ColumnTransformer(transformers=[
    ('num', numeric_transformation, numeric_features),
    ('cat', cat_transformation, cat_features)
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessing),
    ('regressor', RandomForestRegressor(max_depth=10, min_samples_leaf=4, n_estimators=10))
])
pipeline.fit(X_train, y_train)
pickle.dump(pipeline,open("model.pkl",'wb'))
print(X_test.shape)
