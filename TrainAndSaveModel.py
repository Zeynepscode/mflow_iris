# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import argparse
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iris classification arguments")
    
    parser.add_argument("--maxdepth" ,type = int, default = 2,
                        help = "max_depth for decision tree is 2")
    args = parser.parse_args()
    
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=42)

with mlflow.start_run(run_name = "iris decision tree classifier") as run:
    
    #log model params
    
    maxdepth = args.maxdepth
    
   
    mlflow.log_param("max_depth", maxdepth)
    
    
    model = DecisionTreeClassifier(max_depth = maxdepth)
    model.fit(X_train,y_train)
    pred = model.predict(X_test)

    #log model
    
    mlflow.sklearn.log_model(model,"decision tree model")
    
    #log metric
    
    acc = accuracy_score(y_test, pred)
    print("Model Accuracy : {}".format(acc))
    
    mlflow.end_run()
    
    
