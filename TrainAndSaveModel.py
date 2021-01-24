# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import argparse
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iris classification arguments")

    parser.add_argument("--maxdepth",   type = int, default = 2, help = "max_depth for decision tree is 2")
    parser.add_argument("--sepallength",type = float, required = True, help = "Required sepallength")  
    parser.add_argument("--sepalwidth" ,type = float, required = True, help = "Required sepalwidth")      
    parser.add_argument("--petallength",type = float, required = True, help = "Required petallength")      
    parser.add_argument("--petalwidth" ,type = float, required = True, help = "Required petalwidth")
                                                
    args = parser.parse_args()
    
    iris = load_iris()
   
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=42)

    with mlflow.start_run(run_name = "iris decision tree classifier") as run:
        
        #log model params
        
        maxdepth = args.maxdepth
             
        model = DecisionTreeClassifier(max_depth = maxdepth)
        model.fit(X_train,y_train)
        pred = model.predict(X_test)

        #log model
        
        mlflow.sklearn.log_model(model,"decision tree model")
        
        #log metric
        
        acc = accuracy_score(y_test, pred)
        mlflow.log_metric('accuracy',acc)
        print("*"*25)
        print("Model Accuracy : {}".format(acc))
        
        sepallength = args.sepallength
        sepalwidth = args.sepalwidth
        petallength = args.petallength
        petalwidth = args.petalwidth
        print("*"*25)
        
        print("sepallength: {} sepalwidth: {} petalwidth: {} petallength: {}".
            format(sepallength,sepalwidth,petallength,petalwidth))
            
        result = model.predict(np.array([[sepallength,sepalwidth,petallength,petalwidth]]))[0]
        print("Prediction result: ",iris.target_names[result])
        
        #set tag
        
        mlflow.set_tag('prediction' ,iris.target_names[result])
        
        mlflow.end_run()
    
    
