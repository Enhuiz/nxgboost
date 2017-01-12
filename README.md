#About
NXGBoost, standing for Naive XGBoost, is a naive python implementation of model introduced by Tianqi Chen and Carlos Guestrin in XGBoost: A Scalable Tree Boosting System. This work is the term project of Artificial Intelligence given by Hengjie Song in School of Software Engineering of SCUT.

NXGBoost implements a XGBoost with the exact greedy split finding method.

There may be problems in this implementation caused by the misunderstanding of the paper. The author who wrote the code is too young, too simple, sometimes want to implement something naive.

#Run
##Enviroment
The code is written in python 2.7.

To run this, the following python package is needed.

- numpy 
- matplotlib
- seaborn 
- sklearn (provide the random foreset method to make a comparision with our nxgboost)

##Run it

```
$ python main.py
```
Some comparision result will be generated into the fig folder.