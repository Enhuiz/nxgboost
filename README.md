#About
NXGBoost, Naive XGBoost, is a naive python implementation of algorithm introduced by Tianqi Chen and Carlos Guestrin in XGBoost: A Scalable Tree Boosting System. This work is the term project of Artificial Intelligence given by Hengjie Song in School of Software Engineering of SCUT.

The code implements a XGBoost with exact greedy split finding method.

There may be problems in this implementation caused by the misunderstanding of the paper. The author of this code is too young, too simple, sometimes want to implement something naive.

#Run
##Enviroment
The code is written in python 2.7.

To run this, the following python package is needed.
- numpy 
- matplotlib
- seaborn 
- sklearn (provide the random foreset method to make a comparision with our nxgboost)

##Run it, some comparision result will be generated into the fig folder.

```
$ python main.py
```