# randomForest
This is a distributed implementation of random forest algo on spark.
This differs from the implemantation of the same algo available in mllib.
In mllib, randomforest algo is implemented by splitting the data instance wise. 
This implementation is by splitting the data feature wise.
This implementation is very useful for the data which has many features . 
I also made few improvisations ,by removing some classes which can be avoided in this approach of implementation.
One important improvement is:  Now ,it is not compulsory for users of randomForest  to give 
categoricalFeatureInfo (information regarding which are continous features ,how many categories a 
categorical feature contains) as input .It is now converted to an Option This implementation automatically 
detects which are continuous features and how many categories a categorical feature contains when the 
categoricalFeatureInfo is given as None in the input from user.
