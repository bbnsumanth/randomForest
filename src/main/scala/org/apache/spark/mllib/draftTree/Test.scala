
package org.apache.spark.mllib.draftTree
import org.apache.spark.mllib.draftTree.DecisionTree
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
object Test {
def main(args: Array[String]) {
val conf = new SparkConf().setAppName("Simple Application")
val sc = new SparkContext(conf)
val data = MLUtils.loadLibSVMFile(sc, "/home/bharath/iris.txt").cache()
// Train a DecisionTree model.
// Empty categoricalFeaturesInfo indicates all features are continuous.
val numClasses = 4
val categoricalFeaturesInfo = Map[Int,Int]()
val impurity = "gini"
val maxDepth = 5
val maxBins = 100
val model = DecisionTree.trainClassifier(data, numClasses, categoricalFeaturesInfo, impurity,maxDepth, maxBins)
val prediction = model.predict(data.map(x => x.features ))
println(prediction)
}
}
// /home/bharath/workspace/mllib/target/scala-2.10/random-forest_2.10-1.0.jar
/**
* script for testing on shell
*/
/*
import org.apache.spark.mllib.draftTree.DecisionTree
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.Vectors
val data = MLUtils.loadLibSVMFile(sc, "/home/bharath/iris.txt").cache()
val numClasses = 4
val categoricalFeaturesInfo = Map[Int,Int]()
val impurity = "entropy"
val maxDepth = 5
val maxBins = 100
val model = DecisionTree.trainClassifier(data, numClasses,categoricalFeaturesInfo, impurity,maxDepth, maxBins)
val labelAndPreds = data.map { point =>
val prediction = model.predict(point.features)
(point.label, prediction)
}
val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / data.count
println("Training Error = " + trainErr)
println("Learned classification tree model:\n" + model)
*/

