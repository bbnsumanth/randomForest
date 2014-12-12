
package org.apache.spark.mllib.draftTree
import org.apache.spark.mllib.draftTree.DecisionTree
import org.apache.spark.mllib.draftTree.RandomForest
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.util.Utils._
import org.apache.spark.util._
object Test {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Simple Application")
    val sc = new SparkContext(conf)
    val initialData = MLUtils.loadLibSVMFile(sc, "/home/bharath/iris.txt").cache()
    val classCounts = initialData.map(_.label).countByValue()
    val sortedClasses = classCounts.keys.toList.sorted
    val numClasses = classCounts.size
    val classIndexMap = sortedClasses.zipWithIndex.toMap
    val data = initialData.map(lp => LabeledPoint(classIndexMap(lp.label), lp.features))
    val numExamples = data.count()
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val featureSubsetStrategy = "auto"
    val numTrees = 10
    val maxDepth = 10
    val maxBins = 100
    val seed = Utils.random.nextInt()
    val testingData = MLUtils.loadLibSVMFile(sc, "/home/bharath/iris.txt").cache()
    
    if (numTrees == 1) {
      val model = DecisionTree.trainClassifier(data, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)
       if (model.numNodes < 300) {
        println(model.toDebugString) // Print full model.
      } else {
        println(model) // Print model summary.
      }
    
      
    val labelAndPreds = data.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val labelAndPredsForTesting = testingData.map{point =>
      val prediction = model.predict(point.features)
      (point.label, sortedClasses(prediction.toInt))
      
    }
    
    val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / data.count
    val testingErr = labelAndPredsForTesting.filter(r => r._1 != r._2).count.toDouble / testingData.count
    println("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    println("Training Error = " + trainErr)
    println("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")  
    println("Testing Error = " + testingErr)
    println("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    
    
    } else {
      
      val model = RandomForest.trainClassifier(data, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy,
        impurity, maxDepth, maxBins, seed)
        
        if (model.totalNumNodes < 300) {
        println(model.toDebugString) // Print full model.
      } else {
        println(model) // Print model summary.
      }
      
      
    val labelAndPreds = data.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
     val labelAndPredsForTesting = testingData.map{point =>
      val prediction = model.predict(point.features)
      (point.label, sortedClasses(prediction.toInt))
      
    }
    val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / data.count
    val testingErr = labelAndPredsForTesting.filter(r => r._1 != r._2).count.toDouble / testingData.count
    
    println("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    println("Training Error = " + trainErr)
    println("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    println("Testing Error = " + testingErr)
    println("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    }
    

  }
}
// /home/bharath/workspace/mllib/target/scala-2.10/random-forest_2.10-1.0.jar
/**
* script for testing on shell
*/
/*
import org.apache.spark.mllib.draftTree.DecisionTree
import org.apache.spark.mllib.draftTree.RandomForest
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.util.Utils
import org.apache.spark.util._

    val initialData = MLUtils.loadLibSVMFile(sc, "/home/bharath/data/iris.txt").cache()
    val classCounts = initialData.map(_.label).countByValue()
    val sortedClasses = classCounts.keys.toList.sorted
    val numClasses = classCounts.size
    val classIndexMap = sortedClasses.zipWithIndex.toMap
    val data = initialData.map(lp => LabeledPoint(classIndexMap(lp.label), lp.features))
    val numExamples = data.count()
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val featureSubsetStrategy = "auto"
    val numTrees = 10
    val maxDepth = 10
    val maxBins = 100
    val seed = 3
    val testingData = MLUtils.loadLibSVMFile(sc, "/home/bharath/data/iris.txt").cache()
    
    if (numTrees == 1) {
      val model = DecisionTree.trainClassifier(data, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)
       if (model.numNodes < 3000) {
        println(model.toDebugString) // Print full model.
      } else {
        println(model) // Print model summary.
      }
    
      
    val labelAndPreds = data.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val labelAndPredsForTesting = testingData.map{point =>
      val prediction = model.predict(point.features)
      (point.label, sortedClasses(prediction.toInt))
      
    }
    
    val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / data.count
    val testingErr = labelAndPredsForTesting.filter(r => r._1 != r._2).count.toDouble / testingData.count
    println("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    println("Training Error = " + trainErr)
    println("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")  
    println("Testing Error = " + testingErr)
    println("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    
    } else {
      
      val model = RandomForest.trainClassifier(data, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy,
        impurity, maxDepth, maxBins, seed)
        
        if (model.totalNumNodes < 3000) {
        println(model.toDebugString) // Print full model.
      } else {
        println(model) // Print model summary.
      }
      
      
    val labelAndPreds = data.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
     val labelAndPredsForTesting = testingData.map{point =>
      val prediction = model.predict(point.features)
      (point.label, sortedClasses(prediction.toInt))
      
    }
    val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / data.count
    val testingErr = labelAndPredsForTesting.filter(r => r._1 != r._2).count.toDouble / testingData.count
    
    println("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    println("Training Error = " + trainErr)
    println("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    println("Testing Error = " + testingErr)
    println("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    }
    




*/

