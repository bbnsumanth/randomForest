package org.apache.spark.mllib.draftTree.impl

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.draftTree.impl.TreePoint
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint


private[draftTree] class FeaturePoint( val featureIndex:Int, val featureValues: Array[Double])extends Serializable {

}

private[draftTree] object FeaturePoint {

  def convertToFeatureRDD(input: RDD[LabeledPoint]): RDD[FeaturePoint] = {
   
    input.flatMap(x => x.features.toArray.zipWithIndex).map(x => x.swap)
    .groupByKey
    .map(x => new FeaturePoint(x._1,x._2.toArray))
  }
  // need some sorting to maintain the order...do it
}