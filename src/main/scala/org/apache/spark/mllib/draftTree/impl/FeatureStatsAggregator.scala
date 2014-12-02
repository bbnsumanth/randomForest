package org.apache.spark.mllib.draftTree.impl

import org.apache.spark.mllib.draftTree.impurity._
import org.apache.spark.mllib.draftTree.impl._

class FeatureStatsAggregator(
  val metadata: DecisionTreeMetadata,
  val numNodes: Int,
  val featureIndex:Int) extends Serializable {

  val impurityAggregator: ImpurityAggregator = metadata.impurity match {
    case Gini => new GiniAggregator(metadata.numClasses)
    case Entropy => new EntropyAggregator(metadata.numClasses)
    case Variance => new VarianceAggregator()
    case _ => throw new IllegalArgumentException(s"Bad impurity parameter: ${metadata.impurity}")
  }
  
  private val binsForFeature = metadata.numBins(featureIndex)
  
  private val statsSize: Int = impurityAggregator.statsSize//can also use metadata.numClasses
  
  private val featureSize:Int = binsForFeature*statsSize
  
 // this array stores the offsets for every node in the training
 val nodeOffsets:Array[Int] = numNodes match{
    case 1 => Array(0)
    case 2 => Array(0,featureSize)
    case _ => {(2 until numNodes).foldLeft(List(0,featureSize))((a,b) => a :+ a.last+featureSize).toArray}
  }
  
 private val allStatsSize: Int = numNodes*featureSize
 
 private val allStats: Array[Double] = new Array[Double](allStatsSize)
 

 /*For ordered features, this is a pre-computed nodeOffset
   *                           from [[getFeatureOffset]].
   *                           For unordered features, this is a pre-computed
   *                           (node, feature, left/right child) offset from
   *                           [[getLeftRightFeatureOffsets]].
   *                           
   */
 
 def getImpurityCalculator(nodeOffset: Int, binIndex: Int): ImpurityCalculator = {
   impurityAggregator.getCalculator(allStats, nodeOffset + (binIndex) * statsSize)
  }
 //only used for ordered feature
 def update(LocalNodeIndex: Int, binIndex: Int, label: Double, instanceWeight: Double): Unit = {
   
    
    impurityAggregator.update(allStats, nodeOffsets(LocalNodeIndex) + binIndex * statsSize, label, instanceWeight)
  }
 
 /*For ordered features, this is a pre-computed nodeOffset
   *                           from [[getFeatureOffset]].
   *                           For unordered features, this is a pre-computed
   *                           (node, feature, left/right child) offset from
   *                           [[getLeftRightFeatureOffsets]].
   *                           
   */
 
 def nodeUpdate(
      nodeOffset: Int,
      binIndex: Int,
      label: Double,
      instanceWeight: Double): Unit = {
    impurityAggregator.update(allStats, nodeOffset + binIndex * statsSize,
      label, instanceWeight)
  }

  /**
   * Pre-compute feature offset for use with [[featureUpdate]].
   * For unordered features only.
   */
  def getLeftRightNodeOffsets(nodeIndex: Int): (Int, Int) = {
    val baseOffset = nodeOffsets(nodeIndex)
    (baseOffset, baseOffset + (binsForFeature >> 1) * statsSize)
  }
  
   /**
   * Pre-compute node offset for use with [featureUpdate] and getImpurityCalculator methods.
   * For ordered features only.
   */
  def getNodeOffset(nodeIndex: Int): Int = {
    
    nodeOffsets(nodeIndex)
  }
 
  
  /**
   * For a given feature, merge the stats for two bins.
   * @param featureOffset  For ordered features, this is a pre-computed node offset
   *                           from [[getFeatureOffset]].
   *                           For unordered features, this is a pre-computed
   *                           (node, left/right child) offset from
   *                           [[getLeftRightFeatureOffsets]].
   * @param binIndex  The other bin is merged into this bin.
   * @param otherBinIndex  This bin is not modified.
   */
  def mergeForFeature(nodeOffset: Int, binIndex: Int, otherBinIndex: Int): Unit = {
    impurityAggregator.merge(allStats, nodeOffset + binIndex * statsSize,
      nodeOffset + otherBinIndex * statsSize)
  }

  /**
   * Merge this aggregator with another, and returns this aggregator.
   * This method modifies this aggregator in-place.
   */  
  
  
}