/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.draftTree.impl

import scala.collection.mutable

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.draftTree.configuration.Algo._
import org.apache.spark.mllib.draftTree.configuration.QuantileStrategy._
import org.apache.spark.mllib.draftTree.configuration.Strategy
import org.apache.spark.mllib.draftTree.impurity.Impurity
import org.apache.spark.mllib.draftTree.model.Bin
import org.apache.spark.mllib.draftTree.model.Split
import org.apache.spark.mllib.draftTree.model.DummyCategoricalSplit
import org.apache.spark.mllib.draftTree.model.DummyHighSplit
import org.apache.spark.mllib.draftTree.model.DummyLowSplit
import org.apache.spark.mllib.draftTree.configuration.FeatureType
import org.apache.spark.rdd.RDD
import org.apache.spark.util.random.XORShiftRandom
import org.apache.spark.Logging

/**
 * Learning and dataset metadata for DecisionTree.
 *
 * @param numClasses    For classification: labels can take values {0, ..., numClasses - 1}.
 *                      For regression: fixed at 0 (no meaning).
 * @param maxBins  Maximum number of bins, for all features.
 * @param featureArity  Map: categorical feature index --> arity.
 *                      I.e., the feature takes values in {0, ..., arity - 1}.
 * @param numBins  Number of bins for each feature.
 */
private[draftTree] class DecisionTreeMetadata(
    val numFeatures: Int,
    val numExamples: Long,
    val numClasses: Int,
    val maxBins: Int,
    val featureArity: Map[Int, Int],
    val unorderedFeatures: Set[Int],
    val numBins: Array[Int],
    val impurity: Impurity,
    val quantileStrategy: QuantileStrategy,
    val maxDepth: Int,
    val minInstancesPerNode: Int,
    val minInfoGain: Double,
    val numTrees: Int,
    val numFeaturesPerNode: Int) extends Serializable with Logging {

  def isUnordered(featureIndex: Int): Boolean = unorderedFeatures.contains(featureIndex)
  
  def isClassification: Boolean = numClasses >= 2

  def isMulticlass: Boolean = numClasses > 2

  def isMulticlassWithCategoricalFeatures: Boolean = isMulticlass && (featureArity.size > 0)

  def isCategorical(featureIndex: Int): Boolean = featureArity.contains(featureIndex)

  def isContinuous(featureIndex: Int): Boolean = !featureArity.contains(featureIndex)

  /**
   * Number of splits for the given feature.
   * For unordered features, there are 2 bins per split.
   * For ordered features, there is 1 more bin than split.
   */
  def numSplits(featureIndex: Int): Int = if (isUnordered(featureIndex)) {
    numBins(featureIndex) >> 1
  } else {
    numBins(featureIndex) - 1
  }

  /**
   * Indicates if feature subsampling is being used.
   */
  def subsamplingFeatures: Boolean = numFeatures != numFeaturesPerNode
  
  
  /**
   * Splits and Bins are constructed only for continuous features and Unordered Categorical features
   * for Continuous features...(bins = splits+1)bins are created after creating splits 
   * 
   * for unordered features..splits = 2^(arity - 1) - 1..(bins = 2*splits)
   * but we construct only half of the bins(= splits) since the other half are not used
   * 
   * for ordered feature Bins correspond to feature values, so we do not need to compute splits or bins 
   * beforehand.  Splits are constructed as needed during training.
   */
  
  protected[draftTree] def findSplitsBins(
      input: RDD[LabeledPoint]): (Array[Array[Split]], Array[Array[Bin]]) = {

    logDebug("isMulticlass = " + isMulticlass)

    //val numFeatures:Int = numFeatures

    // ......................Sample the input only if there are continuous features.....................................
    val hasContinuousFeatures = Range(0, numFeatures).exists(isContinuous)
    
    val sampledInput = if (hasContinuousFeatures) {
      
      // Calculate the number of samples for approximate quantile calculation.
      val requiredSamples = math.max(maxBins * maxBins, 10000)
      
      val fraction = if (requiredSamples < numExamples) {
        requiredSamples.toDouble / numExamples
      } else {
        1.0
      }
     logDebug("fraction of data used for calculating quantiles = " + fraction)
      
      input.sample(withReplacement = false, fraction, new XORShiftRandom().nextInt()).collect()
    }              //   
    
    else {
      new Array[LabeledPoint](0)
    }
    //....................................................................................................................
    quantileStrategy match {
      case Sort =>
        val splits = new Array[Array[Split]](numFeatures)
        val bins = new Array[Array[Bin]](numFeatures)

        // Iterate over all features.
        var featureIndex = 0
        while (featureIndex < numFeatures) {
          
          val numOfSplits:Int = numSplits(featureIndex)
          val numOfBins:Int = numBins(featureIndex)//????????????????????????
          //in the creation of numBins ,it is created only for categorical features,
         // but here we are using it for continous festures..???????????????????
         
//continuous feature
          if (isContinuous(featureIndex)) {
            val numSamples = sampledInput.length
            val featureType = FeatureType.Continuous
            splits(featureIndex) = new Array[Split](numOfSplits)
            bins(featureIndex) = new Array[Bin](numOfBins)
            
            val featureSamples = sampledInput.map(lp => lp.features(featureIndex)).sorted
            val stride: Double = numSamples.toDouble / numOfBins
            logDebug("stride = " + stride)
            
            
            //finding Splits
            for (splitIndex <- 0 until numOfSplits) {
              val sampleIndex = splitIndex * stride.toInt
              val threshold = (featureSamples(sampleIndex) + featureSamples(sampleIndex + 1)) / 2.0
              splits(featureIndex)(splitIndex) =
                new Split(featureIndex, threshold, featureType, List())
            }
            //finding bins
            /*
             *  once splits are created,,let us say there are 5 splits fpr this feature...then no.of bins will be 6
             *  out of these first and last bin are created using dummy low split and dummy high split respectively
             *  bins in btww are created using two continuous splits
             *  
             */
            
            //first bin using dummy low split and 1st split
            bins(featureIndex)(0) = new Bin(new DummyLowSplit(featureIndex, featureType),
              splits(featureIndex)(0), featureType, Double.MinValue)
           
            for (splitIndex <- 1 until numOfSplits) {
              bins(featureIndex)(splitIndex) =
                new Bin(splits(featureIndex)(splitIndex - 1), splits(featureIndex)(splitIndex),
                  featureType, Double.MinValue)
            }
            //last bin using last split and dummy high split
            bins(featureIndex)(numOfSplits) = new Bin(splits(featureIndex)(numOfSplits - 1),
              new DummyHighSplit(featureIndex, featureType), featureType, Double.MinValue)
            
          } else {
            
            
// Categorical feature
            val arity = featureArity(featureIndex)
            val featureType = FeatureType.Categorical 
    //Unordered Feature
            
            if (isUnordered(featureIndex)) {
              // TODO: The second half of the bins are unused.  Actually, we could just use
              //       splits and not build bins for unordered features.  That should be part of
              //       a later PR since it will require changing other code (using splits instead
              //       of bins in a few places).
              // Unordered features
              //   2^(maxFeatureValue - 1) - 1 combinations
              
              splits(featureIndex) = new Array[Split](numOfSplits)
              bins(featureIndex) = new Array[Bin](numOfBins)
              
              var splitIndex = 0
              while (splitIndex < numOfSplits) {
                
                  val categories: List[Double] =
                  extractMultiClassCategories(splitIndex + 1, arity)
              
                  splits(featureIndex)(splitIndex) =
                  new Split(featureIndex, Double.MinValue, featureType, categories)
                
                  bins(featureIndex)(splitIndex) = {
                  if (splitIndex == 0) {
                    new Bin(
                      new DummyCategoricalSplit(featureIndex, featureType),
                      splits(featureIndex)(0),
                      featureType,
                      Double.MinValue)
                  } else {
                    new Bin(
                      splits(featureIndex)(splitIndex - 1),
                      splits(featureIndex)(splitIndex),
                      featureType,
                      Double.MinValue)
                  }
                }
                  
                splitIndex += 1
              }
            } 
            
    // Ordered features    
            else {
      
              //   Bins correspond to feature values, so we do not need to compute splits or bins
              //   beforehand.  Splits are constructed as needed during training.
              splits(featureIndex) = new Array[Split](0)
              bins(featureIndex) = new Array[Bin](0)
            }
          }
          featureIndex += 1
        }
        
        
        
        (splits, bins)
      case MinMax =>
        throw new UnsupportedOperationException("minmax not supported yet.")
      case ApproxHist =>
        throw new UnsupportedOperationException("approximate histogram not supported yet.")
    }
  }

  /**
   * Nested method to extract list of eligible categories given an index. It extracts the
   * position of ones in a binary representation of the input. If binary
   * representation of an number is 01101 (13), the output list should (3.0, 2.0,
   * 0.0). The maxFeatureValue depict the number of rightmost digits that will be tested for ones.
   */
  private[draftTree] def extractMultiClassCategories(
      input: Int,
    maxFeatureValue: Int): List[Double] = {
    var categories = List[Double]()
    var j = 0
    var bitShiftedInput = input
    while (j < maxFeatureValue) {
      if (bitShiftedInput % 2 != 0) {
        // updating the list of categories.
        categories = j.toDouble :: categories
      }
      // Right shift by one
      bitShiftedInput = bitShiftedInput >> 1
      j += 1
    }
    categories
  }

}

private[draftTree] object DecisionTreeMetadata {

  /**
   * Construct a [[DecisionTreeMetadata]] instance for this dataset and parameters.
   * This computes which categorical features will be ordered vs. unordered,
   * as well as the number of splits and bins for each feature.
   */
  def buildMetadata(
      input: RDD[LabeledPoint],
      strategy: Strategy,
      featureArity:Map[Int,Int]
      ): DecisionTreeMetadata = {
    
    val numFeatures = input.take(1)(0).features.size
    val numExamples = input.count()
   
    /*
     * for regression problems,number of classe sof label values is 0.
     * for classification number of of classes should be mentioned by the user in strategy
     */
    val numClasses = strategy.algo match {
      case Classification => strategy.numClassesForClassification
      case Regression => 0
    } 
    
    /*
     * Bins can not be more than number of instances
     */
    val maxPossibleBins = math.min(strategy.maxBins, numExamples).toInt
 /*
 * We check the number of bins here against maxPossibleBins for categorical features because 
 * categories of all features should be  less than maxBins.
 * This needs to be checked here instead of in Strategy since maxPossibleBins can be modified
 * based on the number of training examples.
 */
    if (featureArity.nonEmpty) {
      val maxCategoriesPerFeature = featureArity.values.max
      println("###################### maxCategoriesPerFeature :"+ maxCategoriesPerFeature)
      require(maxCategoriesPerFeature <= maxPossibleBins,
        s"DecisionTree requires maxBins (= $maxPossibleBins) >= max categories " +
          s"in categorical features (= $maxCategoriesPerFeature)")
    }

    
    //CREATE AN EMPTY SET FOR UNORDEED FEATURES and ARRAY FOR NUM OF BINS
    val unorderedFeatures = new mutable.HashSet[Int]()
    val numBins = Array.fill[Int](numFeatures)(maxPossibleBins)
    
    //UPDATE unorderedFeatures and numBin
    //for  Multiclass classification
    
    //Note :numBins is created only for categorical features,,,,what abt continuous features?????????
    
    if (numClasses > 2) {
      //multiclass classsification
      val maxCategoriesForUnorderedFeature =
        ((math.log(maxPossibleBins / 2 + 1) / math.log(2.0)) + 1).floor.toInt
        
        
      featureArity.foreach { case (featureIndex, numCategories) =>
        // Decide if some categorical features should be treated as unordered features,
        //  which require 2 * ((1 << numCategories - 1) - 1) bins.
        // We do this check with log values to prevent overflows in case numCategories is large.
        // The next check is equivalent to: 2 * ((1 << numCategories - 1) - 1) <= maxBins
        if (numCategories <= maxCategoriesForUnorderedFeature) {
          unorderedFeatures.add(featureIndex)
          numBins(featureIndex) = numUnorderedBins(numCategories)
        } else {
          numBins(featureIndex) = numCategories
        }
      }
    }
    
    // Binary classification or regression
    //NOTE: for binary classification or regression,unorderedFeatures set is empty,because all CATEGORICAL FEATURES ARE CONSIDERED AS ORDERED
    
    else {
      featureArity.foreach { case (featureIndex, numCategories) =>
        numBins(featureIndex) = numCategories
      }
    }

    /*
     * 
     */
    
    val numFeaturesPerNode: Int = strategy.featureSubsetStrategy match {
      case "all" => numFeatures
      case "sqrt" => math.sqrt(numFeatures).ceil.toInt
      case "log2" => math.max(1, (math.log(numFeatures) / math.log(2)).ceil.toInt)
      case "onethird" => (numFeatures / 3.0).ceil.toInt
      case "auto" =>  if (strategy.numTrees == 1) {
          numFeatures
        } else {
          if (strategy.algo == Classification) {
            math.sqrt(numFeatures).ceil.toInt
          } else {
            (numFeatures / 3.0).ceil.toInt
          }
        }
    }

    new DecisionTreeMetadata(numFeatures, numExamples, numClasses, numBins.max,
      featureArity, unorderedFeatures.toSet, numBins,
      strategy.impurity, strategy.quantileCalculationStrategy, strategy.maxDepth,
      strategy.minInstancesPerNode, strategy.minInfoGain, strategy.numTrees, numFeaturesPerNode)
    

  }

    /**
   * Given the arity of a categorical feature (arity = number of categories),
   */
  /*
   * NOTE FOR UNORDERED FEATURES FIRST NUM OF POSSIBLE SPLITS ARE CALCULATED...DAT IS NUMBER OF COMBINATIONS POSSIBLE...(2^(ARITY- 1) - 1
   * SPLIT PARTITIONS THE CATEGORIES INTO TWO DISJOINT SET,,,DAT IS TWO BINS
   * THEREFORE THERE ARE TWO BINS FOR EVERY SPLIT
   * 
   */
  
  
  def numUnorderedBins(arity: Int): Int = 2 * ((1 << arity - 1) - 1)
  
  

}