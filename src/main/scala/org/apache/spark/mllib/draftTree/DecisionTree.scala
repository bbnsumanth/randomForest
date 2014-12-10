package org.apache.spark.mllib.draftTree

import scala.collection.JavaConverters._
import scala.collection.mutable

import org.apache.spark.annotation.Experimental
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.Logging
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.draftTree.RandomForest.NodeIndexInfo
import org.apache.spark.mllib.draftTree.configuration.Strategy
import org.apache.spark.mllib.draftTree.configuration.Algo._
import org.apache.spark.mllib.draftTree.configuration.FeatureType._
import org.apache.spark.mllib.draftTree.configuration.QuantileStrategy._
import org.apache.spark.mllib.draftTree.impl._
import org.apache.spark.mllib.draftTree.impl.FeatureStatsAggregator
import org.apache.spark.mllib.draftTree.impurity.{ Impurities, Impurity }
import org.apache.spark.mllib.draftTree.impurity._
import org.apache.spark.mllib.draftTree.model._
import org.apache.spark.rdd.RDD
import org.apache.spark.util.random.XORShiftRandom
import org.apache.spark.SparkContext._
import org.apache.spark.Accumulable
import org.apache.spark.AccumulableParam

/**
 * :: Experimental ::
 * A class which implements a decision tree learning algorithm for classification and regression.
 * It supports both continuous and categorical features.
 * @param strategy The configuration parameters for the tree algorithm which specify the type
 *                 of algorithm (classification, regression, etc.), feature type (continuous,
 *                 categorical), depth of the tree, quantile calculation strategy, etc.
 */
@Experimental
class DecisionTree(private val strategy: Strategy) extends Serializable with Logging {

  strategy.assertValid()

  /**
   * Method to train a decision tree model over an RDD
   * @param input Training data: RDD of [[org.apache.spark.mllib.regression.LabeledPoint]]
   * @return DecisionTreeModel that can be used for prediction
   */
  def train(input: RDD[LabeledPoint]): DecisionTreeModel = {
    // Note: random seed will not be used since numTrees = 1.
    val rf = new RandomForest(strategy, seed = 0)
    val rfModel = rf.train(input)
    rfModel.trees(0)
  }

}

object DecisionTree extends Serializable with Logging {

  /**
   * Method to train a decision tree model.
   * The method supports binary and multiclass classification and regression.
   *
   * Note: Using [[org.apache.spark.mllib.tree.DecisionTree$#trainClassifier]]
   *       and [[org.apache.spark.mllib.tree.DecisionTree$#trainRegressor]]
   *       is recommended to clearly separate classification and regression.
   *
   * @param input Training dataset: RDD of [[org.apache.spark.mllib.regression.LabeledPoint]].
   *              For classification, labels should take values {0, 1, ..., numClasses-1}.
   *              For regression, labels are real numbers.
   * @param strategy The configuration parameters for the tree algorithm which specify the type
   *                 of algorithm (classification, regression, etc.), feature type (continuous,
   *                 categorical), depth of the tree, quantile calculation strategy, etc.
   * @return DecisionTreeModel that can be used for prediction
   */
  def train(input: RDD[LabeledPoint], strategy: Strategy): DecisionTreeModel = {
    new DecisionTree(strategy).train(input)
  }

  /**
   * Method to train a decision tree model.
   * The method supports binary and multiclass classification and regression.
   *
   * Note: Using [[org.apache.spark.mllib.tree.DecisionTree$#trainClassifier]]
   *       and [[org.apache.spark.mllib.tree.DecisionTree$#trainRegressor]]
   *       is recommended to clearly separate classification and regression.
   *
   * @param input Training dataset: RDD of [[org.apache.spark.mllib.regression.LabeledPoint]].
   *              For classification, labels should take values {0, 1, ..., numClasses-1}.
   *              For regression, labels are real numbers.
   * @param algo algorithm, classification or regression
   * @param impurity impurity criterion used for information gain calculation
   * @param maxDepth Maximum depth of the tree.
   *                 E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.
   * @return DecisionTreeModel that can be used for prediction
   */
  def train(
    input: RDD[LabeledPoint],
    algo: Algo,
    impurity: Impurity,
    maxDepth: Int): DecisionTreeModel = {
    val strategy = new Strategy(algo, impurity, maxDepth)
    new DecisionTree(strategy).train(input)
  }

  /**
   * Method to train a decision tree model.
   * The method supports binary and multiclass classification and regression.
   *
   * Note: Using [[org.apache.spark.mllib.tree.DecisionTree$#trainClassifier]]
   *       and [[org.apache.spark.mllib.tree.DecisionTree$#trainRegressor]]
   *       is recommended to clearly separate classification and regression.
   *
   * @param input Training dataset: RDD of [[org.apache.spark.mllib.regression.LabeledPoint]].
   *              For classification, labels should take values {0, 1, ..., numClasses-1}.
   *              For regression, labels are real numbers.
   * @param algo algorithm, classification or regression
   * @param impurity impurity criterion used for information gain calculation
   * @param maxDepth Maximum depth of the tree.
   *                 E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.
   * @param numClassesForClassification number of classes for classification. Default value of 2.
   * @return DecisionTreeModel that can be used for prediction
   */
  def train(
    input: RDD[LabeledPoint],
    algo: Algo,
    impurity: Impurity,
    maxDepth: Int,
    numClassesForClassification: Int): DecisionTreeModel = {
    val strategy = new Strategy(algo, impurity, maxDepth, numClassesForClassification)
    new DecisionTree(strategy).train(input)
  }

  /**
   * Method to train a decision tree model.
   * The method supports binary and multiclass classification and regression.
   *
   * Note: Using [[org.apache.spark.mllib.tree.DecisionTree$#trainClassifier]]
   *       and [[org.apache.spark.mllib.tree.DecisionTree$#trainRegressor]]
   *       is recommended to clearly separate classification and regression.
   *
   * @param input Training dataset: RDD of [[org.apache.spark.mllib.regression.LabeledPoint]].
   *              For classification, labels should take values {0, 1, ..., numClasses-1}.
   *              For regression, labels are real numbers.
   * @param algo classification or regression
   * @param impurity criterion used for information gain calculation
   * @param maxDepth Maximum depth of the tree.
   *                 E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.
   * @param numClassesForClassification number of classes for classification. Default value of 2.
   * @param maxBins maximum number of bins used for splitting features
   * @param quantileCalculationStrategy  algorithm for calculating quantiles
   * @param categoricalFeaturesInfo Map storing arity of categorical features.
   *                                E.g., an entry (n -> k) indicates that feature n is categorical
   *                                with k categories indexed from 0: {0, 1, ..., k-1}.
   * @return DecisionTreeModel that can be used for prediction
   */
  def train(
    input: RDD[LabeledPoint],
    algo: Algo,
    impurity: Impurity,
    maxDepth: Int,
    numClassesForClassification: Int,
    maxBins: Int,
    quantileCalculationStrategy: QuantileStrategy,
    categoricalFeaturesInfo: Map[Int, Int]): DecisionTreeModel = {
    val strategy = new Strategy(algo, impurity, maxDepth, numClassesForClassification, maxBins,
      quantileCalculationStrategy, categoricalFeaturesInfo)
    new DecisionTree(strategy).train(input)
  }

  /**
   * Method to train a decision tree model for binary or multiclass classification.
   *
   * @param input Training dataset: RDD of [[org.apache.spark.mllib.regression.LabeledPoint]].
   *              Labels should take values {0, 1, ..., numClasses-1}.
   * @param numClassesForClassification number of classes for classification.
   * @param categoricalFeaturesInfo Map storing arity of categorical features.
   *                                E.g., an entry (n -> k) indicates that feature n is categorical
   *                                with k categories indexed from 0: {0, 1, ..., k-1}.
   * @param impurity Criterion used for information gain calculation.
   *                 Supported values: "gini" (recommended) or "entropy".
   * @param maxDepth Maximum depth of the tree.
   *                 E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.
   *                  (suggested value: 5)
   * @param maxBins maximum number of bins used for splitting features
   *                 (suggested value: 32)
   * @return DecisionTreeModel th
   * }at can be used for prediction
   */
  def trainClassifier(
    input: RDD[LabeledPoint],
    numClassesForClassification: Int,
    categoricalFeaturesInfo: Map[Int, Int],
    impurity: String,
    maxDepth: Int,
    maxBins: Int): DecisionTreeModel = {
    val impurityType = Impurities.fromString(impurity)
    train(input, Classification, impurityType, maxDepth, numClassesForClassification, maxBins, Sort,
      categoricalFeaturesInfo)
  }

  /**
   * Java-friendly API for [[org.apache.spark.mllib.tree.DecisionTree$#trainClassifier]]
   */
  def trainClassifier(
    input: JavaRDD[LabeledPoint],
    numClassesForClassification: Int,
    categoricalFeaturesInfo: java.util.Map[java.lang.Integer, java.lang.Integer],
    impurity: String,
    maxDepth: Int,
    maxBins: Int): DecisionTreeModel = {
    trainClassifier(input.rdd, numClassesForClassification,
      categoricalFeaturesInfo.asInstanceOf[java.util.Map[Int, Int]].asScala.toMap,
      impurity, maxDepth, maxBins)
  }

  /**
   * Method to train a decision tree model for regression.
   *
   * @param input Training dataset: RDD of [[org.apache.spark.mllib.regression.LabeledPoint]].
   *              Labels are real numbers.
   * @param categoricalFeaturesInfo Map storing arity of categorical features.
   *                                E.g., an entry (n -> k) indicates that feature n is categorical
   *                                with k categories indexed from 0: {0, 1, ..., k-1}.
   * @param impurity Criterion used for information gain calculation.
   *                 Supported values: "variance".
   * @param maxDepth Maximum depth of the tree.
   *                 E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.
   *                  (suggested value: 5)
   * @param maxBins maximum number of bins used for splitting features
   *                 (suggested value: 32)
   * @return DecisionTreeModel that can be used for prediction
   */
  def trainRegressor(
    input: RDD[LabeledPoint],
    categoricalFeaturesInfo: Map[Int, Int],
    impurity: String,
    maxDepth: Int,
    maxBins: Int): DecisionTreeModel = {
    val impurityType = Impurities.fromString(impurity)
    train(input, Regression, impurityType, maxDepth, 0, maxBins, Sort, categoricalFeaturesInfo)
  }

  /**
   * Java-friendly API for [[org.apache.spark.mllib.tree.DecisionTree$#trainRegressor]]
   */
  def trainRegressor(
    input: JavaRDD[LabeledPoint],
    categoricalFeaturesInfo: java.util.Map[java.lang.Integer, java.lang.Integer],
    impurity: String,
    maxDepth: Int,
    maxBins: Int): DecisionTreeModel = {
    trainRegressor(input.rdd,
      categoricalFeaturesInfo.asInstanceOf[java.util.Map[Int, Int]].asScala.toMap,
      impurity, maxDepth, maxBins)
  }

  /*
     * The high-level descriptions of the best split optimizations are noted here.
     *
     * *Group-wise training*
     * We perform bin calculations for groups of nodes to reduce the number of
     * passes over the data.  Each iteration requires more computation and storage,
     * but saves several iterations over the data.
     *
     * *Bin-wise computation*
     * We use a bin-wise best split computation strategy instead of a straightforward best split
     * computation strategy. Instead of analyzing each sample for contribution to the left/right
     * child node impurity of every split, we first categorize each feature of a sample into a
     * bin. We exploit this structure to calculate aggregates for bins and then use these aggregates
     * to calculate information gain for each split.
     *
     * *Aggregation over partitions*
     * Instead of performing a flatMap/reduceByKey operation, we exploit the fact that we know
     * the number of splits in advance. Thus, we store the aggregates (at the appropriate
     * indices) in a single array for all bins and rely upon the RDD aggregate method to
     * drastically reduce the communication overhead.
     */

  /**
   * Given a group of nodes, this finds the best split for each node.
   *
   * @param input Training data: RDD of [[org.apache.spark.mllib.tree.impl.TreePoint]]
   * @param metadata Learning and dataset metadata
   * @param topNodes Root node for each tree.  Used for matching instances with nodes.
   * @param nodesForGroup Mapping: treeIndex --> nodes to be split in tree
   * @param treeToNodeToIndexInfo Mapping: treeIndex --> nodeIndex --> nodeIndexInfo,
   *                              where nodeIndexInfo stores the index in the group and the
   *                              feature subsets (if using feature subsets).
   * @param splits possible splits for all features, indexed (numFeatures)(numSplits)
   * @param bins possible bins for all features, indexed (numFeatures)(numBins)
   * @param nodeQueue  Queue of nodes to split, with values (treeIndex, node).
   *                   Updated with new non-leaf nodes which are created.
   */

  /**
   * this method should update the treeModel and also return the a Map:(treeIndex-->Map(globalNodeIndex-->(Split,Node))
   * which is used for updating NodeInstanceMatrix.
   */

  private[draftTree] def findBestSplits(
    input: RDD[FeaturePoint],
    metadata: DecisionTreeMetadata,
    topNodes: Array[Node],
    nodesForGroup: Map[Int, Array[Node]],
    treeToNodeToIndexInfo: Map[Int, Map[Int, NodeIndexInfo]],
    splits: Array[Array[Split]],
    bins: Array[Array[Bin]],
    nodeQueue: mutable.Queue[(Int, Node)],
    label: Array[Double],
    weightMatrix: Array[Array[Int]],
    nodeInstanceMatrix: Array[Array[Int]],
    timer: TimeTracker = new TimeTracker): Map[Int, Option[Map[Int, Node]]] = {

    val numNodes = nodesForGroup.values.map(_.size).sum

    logDebug("numNodes = " + numNodes)
    logDebug("numFeatures = " + metadata.numFeatures)
    logDebug("numClasses = " + metadata.numClasses)
    logDebug("isMulticlass = " + metadata.isMulticlass)
    logDebug("isMulticlassWithCategoricalFeatures = " + metadata.isMulticlassWithCategoricalFeatures)

    /**
     * Get map:{node index in group --> Array[features indices]}
     * which is a short cut to find feature indices for a node given node index(local node index) in group
     * @param treeToNodeToIndexInfo
     * @return
     */
    def getNodeToFeatures(treeToNodeToIndexInfo: Map[Int, Map[Int, NodeIndexInfo]]): Option[Map[Int, Array[Int]]] = if (!metadata.subsamplingFeatures) {
      None
    } else {
      val mutableNodeToFeatures = new mutable.HashMap[Int, Array[Int]]()
      treeToNodeToIndexInfo.values.foreach { nodeIdToNodeInfo =>
        nodeIdToNodeInfo.values.foreach { nodeIndexInfo =>
          assert(nodeIndexInfo.featureSubset.isDefined)
          mutableNodeToFeatures(nodeIndexInfo.nodeIndexInGroup) = nodeIndexInfo.featureSubset.get
        }
      }
      Some(mutableNodeToFeatures.toMap)
    }

    /**
     * Get a Map:(treeIndex->Array(nodeIndex of nodes in the group))
     *
     */

    def getTreeToGlobaNodeId(nodesForGroup: Map[Int, Array[Node]]): Map[Int, Option[Array[Int]]] = {

      val mutableNodeToFeatures = scala.collection.mutable.Map[Int, Option[Array[Int]]]()

      (0 to metadata.numTrees - 1).foreach(treeId =>
        nodesForGroup.contains(treeId) match {
          case true => mutableNodeToFeatures += (treeId -> Some(nodesForGroup(treeId).map(x => x.id)))
          case false => mutableNodeToFeatures += (treeId -> None)
        })
      mutableNodeToFeatures.toMap
    }

    /**
     * it takes nodeToBestSplit: Map[Int,(Split, InformationGainStats, Predict)]
     * and gives back TreeToGlobalIndexToSplit:Map[Int,Map[Int,Node]]
     * This is used for updating the nodeInstanceMatrix
     * this is called at the end of training of a group of nodes after updating them
     */

    def getTreeToGlobalIndexToSplit(
      nodesForGroup: Map[Int, Array[Node]]): Map[Int, Option[Map[Int, Node]]] = {

      val treeToGlobalIndexToSplit = scala.collection.mutable.Map[Int, Option[Map[Int, Node]]]()

      (0 to metadata.numTrees - 1).foreach(treeId =>
        nodesForGroup.contains(treeId) match {
          case true => {
            val nodesForTree = nodesForGroup(treeId)
            val globalIndexToSplit = scala.collection.mutable.Map[Int, Node]()
            nodesForTree.foreach { node =>
              val nodeIndex = node.id
              globalIndexToSplit += (nodeIndex -> node)

            }
            treeToGlobalIndexToSplit += (treeId -> Some(globalIndexToSplit.toMap))
          }
          case false => treeToGlobalIndexToSplit += (treeId -> None)
        })
      treeToGlobalIndexToSplit.toMap

    }

    /**
     * Calculate the information gain for a given (feature, split) based upon left/right aggregates.
     * @param leftImpurityCalculator left node aggregates for this (feature, split)
     * @param rightImpurityCalculator right node aggregate for this (feature, split)
     * @return information gain and statistics for split
     */
    def calculateGainForSplit(
      leftImpurityCalculator: ImpurityCalculator,
      rightImpurityCalculator: ImpurityCalculator,
      metadata: DecisionTreeMetadata): InformationGainStats = {

      val leftCount = leftImpurityCalculator.count
      val rightCount = rightImpurityCalculator.count

      // If left child or right child doesn't satisfy minimum instances per node,
      // then this split is invalid, return invalid information gain stats.
      if ((leftCount < metadata.minInstancesPerNode) ||
        (rightCount < metadata.minInstancesPerNode)) {
        return InformationGainStats.invalidInformationGainStats
      }

      val totalCount = leftCount + rightCount

      val parentNodeAgg = leftImpurityCalculator.copy
      parentNodeAgg.add(rightImpurityCalculator)

      val impurity = parentNodeAgg.calculate()

      val leftImpurity = leftImpurityCalculator.calculate() // Note: This equals 0 if count = 0
      val rightImpurity = rightImpurityCalculator.calculate()

      val leftWeight = leftCount / totalCount.toDouble
      val rightWeight = rightCount / totalCount.toDouble

      val gain = impurity - leftWeight * leftImpurity - rightWeight * rightImpurity

      val leftPredict = new Predict(leftImpurityCalculator.predict,
        leftImpurityCalculator.prob(leftImpurityCalculator.predict))
      val rightPredict = new Predict(rightImpurityCalculator.predict,
        rightImpurityCalculator.prob(rightImpurityCalculator.predict))

      // if information gain doesn't satisfy minimum information gain,
      // then this split is invalid, return invalid information gain stats.
      if (gain < metadata.minInfoGain) {
        return InformationGainStats.invalidInformationGainStats
      }

      new InformationGainStats(gain, impurity, leftImpurity, rightImpurity, leftPredict, rightPredict)
    }

    /**
     * Calculate predict value for current node, given stats of any split.
     * Note that this function is called only once for each node.
     * @param leftImpurityCalculator left node aggregates for a split
     * @param rightImpurityCalculator right node aggregates for a split
     * @return predict value for current node
     */

    def calculatePredict(
      leftImpurityCalculator: ImpurityCalculator,
      rightImpurityCalculator: ImpurityCalculator): Predict = {
      val parentNodeAgg = leftImpurityCalculator.copy
      parentNodeAgg.add(rightImpurityCalculator)
      val predict = parentNodeAgg.predict
      val prob = parentNodeAgg.prob(predict)

      new Predict(predict, prob)
    }

    /**
     * It return the nodeToBestSplit:Map[Int,(Split, InformationGainStats, Predict)] for a feature
     * is a map: LocalnodeIndex-->(what is the split for this feature , the info gain for that split,Predict of that)
     */

    def NodeToFeatureSplit(
      binAggregates: FeatureStatsAggregator,
      splits: Array[Split],
      featureIndex: Int): Map[Int, (Split, InformationGainStats, Predict)] = {

      val nodeToFeatureSplit = new mutable.HashMap[Int, (Split, InformationGainStats, Predict)]()

      val numSplits = binAggregates.metadata.numSplits(featureIndex)

      val numBins = binAggregates.metadata.numBins(featureIndex)

      //*************************** for continuous feature ***************************************
      if (binAggregates.metadata.isContinuous(featureIndex)) {

        Range(0, binAggregates.numNodes).map { nodeIndex =>

          val nodeOffset = binAggregates.getNodeOffset(nodeIndex)

          //merging Bins for gain calculations for all splits for a given node
          var splitIndex = 0
          while (splitIndex < numSplits) {
            binAggregates.mergeForFeature(nodeOffset, splitIndex + 1, splitIndex)
            splitIndex += 1
          }

          val (bestFeatureSplitIndex: Int, bestFeatureGainStats: InformationGainStats) = {
            Range(0, numSplits).map {
              case splitIdx =>
                //TAKE ONE SPLIT AND CALCULATE LEFT AND RIGHT STATS
                val leftChildStats = binAggregates.getImpurityCalculator(nodeOffset, splitIdx)
                val rightChildStats = binAggregates.getImpurityCalculator(nodeOffset, numSplits).subtract(leftChildStats)
                //find the gainStats
                val gainStats = calculateGainForSplit(leftChildStats, rightChildStats, binAggregates.metadata)
                (splitIdx, gainStats)
            }.maxBy(_._2.gain)
          }

          //USING the LEFT AND RIGHT STATS for the best split find  PREDICT FOR THAT SPLIT. 
          val leftChildStats = binAggregates.getImpurityCalculator(nodeOffset, bestFeatureSplitIndex)
          val rightChildStats = binAggregates.getImpurityCalculator(nodeOffset, numSplits).subtract(leftChildStats)
          val predict = calculatePredict(leftChildStats, rightChildStats)
          /*
          println("#############################################################################################################################")
          println("For LocalNodeIndex: " + nodeIndex + " best gain for the feature:" + featureIndex +
            " is " + bestFeatureGainStats.gain + " and the predict is: " + predict.predict + " with prob:" + predict.prob)
          println("##############################################################################################################################")
          * 
          */

          //Update the map for this node  
          nodeToFeatureSplit(nodeIndex) = (splits(bestFeatureSplitIndex), bestFeatureGainStats, predict)
        }
      } else if (binAggregates.metadata.isUnordered(featureIndex)) {

        //************************************unordered features*************************************
        Range(0, binAggregates.numNodes).map { nodeIndex =>

          val (leftChildOffset, rightChildOffset) = binAggregates.getLeftRightNodeOffsets(nodeIndex)

          val (bestFeatureSplitIndex: Int, bestFeatureGainStats: InformationGainStats) = {
            Range(0, numSplits).map {
              case splitIdx =>
                val leftChildStats = binAggregates.getImpurityCalculator(leftChildOffset, splitIdx)
                val rightChildStats = binAggregates.getImpurityCalculator(rightChildOffset, splitIdx)
                val gainStats = calculateGainForSplit(leftChildStats, rightChildStats, binAggregates.metadata)
                (splitIdx, gainStats)
            }.maxBy(_._2.gain)
          }

          //USING the LEFT AND RIGHT STATS for the best split find  PREDICT FOR THAT SPLIT. 
          val leftChildStats = binAggregates.getImpurityCalculator(leftChildOffset, bestFeatureSplitIndex)
          val rightChildStats = binAggregates.getImpurityCalculator(rightChildOffset, bestFeatureSplitIndex)
          val predict = calculatePredict(leftChildStats, rightChildStats)
          /*
          println("#############################################################################################################################")
          println("For loaalNodeIndex: " + nodeIndex + " best gain for the feature:" + featureIndex +
            " is " + bestFeatureGainStats.gain + " and the predict is: " + predict.predict + " with prob:" + predict.prob)
          println("##############################################################################################################################")
          * 
          */

          //Update the map for this node  
          nodeToFeatureSplit(nodeIndex) = (splits(bestFeatureSplitIndex), bestFeatureGainStats, predict)
        }
      } else {
        //************************************ordered features***************************************

        Range(0, binAggregates.numNodes).map { nodeIndex =>

          val nodeOffset = binAggregates.getNodeOffset(nodeIndex)

          /*Each bin is one category (feature value).
          * The bins are ordered based on centroidForCategories, and this ordering determines which
          * splits are considered. (With K categories, we consider K - 1 possible splits.)
          * centroidForCategories is a list: (category, centroid)
          */
          val centroidForCategories = if (binAggregates.metadata.isMulticlass) {
            // For categorical variables in multiclass classification,
            // the bins are ordered by the impurity of their corresponding labels.
            Range(0, numBins).map {
              case featureValue =>
                val categoryStats = binAggregates.getImpurityCalculator(nodeOffset, featureValue)
                val centroid = if (categoryStats.count != 0) {
                  categoryStats.calculate()
                } else {
                  Double.MaxValue
                }
                (featureValue, centroid)
            }
          } else {
            // regression or binary classification
            // For categorical variables in regression and binary classification,
            // the bins are ordered by the centroid of their corresponding labels. 
            Range(0, numBins).map {
              case featureValue =>
                val categoryStats = binAggregates.getImpurityCalculator(nodeOffset, featureValue)
                val centroid = if (categoryStats.count != 0) {
                  categoryStats.predict
                } else {
                  Double.MaxValue
                }
                (featureValue, centroid)
            }
          }

          logDebug("Centroids for categorical variable: " + centroidForCategories.mkString(","))
          // bins sorted by centroids
          val categoriesSortedByCentroid = centroidForCategories.toList.sortBy(_._2)
          logDebug("Sorted centroids for categorical variable = " + categoriesSortedByCentroid.mkString(","))

          // Cumulative sum (scanLeft) of bin statistics.
          // Afterwards, binAggregates for a bin is the sum of aggregates for
          // that bin + all preceding bins.
          var splitIndex = 0
          while (splitIndex < numSplits) {
            val currentCategory = categoriesSortedByCentroid(splitIndex)._1
            val nextCategory = categoriesSortedByCentroid(splitIndex + 1)._1
            binAggregates.mergeForFeature(nodeOffset, nextCategory, currentCategory)
            splitIndex += 1
          }

          // lastCategory = index of bin with total aggregates for this (node, feature)
          val lastCategory = categoriesSortedByCentroid.last._1

          val (bestFeatureSplitIndex: Int, bestFeatureGainStats) = {
            Range(0, numSplits).map { splitIndex =>
              val featureValue = categoriesSortedByCentroid(splitIndex)._1
              val leftChildStats = binAggregates.getImpurityCalculator(nodeOffset, featureValue)
              val rightChildStats = binAggregates.getImpurityCalculator(nodeOffset, lastCategory).subtract(leftChildStats)
              val gainStats = calculateGainForSplit(leftChildStats, rightChildStats, binAggregates.metadata)
              (splitIndex, gainStats)
            }.maxBy(_._2.gain)
          }

          val categoriesForSplit = categoriesSortedByCentroid.map(_._1.toDouble).slice(0, bestFeatureSplitIndex + 1)
          val bestFeatureSplit = new Split(featureIndex, Double.MinValue, Categorical, categoriesForSplit)

          //USING the LEFT AND RIGHT STATS for the best split find  PREDICT FOR THAT SPLIT. 
          val leftChildStats = binAggregates.getImpurityCalculator(
            nodeOffset, categoriesSortedByCentroid(bestFeatureSplitIndex)._1)
          val rightChildStats = binAggregates.getImpurityCalculator(nodeOffset, lastCategory).subtract(leftChildStats)
          val predict = calculatePredict(leftChildStats, rightChildStats)
          /*
          println("#############################################################################################################################")
          println("For localNodeIndex:" + nodeIndex + "best gain for the feature:" + featureIndex +
            " is " + bestFeatureGainStats.gain + "predict is:" + predict.predict + "with prob:" + predict.prob)
          println("##############################################################################################################################")
          * 
          */

          //Update the map for this node  
          nodeToFeatureSplit(nodeIndex) = (bestFeatureSplit, bestFeatureGainStats, predict)
        }
      }

      //return the map
      nodeToFeatureSplit.toMap
    }

    /**
     * Main code for finding the splits for nodes in the group
     *
     */
    timer.start("chooseSplits")

    val nodeToFeatures = getNodeToFeatures(treeToNodeToIndexInfo)
    val nodeToFeaturesBc = input.sparkContext.broadcast(nodeToFeatures)
    val treeToGlobalNodeId = getTreeToGlobaNodeId(nodesForGroup)

    /**
     * creates a featureStatsAggregator for a particular ORDERED feature
     * and updated the featureStatsAggregator for all the nodes in the training for a given feature.
     *
     */
    def orderedBinSeqOp(
      featurePoint: FeaturePoint): FeatureStatsAggregator = {

      //create a FSA for this feature

      val featureStatsAggregator = new FeatureStatsAggregator(metadata, numNodes, featurePoint.featureIndex)

      val totalFeatures = metadata.numFeatures
      var instanceId = 0
      while (instanceId < metadata.numExamples) {

        (0 to metadata.numTrees - 1).foreach { treeId =>
          val GlobalNodeIndex = nodeInstanceMatrix(treeId)(instanceId)
          treeToGlobalNodeId(treeId) match {
            case Some(nodesForTree) => {
              if (nodesForTree.contains(GlobalNodeIndex)) {
                val nodeInfo = treeToNodeToIndexInfo(treeId).getOrElse(GlobalNodeIndex, null)
                val localNodeIndex = nodeInfo.nodeIndexInGroup
                val featuresForNode = nodeInfo.featureSubset.getOrElse(Range(0, totalFeatures).toArray)
                val instanceWeight = weightMatrix(treeId)(instanceId)
                if (featuresForNode.contains(featurePoint.featureIndex)) {
                  val offset = featureStatsAggregator.getNodeOffset(localNodeIndex)
                  featureStatsAggregator.nodeUpdate(offset, featurePoint.featureValues(instanceId).toInt,
                    label(instanceId), instanceWeight: Double)
                }
              }
            }
            case None => {}
          }
        }
        instanceId += 1
      }

      featureStatsAggregator
    }

    /**
     * creates a featureStatsAggregator for a particular UNORDERED feature
     * and updated the featureStatsAggregator for all the nodes in the training for a given feature.
     *
     */

    def unorderedBinSeqOp(
      featurePoint: FeaturePoint,
      bins: Array[Bin] //bins for a particular feature
      ): FeatureStatsAggregator = {

      //create a FSA for this feature

      val featureStatsAggregator = {
        new FeatureStatsAggregator(metadata, numNodes, featurePoint.featureIndex)
      }
      val totalFeatures = metadata.numFeatures

      var instanceId = 0
      while (instanceId < metadata.numExamples) {

        (0 to metadata.numTrees - 1).foreach { treeId =>
          val GlobalNodeIndex = nodeInstanceMatrix(treeId)(instanceId)

          treeToGlobalNodeId(treeId) match {
            case Some(nodesForTree) => {
              if (nodesForTree.contains(GlobalNodeIndex)) {
                val nodeInfo = treeToNodeToIndexInfo(treeId).getOrElse(GlobalNodeIndex, null)
                val localNodeIndex = nodeInfo.nodeIndexInGroup
                val featuresForNode = nodeInfo.featureSubset.getOrElse(Range(0, totalFeatures).toArray)
                val instanceWeight = weightMatrix(treeId)(instanceId)

                if (featuresForNode.contains(featurePoint.featureIndex)) {
                  val featureValue = featurePoint.featureValues(instanceId)
                  val (leftNodeOffset, rightNodeOffset) = featureStatsAggregator.getLeftRightNodeOffsets(localNodeIndex)
                  val numSplits = featureStatsAggregator.metadata.numSplits(featurePoint.featureIndex)

                  (0 to numSplits - 1).foreach { splitIndex =>
                    bins(splitIndex).highSplit.categories.contains(featureValue) match {
                      case true => featureStatsAggregator.nodeUpdate(leftNodeOffset, splitIndex, label(instanceId),
                        instanceWeight)
                      case false => featureStatsAggregator.nodeUpdate(rightNodeOffset, splitIndex, label(instanceId),
                        instanceWeight)
                    }
                  }

                }
              }

            }

            case None => {}
          }

        }
        instanceId += 1
      }

      featureStatsAggregator
    }

    /**
     * creates a featureStatsAggregator for a particular continuous feature
     * and update the featureStatsAggregator for all the nodes in the training for a given feature.
     * it requires finding binIndices for every featureValue because we no longer use treePoint
     */
    def continuousBinSeqOp(
      featurePoint: FeaturePoint,
      bins: Array[Bin] //bins for a particular feature
      ): FeatureStatsAggregator = {

      //create a FSA for this feature
      val featureStatsAggregator = {
        new FeatureStatsAggregator(metadata, numNodes, featurePoint.featureIndex)
      }

      //helper method to find the bin index given its featureValue and bins for that feature
      def binarySearchForBins(featureValue: Double): Int = {
        var left = 0
        var right = bins.length - 1
        while (left <= right) {
          val mid = left + (right - left) / 2
          val bin = bins(mid)
          val lowThreshold = bin.lowSplit.threshold
          val highThreshold = bin.highSplit.threshold
          if ((lowThreshold < featureValue) && (highThreshold >= featureValue)) {
            return mid
          } else if (lowThreshold >= featureValue) {
            right = mid - 1
          } else {
            left = mid + 1
          }
        }
        -1
      }

      val totalFeatures = metadata.numFeatures

      var instanceId = 0
      while (instanceId < metadata.numExamples) {
        // find the bin index for this featureValue using binary search on bins
        val binIndex = binarySearchForBins(featurePoint.featureValues(instanceId))
        (0 to metadata.numTrees - 1).foreach { treeId =>

          val GlobalNodeIndex = nodeInstanceMatrix(treeId)(instanceId)

          treeToGlobalNodeId(treeId) match {
            case Some(nodesForTree) => {
              if (nodesForTree.contains(GlobalNodeIndex)) {
                val nodeInfo = treeToNodeToIndexInfo(treeId).getOrElse(GlobalNodeIndex, null)
                val localNodeIndex = nodeInfo.nodeIndexInGroup
                val featuresForNode = nodeInfo.featureSubset.getOrElse(Range(0, totalFeatures).toArray)
                val instanceWeight = weightMatrix(treeId)(instanceId)

                if (featuresForNode.contains(featurePoint.featureIndex)) {
                  val offset = featureStatsAggregator.getNodeOffset(localNodeIndex)
                  featureStatsAggregator.nodeUpdate(offset, binIndex,
                    label(instanceId), instanceWeight: Double)

                }
              }

            }
            case None => {}
          }

        }
        instanceId += 1
      }

      featureStatsAggregator
    }

    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& MAP PROGRAM &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&  
    /*
     * this RDD contains each element as MAP:LocalnodeIndex-->(split,gainStats)
     * split is nothing but the best split for that feature.
     * 
     */

    val nodeToSplitRDD = input.map { x =>
      {

        // crete an FSA for all nodes in training and update it for that feature
        //need to pass bins for unordered features because we have to update half of the bins(that is for every split we have to update one bin) 
        //while updating stats for unordered features
        // bins are requied to find binIndex for a feature value for continous features

        val updatedStats: FeatureStatsAggregator = metadata.isContinuous(x.featureIndex) match {
          case true => continuousBinSeqOp(x, bins(x.featureIndex))
          case false => {
            metadata.isUnordered(x.featureIndex) match {
              case true => unorderedBinSeqOp(x, bins(x.featureIndex))
              case false => orderedBinSeqOp(x)
            }
          }
        }

        // now calculate the best split for every node in that feature
        /*
          * this is a Map:LocalnodeIndex-->(split,gainStat,predict)
          * 
          */

        NodeToFeatureSplit(updatedStats, splits(x.featureIndex), x.featureIndex)

      }

    }

    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& END OF MAP PROGRAM &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&     

    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& REDUCE PROGRAM     &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&     

    /*
     * now we have to reduce this RDD to MAP:LocalnodeIndex-->(bestSplit,gainStats,predict)
     */

    val nodeToBestSplit = nodeToSplitRDD.reduce { (a, b) =>

      val temp = scala.collection.mutable.Map[Int, (Split, InformationGainStats, Predict)]()

      a.foreach {
        case (k, v) =>

          temp += (k -> List(v, b(k)).maxBy(_._2.gain))
      }

      temp.toMap

    }
    /*
    println("###############################################################################################")
    println(" finding best splits completed for this group of training")
    println("nodeToBestSplit map:LocalnodeIndex-->(bestSplit,gainStats,predict) is : " + nodeToBestSplit)
    println("###############################################################################################")
    * 
    */

    timer.stop("chooseSplits")
    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& END OF REDUCE PROGRAM &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&       

    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& DRIVER PROGRAM     &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&       

    /*
      *this is the driver program to update evry tree after findng best splits for every node
      * update every node in the grp with relevent info like split,predict,is leaf etc 
      * then create child nodes for every node...these nodes are used for next level of training
      */

    nodesForGroup.foreach {
      case (treeIndex, nodesForTree) =>
        nodesForTree.foreach { node =>
          val nodeIndex = node.id
          val nodeInfo = treeToNodeToIndexInfo(treeIndex)(nodeIndex)
          val localNodeIndex = nodeInfo.nodeIndexInGroup
          val (split: Split, stats: InformationGainStats, predict: Predict) =
            nodeToBestSplit(localNodeIndex)
          logDebug("best split = " + split)

          // Extract info for this node.  Create children if not leaf.
          val isLeaf = (stats.gain <= 0) || (Node.indexToLevel(nodeIndex) == metadata.maxDepth)
          assert(node.id == nodeIndex)
          node.predict = predict
          node.isLeaf = isLeaf
          node.stats = Some(stats)
          logDebug("Node = " + node)

          if (!isLeaf) {
            node.split = Some(split)
            node.leftNode = Some(Node.emptyNode(Node.leftChildIndex(nodeIndex)))
            node.rightNode = Some(Node.emptyNode(Node.rightChildIndex(nodeIndex)))
            nodeQueue.enqueue((treeIndex, node.leftNode.get))
            nodeQueue.enqueue((treeIndex, node.rightNode.get))
            logDebug("leftChildIndex = " + node.leftNode.get.id +
              ", impurity = " + stats.leftImpurity)
            logDebug("rightChildIndex = " + node.rightNode.get.id +
              ", impurity = " + stats.rightImpurity)
          }
        }
    }
    /*
      * return the Map:[Int,Map[Int,(Split,Node)]]
      * this is needed for updating the nodeInstanceMatrix
      * 
      */

    val treeToGlobalIndexToSplit = getTreeToGlobalIndexToSplit(nodesForGroup)
    /*
    println("###############################################################################################")
    println("driver program completed for this group ..nodes in groups  are updated with splits")
    println("treeToGlobalIndexToSplit Map:[treeIndex,Map[GlobalNodeIndex,Node]]: " + treeToGlobalIndexToSplit(0))
    println("###############################################################################################")
    * 
    */

    treeToGlobalIndexToSplit // this is returned from findBestSplits method

    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& END OF DRIVER PROGRAM &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&   

  }
  //.............................................end of findBestSplit.....................................................  

  /**
   * this method is  called from ranfdomForest to update the nodeInstanceMatrix
   * treeToGlobalIndexToSplit:Map[featureIndex -> (NodeIndex, Split)]
   */

  private[draftTree] def updateNodeInstanceMatrix(input: RDD[FeaturePoint],
    treeToGlobalIndexToSplit: Map[Int, Option[Map[Int, Node]]],
    nodeInstanceMatrix: Array[Array[Int]],
    accumalator: Accumulable[Array[Array[Int]], (Int, Int, Int)],
    metadata: DecisionTreeMetadata): Array[Array[Int]] = {

    input.foreach { x =>
      /*
      var instanceId = 0
      while (instanceId < metadata.numExamples) {
      * 
      */
      (0L to metadata.numExamples - 1).foreach { instanceId =>

        (0 to metadata.numTrees - 1).foreach { treeId =>

          val globalNodeIndex = nodeInstanceMatrix(treeId)(instanceId.toInt)

          treeToGlobalIndexToSplit(treeId) match {
            //if some nodes of this tree are in this group of training
            case Some(nodesForTree) => {

              nodesForTree.contains(globalNodeIndex) match {
                //if  this node is present in this group of training
                case true => {
                  val node = nodesForTree(globalNodeIndex)
                  val split = nodesForTree(globalNodeIndex).split.getOrElse(null)

                  node.isLeaf match {
                    case true => {
                      accumalator += (globalNodeIndex, treeId, instanceId.toInt)
                    }
                    case false => {
                      if (split.feature == x.featureIndex) {

                        val featureValue = x.featureValues(instanceId.toInt)

                        val updatedNodeIndex: Int = split.featureType match {
                          case Continuous => if (featureValue < split.threshold) { Node.leftChildIndex(node.id) } else { Node.rightChildIndex(node.id) }
                          case Categorical => if (split.categories.contains(featureValue)) { Node.leftChildIndex(node.id) } else { Node.rightChildIndex(node.id) }

                        }
                        accumalator += (updatedNodeIndex, treeId, instanceId.toInt)
                        /*   
                      if(globalNodeIndex > 1000 ) {
                       
                        println("########################################################################################")
                        println("initial index: " + globalNodeIndex)
                        println("featureIndex: " + x.featureIndex)
                        println("treeId: " + treeId)
                        println("instanceId: " + instanceId)
                        println("updated Index: " + updatedNodeIndex)

                        
                      } 
                      * 
                      */

                      }
                    }
                  }
                }
                //if this node is not present in this group of training
                case false => {
                  accumalator += (globalNodeIndex, treeId, instanceId.toInt)
                }

              }
            }
            //if no nodes of this tree are in this group of training
            case None => {

              accumalator += (globalNodeIndex, treeId, instanceId.toInt)

            }

          }
        }

      }

    } // end of rdd operation

    accumalator.value

  }
  //...........................................end of updateNodeInstanceMatrix

}
//.............................................end of decisionTree Object...............................................

  

  
  

