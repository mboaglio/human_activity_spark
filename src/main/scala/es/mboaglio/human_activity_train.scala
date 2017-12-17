package es.mboaglio

import es.mboaglio.human_activity_predict.vectorizeInput
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

// Heavily inspired on
// https://mapr.com/blog/predicting-loan-credit-risk-using-apache-spark-machine-learning-random-forests/
object human_activity_train extends DatasetUtil {
  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println(
        s"""
           |Usage: human_activity_train <datasource> <model>
           |  <datasource> CSV dataset to learn from
           |  <model> path to save model to
           |
           |  human_activity_train /dataset/human_activity/human_activity_dataset_train.csv /dataset/human_activity.model
        """.stripMargin)
      System.exit(1)
    }

    val Array(datasource, modelPath) = args


    // When using Spark-Shell:
    // implicit val ss = spark
    implicit val spark = SparkSession.
      builder.
      appName("HumanActivity").
      getOrCreate()

    import spark.implicits._

    println("OK/antes de loadtraindata")

    val user_activity_trainDF = loadTrainData(datasource)

    user_activity_trainDF.printSchema
    user_activity_trainDF.show

    println("OK/loadtraindata")

    val dfVector = vectorizeInput(user_activity_trainDF)
    println("OK/vectorize")

    // Convert Strings into Label Identifiers (Double)
    val labelIndexer = new StringIndexer().setInputCol("activity").setOutputCol("label")
    println("OK/labelindexer")

    // Add Label Identifiers field to the DF
    val dfLabeled = labelIndexer.fit(dfVector).transform(dfVector)
    println("OK/transform")

    dfLabeled.select($"features", $"label", $"activity").show(30, false)
    println("OK/select")

    val splitSeed = 5043
    val Array(trainingData, testData) = dfLabeled.randomSplit(Array(0.7, 0.3), splitSeed)

    println("OK/antes classifier")
    val classifier = new RandomForestClassifier().
      setImpurity("gini").
      setMaxDepth(3).
      setNumTrees(20).
      setFeatureSubsetStrategy("auto").
      setSeed(5043)

    println("OK/antes classifier.fit data")
    val model = classifier.fit(trainingData)
    println(model.toDebugString)

    println("=" * 30)
    println("Before pipeline fitting\n")
    val predictions = model.transform(testData)

    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")
    val accuracy = evaluator.evaluate(predictions)
    println(f"Accuracy: $accuracy%2.3f")
    printPredictionMetrics(predictions)


    // Let's try to do better
    val paramGrid = new ParamGridBuilder().
      addGrid(classifier.maxBins, Array(20, 40)).
      addGrid(classifier.maxDepth, Array(2, 10)).
      addGrid(classifier.numTrees, Array(10, 60)).
      addGrid(classifier.impurity, Array("entropy", "gini")).
      build()

    val steps: Array[PipelineStage] = Array(classifier)
    val pipeline = new Pipeline().setStages(steps)

    val cv = new CrossValidator().
      setEstimator(pipeline).
      setEvaluator(evaluator).
      setEstimatorParamMaps(paramGrid).
      setNumFolds(10)

    val pipelineFittedModel = cv.fit(trainingData)

    val predictions2 = pipelineFittedModel.transform(testData)
    val accuracy2 = evaluator.evaluate(predictions2)
    println("=" * 30)
    println("AFTER pipeline fitting\n")
    println(f"Accuracy: $accuracy2%2.3f")

    val bestModel = pipelineFittedModel.bestModel.asInstanceOf[PipelineModel].stages(0)
    val params = bestModel.extractParamMap

    println(
      s"""
         |The best model found was:
         |${bestModel}
         |
        |Using params:
         |${params}
         |
      """.stripMargin)

    printPredictionMetrics(predictions2)


    // Save the model to latter use
    model.write.overwrite().save(modelPath)

  }

  def printPredictionMetrics(predictions: DataFrame)(implicit spark: SparkSession) {
    // Extract PREDICTED and CORRECT (label) values
    import spark.implicits._
    val predictionAndObservations = predictions.select('prediction, 'label)
    val rdd = predictionAndObservations.rdd.map(r => (r.getDouble(0), r.getDouble(1)))

    // Calculate the Quality Metrics
    val rm = new RegressionMetrics(rdd)
    val msg =
      s"""
         |MSE:           ${rm.meanSquaredError}
         |MAE:           ${rm.meanAbsoluteError}
         |RMSE Squared:  ${rm.rootMeanSquaredError}
         |R Squared:     ${rm.r2}
         |Exp. Variance: ${rm.explainedVariance}
         |
      """.stripMargin

    println(msg)
  }
}

