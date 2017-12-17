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
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics

// Heavily inspired on
// https://mapr.com/blog/predicting-loan-credit-risk-using-apache-spark-machine-learning-random-forests/

object human_activity_train_nnet extends DatasetUtil {
  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println(
        s"""
           |Usage: human_activity_train_nnet <datasource> <model>
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

    // Load Data
    val user_activity_trainDF = loadTrainData(datasource)

    //Vectorize input
    val dfVector = vectorizeInput(user_activity_trainDF)

    // Convert Strings into Label Identifiers (Double)
    val labelIndexer = new StringIndexer().setInputCol("activity").setOutputCol("label")

    // Add Label Identifiers field to the DF
    val dfLabeled = labelIndexer.fit(dfVector).transform(dfVector)

    // Muestro como quedaron los labels
    dfLabeled.select($"label", $"activity").groupBy("label","activity").count().show()

    // Divido el dataset en 70/30 para entrenar y medir accuracy
    val splitSeed = 5043
    val Array(trainingData, testData) = dfLabeled.randomSplit(Array(0.7, 0.3), splitSeed)

    // Armo la parametrizacion de la Red Neuronal
    // specify layers for the neural network:
    // input layer of size 561 (features), two intermediate of size 50 and 40
    // and output of size 6 (classes)
    val layers = Array[Int](561, 50, 40, 6)

    // Armo el clasificador
    val classifier = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(1000)

    // Entreno el clasificador ... agarrate catalina
    val model = classifier.fit(trainingData)

    // Guardo el modelo
    model.write.overwrite().save(modelPath)

    // Imprimo info sobre el proceso
    println("=" * 30)
    val result = model.transform(testData)

    // Muestro como quedaron las etiquetas de las actividades versus el numero que le asigna el entrenamiento.
    val predictionAndLabels = result.select("prediction", "label")

    // Ahora miro como quedo el modelo ... mido accuracy prediciendo el 30% restante del dataset y lo comparo con
    // lo que se que deberia dar

    // Armo el Multiclass Evaluator
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

    // Calculo e imprimo las metricas usando el evaluator
    val predictions = model.transform(testData)
    val accuracy = evaluator.evaluate(predictions)
    println(f"Accuracy: $accuracy%2.3f")
    printPredictionMetrics(predictions)

    // Armo la matriz de confusion y la imprimo
    println("Confusion Matrix:")
    val confusionMatrix = result.select("prediction","label").groupBy("prediction").pivot("label").count().sort("prediction").na.fill(0).show()

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

