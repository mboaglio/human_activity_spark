package es.mboaglio

import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.sql.SparkSession

object human_activity_predict extends DatasetUtil {

  def main(args: Array[String]): Unit = {
    if (args.length < 2) {
      System.err.println(
        s"""
           |Usage: human_activity_predict <datasource> <model>
           |  <datasource> CSV dataset to PREDICT activity
           |  <model> path to the model
           |
           |  human_activity_predict /dataset/human_activity/human_activity_dataset_test.csv /dataset/human_activity.model
        """.stripMargin)
      System.exit(1)
    }

    val Array(datasource, modelPath) = args

    //    implicit val ss = spark
    implicit val spark = SparkSession.
      builder.
      appName("HumanActivity").
      getOrCreate()

    val df = loadUserInputData(datasource)
    val dfVector = vectorizeInput(df)

    val model = RandomForestClassificationModel.load(modelPath)
    val predictions = model.transform(dfVector)

    import spark.implicits._

    println("=" * 30)
    println("Prediction are:")
//    predictions.select($"userId", $"prediction").show(false)
    predictions.select($"label", $"prediction").show(false)

  }


}
