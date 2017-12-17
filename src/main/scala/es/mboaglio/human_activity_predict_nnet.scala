package es.mboaglio

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.{MultilayerPerceptronClassificationModel, MultilayerPerceptronClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator


object human_activity_predict_nnet extends DatasetUtil {

  def main(args: Array[String]): Unit = {
    if (args.length < 2) {
      System.err.println(
        s"""
           |Usage: human_activity_predict_nnet <datasource> <model>
           |  <datasource> CSV dataset to PREDICT activity
           |  <model> path to the model
           |
           |  human_activity_predict_nnet /dataset/human_activity/human_activity_dataset_test.csv /dataset/human_activity.nnet_model
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

    val model = MultilayerPerceptronClassificationModel.load(modelPath)

    println("=" * 30)

    val result = model.transform(dfVector)
    result.select("measure", "prediction").show(10)

    println("Saving Predictions to a CSV file:")

    result.select("prediction","measure", "tBodyAcc_mean_X","tBodyAcc_mean_Y","tBodyAcc_mean_Z","tBodyAcc_std_X",
      "tBodyAcc_std_Y","tBodyAcc_std_Z","tBodyAcc_mad_X","tBodyAcc_mad_Y","tBodyAcc_mad_Z","tBodyAcc_max_X",
      "tBodyAcc_max_Y","tBodyAcc_max_Z","tBodyAcc_min_X","tBodyAcc_min_Y","tBodyAcc_min_Z","tBodyAcc_sma",
      "tBodyAcc_energy_X","tBodyAcc_energy_Y","tBodyAcc_energy_Z","tBodyAcc_iqr_X","tBodyAcc_iqr_Y",
      "tBodyAcc_iqr_Z","tBodyAcc_entropy_X","tBodyAcc_entropy_Y","tBodyAcc_entropy_Z","tBodyAcc_arCoeff_X_1",
      "tBodyAcc_arCoeff_X_2","tBodyAcc_arCoeff_X_3","tBodyAcc_arCoeff_X_4","tBodyAcc_arCoeff_Y_1",
      "tBodyAcc_arCoeff_Y_2","tBodyAcc_arCoeff_Y_3","tBodyAcc_arCoeff_Y_4","tBodyAcc_arCoeff_Z_1",
      "tBodyAcc_arCoeff_Z_2","tBodyAcc_arCoeff_Z_3","tBodyAcc_arCoeff_Z_4","tBodyAcc_correlation_X_Y",
      "tBodyAcc_correlation_X_Z","tBodyAcc_correlation_Y_Z","tGravityAcc_mean_X","tGravityAcc_mean_Y",
      "tGravityAcc_mean_Z","tGravityAcc_std_X","tGravityAcc_std_Y","tGravityAcc_std_Z","tGravityAcc_mad_X",
      "tGravityAcc_mad_Y", "tGravityAcc_mad_Z","tGravityAcc_max_X","tGravityAcc_max_Y","tGravityAcc_max_Z",
      "tGravityAcc_min_X","tGravityAcc_min_Y","tGravityAcc_min_Z","tGravityAcc_sma","tGravityAcc_energy_X",
      "tGravityAcc_energy_Y","tGravityAcc_energy_Z","tGravityAcc_iqr_X","tGravityAcc_iqr_Y","tGravityAcc_iqr_Z",
      "tGravityAcc_entropy_X","tGravityAcc_entropy_Y","tGravityAcc_entropy_Z","tGravityAcc_arCoeff_X_1",
      "tGravityAcc_arCoeff_X_2","tGravityAcc_arCoeff_X_3","tGravityAcc_arCoeff_X_4","tGravityAcc_arCoeff_Y_1",
      "tGravityAcc_arCoeff_Y_2","tGravityAcc_arCoeff_Y_3","tGravityAcc_arCoeff_Y_4","tGravityAcc_arCoeff_Z_1",
      "tGravityAcc_arCoeff_Z_2","tGravityAcc_arCoeff_Z_3","tGravityAcc_arCoeff_Z_4","tGravityAcc_correlation_X_Y",
      "tGravityAcc_correlation_X_Z","tGravityAcc_correlation_Y_Z","tBodyAccJerk_mean_X","tBodyAccJerk_mean_Y",
      "tBodyAccJerk_mean_Z","tBodyAccJerk_std_X","tBodyAccJerk_std_Y","tBodyAccJerk_std_Z","tBodyAccJerk_mad_X",
      "tBodyAccJerk_mad_Y","tBodyAccJerk_mad_Z","tBodyAccJerk_max_X","tBodyAccJerk_max_Y","tBodyAccJerk_max_Z",
      "tBodyAccJerk_min_X","tBodyAccJerk_min_Y","tBodyAccJerk_min_Z","tBodyAccJerk_sma","tBodyAccJerk_energy_X",
      "tBodyAccJerk_energy_Y","tBodyAccJerk_energy_Z","tBodyAccJerk_iqr_X","tBodyAccJerk_iqr_Y","tBodyAccJerk_iqr_Z",
      "tBodyAccJerk_entropy_X","tBodyAccJerk_entropy_Y","tBodyAccJerk_entropy_Z","tBodyAccJerk_arCoeff_X_1",
      "tBodyAccJerk_arCoeff_X_2","tBodyAccJerk_arCoeff_X_3","tBodyAccJerk_arCoeff_X_4","tBodyAccJerk_arCoeff_Y_1",
      "tBodyAccJerk_arCoeff_Y_2","tBodyAccJerk_arCoeff_Y_3","tBodyAccJerk_arCoeff_Y_4","tBodyAccJerk_arCoeff_Z_1",
      "tBodyAccJerk_arCoeff_Z_2","tBodyAccJerk_arCoeff_Z_3","tBodyAccJerk_arCoeff_Z_4","tBodyAccJerk_correlation_X_Y",
      "tBodyAccJerk_correlation_X_Z","tBodyAccJerk_correlation_Y_Z","tBodyGyro_mean_X","tBodyGyro_mean_Y","tBodyGyro_mean_Z",
      "tBodyGyro_std_X","tBodyGyro_std_Y","tBodyGyro_std_Z","tBodyGyro_mad_X","tBodyGyro_mad_Y",
      "tBodyGyro_mad_Z","tBodyGyro_max_X","tBodyGyro_max_Y","tBodyGyro_max_Z","tBodyGyro_min_X",
      "tBodyGyro_min_Y","tBodyGyro_min_Z","tBodyGyro_sma","tBodyGyro_energy_X","tBodyGyro_energy_Y",
      "tBodyGyro_energy_Z","tBodyGyro_iqr_X","tBodyGyro_iqr_Y","tBodyGyro_iqr_Z","tBodyGyro_entropy_X",
      "tBodyGyro_entropy_Y","tBodyGyro_entropy_Z","tBodyGyro_arCoeff_X_1","tBodyGyro_arCoeff_X_2",
      "tBodyGyro_arCoeff_X_3","tBodyGyro_arCoeff_X_4","tBodyGyro_arCoeff_Y_1","tBodyGyro_arCoeff_Y_2",
      "tBodyGyro_arCoeff_Y_3","tBodyGyro_arCoeff_Y_4","tBodyGyro_arCoeff_Z_1","tBodyGyro_arCoeff_Z_2",
      "tBodyGyro_arCoeff_Z_3","tBodyGyro_arCoeff_Z_4","tBodyGyro_correlation_X_Y","tBodyGyro_correlation_X_Z",
      "tBodyGyro_correlation_Y_Z","tBodyGyroJerk_mean_X","tBodyGyroJerk_mean_Y","tBodyGyroJerk_mean_Z",
      "tBodyGyroJerk_std_X","tBodyGyroJerk_std_Y","tBodyGyroJerk_std_Z","tBodyGyroJerk_mad_X",
      "tBodyGyroJerk_mad_Y","tBodyGyroJerk_mad_Z","tBodyGyroJerk_max_X","tBodyGyroJerk_max_Y",
      "tBodyGyroJerk_max_Z","tBodyGyroJerk_min_X","tBodyGyroJerk_min_Y","tBodyGyroJerk_min_Z",
      "tBodyGyroJerk_sma","tBodyGyroJerk_energy_X","tBodyGyroJerk_energy_Y","tBodyGyroJerk_energy_Z",
      "tBodyGyroJerk_iqr_X","tBodyGyroJerk_iqr_Y","tBodyGyroJerk_iqr_Z","tBodyGyroJerk_entropy_X",
      "tBodyGyroJerk_entropy_Y","tBodyGyroJerk_entropy_Z","tBodyGyroJerk_arCoeff_X_1","tBodyGyroJerk_arCoeff_X_2",
      "tBodyGyroJerk_arCoeff_X_3","tBodyGyroJerk_arCoeff_X_4","tBodyGyroJerk_arCoeff_Y_1",
      "tBodyGyroJerk_arCoeff_Y_2","tBodyGyroJerk_arCoeff_Y_3","tBodyGyroJerk_arCoeff_Y_4",
      "tBodyGyroJerk_arCoeff_Z_1","tBodyGyroJerk_arCoeff_Z_2","tBodyGyroJerk_arCoeff_Z_3",
      "tBodyGyroJerk_arCoeff_Z_4","tBodyGyroJerk_correlation_X_Y","tBodyGyroJerk_correlation_X_Z",
      "tBodyGyroJerk_correlation_Y_Z","tBodyAccMag_mean","tBodyAccMag_std","tBodyAccMag_mad","tBodyAccMag_max",
      "tBodyAccMag_min","tBodyAccMag_sma","tBodyAccMag_energy","tBodyAccMag_iqr","tBodyAccMag_entropy",
      "tBodyAccMag_arCoeff1","tBodyAccMag_arCoeff2","tBodyAccMag_arCoeff3","tBodyAccMag_arCoeff4",
      "tGravityAccMag_mean","tGravityAccMag_std","tGravityAccMag_mad","tGravityAccMag_max","tGravityAccMag_min",
      "tGravityAccMag_sma","tGravityAccMag_energy","tGravityAccMag_iqr","tGravityAccMag_entropy",
      "tGravityAccMag_arCoeff1","tGravityAccMag_arCoeff2","tGravityAccMag_arCoeff3","tGravityAccMag_arCoeff4",
      "tBodyAccJerkMag_mean","tBodyAccJerkMag_std","tBodyAccJerkMag_mad","tBodyAccJerkMag_max",
      "tBodyAccJerkMag_min","tBodyAccJerkMag_sma","tBodyAccJerkMag_energy","tBodyAccJerkMag_iqr",
      "tBodyAccJerkMag_entropy","tBodyAccJerkMag_arCoeff1","tBodyAccJerkMag_arCoeff2","tBodyAccJerkMag_arCoeff3",
      "tBodyAccJerkMag_arCoeff4","tBodyGyroMag_mean","tBodyGyroMag_std","tBodyGyroMag_mad","tBodyGyroMag_max",
      "tBodyGyroMag_min","tBodyGyroMag_sma","tBodyGyroMag_energy","tBodyGyroMag_iqr","tBodyGyroMag_entropy",
      "tBodyGyroMag_arCoeff1","tBodyGyroMag_arCoeff2","tBodyGyroMag_arCoeff3","tBodyGyroMag_arCoeff4",
      "tBodyGyroJerkMag_mean","tBodyGyroJerkMag_std","tBodyGyroJerkMag_mad","tBodyGyroJerkMag_max",
      "tBodyGyroJerkMag_min","tBodyGyroJerkMag_sma","tBodyGyroJerkMag_energy","tBodyGyroJerkMag_iqr",
      "tBodyGyroJerkMag_entropy","tBodyGyroJerkMag_arCoeff1","tBodyGyroJerkMag_arCoeff2",
      "tBodyGyroJerkMag_arCoeff3","tBodyGyroJerkMag_arCoeff4","fBodyAcc_mean_X","fBodyAcc_mean_Y",
      "fBodyAcc_mean_Z","fBodyAcc_std_X","fBodyAcc_std_Y","fBodyAcc_std_Z","fBodyAcc_mad_X","fBodyAcc_mad_Y",
      "fBodyAcc_mad_Z","fBodyAcc_max_X","fBodyAcc_max_Y","fBodyAcc_max_Z","fBodyAcc_min_X","fBodyAcc_min_Y",
      "fBodyAcc_min_Z","fBodyAcc_sma","fBodyAcc_energy_X","fBodyAcc_energy_Y","fBodyAcc_energy_Z",
      "fBodyAcc_iqr_X","fBodyAcc_iqr_Y","fBodyAcc_iqr_Z","fBodyAcc_entropy_X","fBodyAcc_entropy_Y",
      "fBodyAcc_entropy_Z","fBodyAcc_maxInds_X","fBodyAcc_maxInds_Y","fBodyAcc_maxInds_Z","fBodyAcc_meanFreq_X",
      "fBodyAcc_meanFreq_Y","fBodyAcc_meanFreq_Z","fBodyAcc_skewness_X","fBodyAcc_kurtosis_X",
      "fBodyAcc_skewness_Y","fBodyAcc_kurtosis_Y","fBodyAcc_skewness_Z","fBodyAcc_kurtosis_Z",
      "fBodyAcc_bandsEnergy_1_8","fBodyAcc_bandsEnergy_9_16","fBodyAcc_bandsEnergy_17_24",
      "fBodyAcc_bandsEnergy_25_32","fBodyAcc_bandsEnergy_33_40","fBodyAcc_bandsEnergy_41_48",
      "fBodyAcc_bandsEnergy_49_56","fBodyAcc_bandsEnergy_57_64","fBodyAcc_bandsEnergy_1_16",
      "fBodyAcc_bandsEnergy_17_32","fBodyAcc_bandsEnergy_33_48","fBodyAcc_bandsEnergy_49_64",
      "fBodyAcc_bandsEnergy_1_24","fBodyAcc_bandsEnergy_25_48","fBodyAcc_bandsEnergy_1_8_1",
      "fBodyAcc_bandsEnergy_9_16_1","fBodyAcc_bandsEnergy_17_24_1","fBodyAcc_bandsEnergy_25_32_1",
      "fBodyAcc_bandsEnergy_33_40_1","fBodyAcc_bandsEnergy_41_48_1","fBodyAcc_bandsEnergy_49_56_1",
      "fBodyAcc_bandsEnergy_57_64_1","fBodyAcc_bandsEnergy_1_16_1","fBodyAcc_bandsEnergy_17_32_1",
      "fBodyAcc_bandsEnergy_33_48_1","fBodyAcc_bandsEnergy_49_64_1","fBodyAcc_bandsEnergy_1_24_1",
      "fBodyAcc_bandsEnergy_25_48_1","fBodyAcc_bandsEnergy_1_8_2","fBodyAcc_bandsEnergy_9_16_2",
      "fBodyAcc_bandsEnergy_17_24_2","fBodyAcc_bandsEnergy_25_32_2","fBodyAcc_bandsEnergy_33_40_2",
      "fBodyAcc_bandsEnergy_41_48_2","fBodyAcc_bandsEnergy_49_56_2","fBodyAcc_bandsEnergy_57_64_2",
      "fBodyAcc_bandsEnergy_1_16_2","fBodyAcc_bandsEnergy_17_32_2","fBodyAcc_bandsEnergy_33_48_2",
      "fBodyAcc_bandsEnergy_49_64_2","fBodyAcc_bandsEnergy_1_24_2","fBodyAcc_bandsEnergy_25_48_2",
      "fBodyAccJerk_mean_X","fBodyAccJerk_mean_Y","fBodyAccJerk_mean_Z","fBodyAccJerk_std_X",
      "fBodyAccJerk_std_Y","fBodyAccJerk_std_Z","fBodyAccJerk_mad_X","fBodyAccJerk_mad_Y","fBodyAccJerk_mad_Z",
      "fBodyAccJerk_max_X","fBodyAccJerk_max_Y","fBodyAccJerk_max_Z","fBodyAccJerk_min_X","fBodyAccJerk_min_Y",
      "fBodyAccJerk_min_Z","fBodyAccJerk_sma","fBodyAccJerk_energy_X","fBodyAccJerk_energy_Y",
      "fBodyAccJerk_energy_Z","fBodyAccJerk_iqr_X","fBodyAccJerk_iqr_Y","fBodyAccJerk_iqr_Z",
      "fBodyAccJerk_entropy_X","fBodyAccJerk_entropy_Y","fBodyAccJerk_entropy_Z","fBodyAccJerk_maxInds_X",
      "fBodyAccJerk_maxInds_Y","fBodyAccJerk_maxInds_Z","fBodyAccJerk_meanFreq_X","fBodyAccJerk_meanFreq_Y",
      "fBodyAccJerk_meanFreq_Z","fBodyAccJerk_skewness_X","fBodyAccJerk_kurtosis_X","fBodyAccJerk_skewness_Y",
      "fBodyAccJerk_kurtosis_Y","fBodyAccJerk_skewness_Z","fBodyAccJerk_kurtosis_Z","fBodyAccJerk_bandsEnergy_1_8",
      "fBodyAccJerk_bandsEnergy_9_16","fBodyAccJerk_bandsEnergy_17_24","fBodyAccJerk_bandsEnergy_25_32",
      "fBodyAccJerk_bandsEnergy_33_40","fBodyAccJerk_bandsEnergy_41_48","fBodyAccJerk_bandsEnergy_49_56",
      "fBodyAccJerk_bandsEnergy_57_64","fBodyAccJerk_bandsEnergy_1_16","fBodyAccJerk_bandsEnergy_17_32",
      "fBodyAccJerk_bandsEnergy_33_48","fBodyAccJerk_bandsEnergy_49_64","fBodyAccJerk_bandsEnergy_1_24",
      "fBodyAccJerk_bandsEnergy_25_48","fBodyAccJerk_bandsEnergy_1_8_1","fBodyAccJerk_bandsEnergy_9_16_1",
      "fBodyAccJerk_bandsEnergy_17_24_1","fBodyAccJerk_bandsEnergy_25_32_1","fBodyAccJerk_bandsEnergy_33_40_1",
      "fBodyAccJerk_bandsEnergy_41_48_1","fBodyAccJerk_bandsEnergy_49_56_1","fBodyAccJerk_bandsEnergy_57_64_1",
      "fBodyAccJerk_bandsEnergy_1_16_1","fBodyAccJerk_bandsEnergy_17_32_1","fBodyAccJerk_bandsEnergy_33_48_1",
      "fBodyAccJerk_bandsEnergy_49_64_1","fBodyAccJerk_bandsEnergy_1_24_1","fBodyAccJerk_bandsEnergy_25_48_1",
      "fBodyAccJerk_bandsEnergy_1_8_2","fBodyAccJerk_bandsEnergy_9_16_2","fBodyAccJerk_bandsEnergy_17_24_2",
      "fBodyAccJerk_bandsEnergy_25_32_2","fBodyAccJerk_bandsEnergy_33_40_2","fBodyAccJerk_bandsEnergy_41_48_2",
      "fBodyAccJerk_bandsEnergy_49_56_2","fBodyAccJerk_bandsEnergy_57_64_2","fBodyAccJerk_bandsEnergy_1_16_2",
      "fBodyAccJerk_bandsEnergy_17_32_2","fBodyAccJerk_bandsEnergy_33_48_2","fBodyAccJerk_bandsEnergy_49_64_2",
      "fBodyAccJerk_bandsEnergy_1_24_2","fBodyAccJerk_bandsEnergy_25_48_2","fBodyGyro_mean_X","fBodyGyro_mean_Y",
      "fBodyGyro_mean_Z","fBodyGyro_std_X","fBodyGyro_std_Y","fBodyGyro_std_Z","fBodyGyro_mad_X","fBodyGyro_mad_Y",
      "fBodyGyro_mad_Z","fBodyGyro_max_X","fBodyGyro_max_Y","fBodyGyro_max_Z","fBodyGyro_min_X","fBodyGyro_min_Y",
      "fBodyGyro_min_Z","fBodyGyro_sma","fBodyGyro_energy_X","fBodyGyro_energy_Y","fBodyGyro_energy_Z",
      "fBodyGyro_iqr_X","fBodyGyro_iqr_Y","fBodyGyro_iqr_Z","fBodyGyro_entropy_X","fBodyGyro_entropy_Y",
      "fBodyGyro_entropy_Z","fBodyGyro_maxInds_X","fBodyGyro_maxInds_Y","fBodyGyro_maxInds_Z",
      "fBodyGyro_meanFreq_X","fBodyGyro_meanFreq_Y","fBodyGyro_meanFreq_Z","fBodyGyro_skewness_X",
      "fBodyGyro_kurtosis_X","fBodyGyro_skewness_Y","fBodyGyro_kurtosis_Y","fBodyGyro_skewness_Z",
      "fBodyGyro_kurtosis_Z","fBodyGyro_bandsEnergy_1_8","fBodyGyro_bandsEnergy_9_16","fBodyGyro_bandsEnergy_17_24",
      "fBodyGyro_bandsEnergy_25_32","fBodyGyro_bandsEnergy_33_40","fBodyGyro_bandsEnergy_41_48",
      "fBodyGyro_bandsEnergy_49_56","fBodyGyro_bandsEnergy_57_64","fBodyGyro_bandsEnergy_1_16",
      "fBodyGyro_bandsEnergy_17_32","fBodyGyro_bandsEnergy_33_48","fBodyGyro_bandsEnergy_49_64",
      "fBodyGyro_bandsEnergy_1_24","fBodyGyro_bandsEnergy_25_48","fBodyGyro_bandsEnergy_1_8_1",
      "fBodyGyro_bandsEnergy_9_16_1","fBodyGyro_bandsEnergy_17_24_1","fBodyGyro_bandsEnergy_25_32_1",
      "fBodyGyro_bandsEnergy_33_40_1","fBodyGyro_bandsEnergy_41_48_1","fBodyGyro_bandsEnergy_49_56_1",
      "fBodyGyro_bandsEnergy_57_64_1","fBodyGyro_bandsEnergy_1_16_1","fBodyGyro_bandsEnergy_17_32_1",
      "fBodyGyro_bandsEnergy_33_48_1","fBodyGyro_bandsEnergy_49_64_1","fBodyGyro_bandsEnergy_1_24_1",
      "fBodyGyro_bandsEnergy_25_48_1","fBodyGyro_bandsEnergy_1_8_2","fBodyGyro_bandsEnergy_9_16_2",
      "fBodyGyro_bandsEnergy_17_24_2","fBodyGyro_bandsEnergy_25_32_2","fBodyGyro_bandsEnergy_33_40_2",
      "fBodyGyro_bandsEnergy_41_48_2","fBodyGyro_bandsEnergy_49_56_2","fBodyGyro_bandsEnergy_57_64_2",
      "fBodyGyro_bandsEnergy_1_16_2","fBodyGyro_bandsEnergy_17_32_2","fBodyGyro_bandsEnergy_33_48_2",
      "fBodyGyro_bandsEnergy_49_64_2","fBodyGyro_bandsEnergy_1_24_2","fBodyGyro_bandsEnergy_25_48_2",
      "fBodyAccMag_mean","fBodyAccMag_std","fBodyAccMag_mad","fBodyAccMag_max","fBodyAccMag_min",
      "fBodyAccMag_sma","fBodyAccMag_energy","fBodyAccMag_iqr","fBodyAccMag_entropy","fBodyAccMag_maxInds",
      "fBodyAccMag_meanFreq","fBodyAccMag_skewness","fBodyAccMag_kurtosis","fBodyBodyAccJerkMag_mean",
      "fBodyBodyAccJerkMag_std","fBodyBodyAccJerkMag_mad","fBodyBodyAccJerkMag_max","fBodyBodyAccJerkMag_min",
      "fBodyBodyAccJerkMag_sma","fBodyBodyAccJerkMag_energy","fBodyBodyAccJerkMag_iqr","fBodyBodyAccJerkMag_entropy",
      "fBodyBodyAccJerkMag_maxInds","fBodyBodyAccJerkMag_meanFreq","fBodyBodyAccJerkMag_skewness",
      "fBodyBodyAccJerkMag_kurtosis","fBodyBodyGyroMag_mean","fBodyBodyGyroMag_std","fBodyBodyGyroMag_mad",
      "fBodyBodyGyroMag_max","fBodyBodyGyroMag_min","fBodyBodyGyroMag_sma","fBodyBodyGyroMag_energy",
      "fBodyBodyGyroMag_iqr","fBodyBodyGyroMag_entropy","fBodyBodyGyroMag_maxInds","fBodyBodyGyroMag_meanFreq",
      "fBodyBodyGyroMag_skewness","fBodyBodyGyroMag_kurtosis","fBodyBodyGyroJerkMag_mean",
      "fBodyBodyGyroJerkMag_std","fBodyBodyGyroJerkMag_mad","fBodyBodyGyroJerkMag_max","fBodyBodyGyroJerkMag_min",
      "fBodyBodyGyroJerkMag_sma","fBodyBodyGyroJerkMag_energy","fBodyBodyGyroJerkMag_iqr",
      "fBodyBodyGyroJerkMag_entropy","fBodyBodyGyroJerkMag_maxInds","fBodyBodyGyroJerkMag_meanFreq",
      "fBodyBodyGyroJerkMag_skewness","fBodyBodyGyroJerkMag_kurtosis","angle_tBodyAccMean_gravity",
      "angle_tBodyAccJerkMean_gravityMean","angle_tBodyGyroMean_gravityMean","angle_tBodyGyroJerkMean_gravityMean",
      "angle_X_gravityMean","angle_Y_gravityMean","angle_Z_gravityMean")
      .write
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("delimiter",";")
      .save(datasource + ".predictions.csv")
  }

}
