# Human Activity: accelerometer activity recognition
Spark Machine Learning (sparkmllib)
Multilayer Perceptron Classifier
Marcos Boaglio (mboaglio@acm.org)

# Introduccion
El codigo esta escrito en Scala para ser ejecutado en Spark.
Fue probado en Spark version 2.1.2 y Scala version 2.11.8 (OpenJDK 64-Bit Server VM, Java 1.8.0_151)
La clase "human_activity_train_nnet" entrena un modelo Multilayer Perceptron Classifier usando el archivo de entrada
y dejando el modelo grabado. La clase "human_activity_predict_nnet" usa el modelo generado anteriormente y lo aplica al
archivo de entrada para predecir la clase de cada registro.


# Paso 0: Compilar
```bash
sbt clean assembly
```

# Paso 1: human_activity_train_nnet
Se envia el job a spark con el siguiente comando:

```bash
spark-submit \
  --class es.mboaglio.human_activity_train_nnet \
  --master 'spark://master:7077' \
  target/scala-2.11/human_activity-assembly-0.1.jar \
  /dataset/human_activity/human_activity_train.csv \
  /dataset/human_activity.nnet_model
```

El comando entrena y crea un modelo MPC usando el archivo de training human_activity_train.csv y deja el modelo
grabado en /dataset/human_activity.nnet_model
Para hacerlo toma el archivo de entrada y lo divide en 70% para entrenamiento y un 30% para testing.
Entrena el modelo con el 70%, lo usa para predecir la variable dependiente sobre el 30% y compara resultados
para finalmente medir los errores en las predicciones y mostrar metricas asociadas al modelo.

En el camino muestra la siguiente informacion:

a. Mapeo de la actividad (variable dependiente) al label asignado por el MPC

```bash
+-----+------------------+-----+
|label|          activity|count|
+-----+------------------+-----+
|  5.0|WALKING_DOWNSTAIRS| 1055|
|  1.0|          STANDING| 1430|
|  0.0|            LAYING| 1458|
|  2.0|           SITTING| 1333|
|  3.0|           WALKING| 1292|
|  4.0|  WALKING_UPSTAIRS| 1158|
+-----+------------------+-----+
```


b. Metricas del modelo entrenado:

```bash
Accuracy: 0.977
MSE:           0.03373702422145329
MAE:           0.026816608996539794
RMSE Squared:  0.18367641171759996
R Squared:     0.9880915645214257
Exp. Variance: 2.829586084637395
```


c. Matriz de confusion:
```bash
+----------+---+---+---+---+---+---+
|prediction|0.0|1.0|2.0|3.0|4.0|5.0|
+----------+---+---+---+---+---+---+
|       0.0|451|  0|  0|  0|  0|  0|
|       1.0|  0|448| 19|  0|  0|  0|
|       2.0|  0| 23|361|  0|  0|  0|
|       3.0|  0|  0|  0|372|  1|  4|
|       4.0|  0|  0|  0|  2|329|  1|
|       5.0|  0|  0|  0|  4|  0|297|
+----------+---+---+---+---+---+---+
```


# Paso 2: human_activity_predict_nnet
Se submitea el job a spark con el siguiente comando:

```bash
spark-submit \
  --class es.mboaglio.human_activity_predict_nnet \
  --master 'spark://master:7077' \
  target/scala-2.11/human_activity-assembly-0.1.jar \
  /dataset/human_activity/human_activity_test.csv \
  /dataset/human_activity.nnet_model
```

El comando deja en el directorio /dataset/human_activity/human_activity_test.csv.predict.csv los archivos csv con
la misma estructura que el archivo de origen, pero agregando una columna adicional con la prediccion.


# Acerca del dataset
Human Activity Recognition Using Smartphones Dataset / Version 1.0
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
Se encuentra en la carpeta https://github.com/mboaglio/datasets/tree/master/human_activity, transformado y simplificado para poder ser usado.
1. No tiene titulos
2. Esta dividido en dos archivos 70% "train" y 30% llamado "test" (utilizado para predecir)
    a.human_activity_train.csv
    b.human_activity_test.csv
3. La variable dependiente "activity" solo se encuentra en el dataset "train"
4. En el dataset test se agrega un campo adicional "measure" para poder identificar cada registro con una clave

##

# Clasificador Random Forest
 Jobs en scala para entrenar el modelo y para predecir sobre el dataset
 Misma metodologia que el MPC de arriba, pero ojo que esta por la mitad, no funciona bien y el accuracy es muy bajo.

# Entrenar el modelo
```bash
spark-submit \
  --class es.mboaglio.human_activity_train \
  --master 'spark://master:7077' \
  target/scala-2.11/human_activity-assembly-0.1.jar \
  /dataset/human_activity/human_activity_train.csv \
  /dataset/human_activity.model
```

# Clasificar el dataset
```bash
spark-submit \
  --class es.mboaglio.human_activity_predict \
  --master 'spark://master:7077' \
  target/scala-2.11/human_activity-assembly-0.1.jar \
  /dataset/human_activity/human_activity_test.csv \
  /dataset/human_activity.model
```

