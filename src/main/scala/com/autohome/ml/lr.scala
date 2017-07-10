package com.autohome.ml

import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
 * Hello world!
 *
 */

object lr extends App {

  override def main(args: Array[String]): Unit = {

    val jobName = "lr_clz"
    val input_path = "/team/ad_wajue/chenlongzhen/mobai/train_file.data"

    // print warn
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val logger = Logger.getLogger("MY LOGGER")

    val conf = new SparkConf().setAppName(jobName)
    conf.set("spark.hadoop.validateOutputSpecs","false")
    conf.set("spark.kryoserializer.buffer.max","2047m")
    val sc: SparkContext = new SparkContext(conf)
    sc.setCheckpointDir("/team/ad_wajue/chenlongzhen/checkpoint")


    val data = MLUtils.loadLibSVMFile(sc,input_path)
    // Split data into training (60%) and test (40%).
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    // Run training algorithm to build the model
    val model = new LogisticRegressionWithLBFGS()
        .setIntercept(true)
        .setValidateData(true)
        .run(training)



    // Compute raw scores on the test set.
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }


    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    val auROC = metrics.areaUnderROC
    logger.info(auROC)
    // Save and load model
    model.save(sc, "target/tmp/scalaLogisticRegressionWithLBFGSModel")
    //val sameModel = LogisticRegressionModel.load(sc,"/tmp")

  }
}
