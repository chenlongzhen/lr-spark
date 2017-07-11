import com.autohome.ml.MyUtil
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Hello world!
 *
 */

object lr extends App {

  def indiceChange(sc: SparkContext,path_in :String): RDD[String] = {
    """
    """.stripMargin
    val data = sc.textFile(path_in)
    val train: RDD[String] = data.map {
      line =>
        val segs: Array[String] = line.split(' ')
        val label = if (segs(0) == "1") "1" else "0"
        val features = segs.drop(1)
        // add indices 1
        val features_process: Array[String] = features.map {
          elem =>
            val index = elem.split(":")(0).toInt
            val value = elem.split(":")(1)
            val new_index = index + 1 //index should be begin 1
            //val new_index = index
            new_index.toString + ":" + value
        }
        // sort index
        val features_sort: Array[String] = features_process.sortWith {
          (leftE, rightE) =>
            leftE.split(":")(0).toInt < rightE.split(":")(0).toInt
        }
        val line_arr: Array[String] = label +: features_sort
        // string line
        line_arr.mkString(" ")
    }
    train
  }

  def process_data(sc:SparkContext,path_in:String,ifSplit:Double):Array[RDD[LabeledPoint]]={

    val train: RDD[String] = indiceChange(sc,path_in)
    val util = new MyUtil
    val data: RDD[LabeledPoint] = util.loadLibSVMFile(sc, train,numFeatures = -1).persist(StorageLevel.MEMORY_AND_DISK)
    if (ifSplit > 0 && ifSplit < 1){
      val splitRdd: Array[RDD[LabeledPoint]] = data.randomSplit(Array(10*ifSplit,10*(1-ifSplit)),2017)
      return splitRdd
    }else{
      return Array(data)
    }
  }

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


    // Split data into training (60%) and test (40%).
    logger.info("PROCESS DATA")
    val useData: Array[RDD[LabeledPoint]] = process_data(sc,input_path,0.6)
    val trainData = useData(0).cache()
    val testData = useData(1)

    // Run training algorithm to build the model
    logger.info("TRAIN DATA")
    val model = new LogisticRegressionWithLBFGS()
        .setIntercept(true)
        .setValidateData(true)
        .run(trainData)



    // Compute raw scores on the test set.
    logger.info("predict DATA")
    val predictionAndLabels = testData.map { case LabeledPoint(label, features) =>
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
