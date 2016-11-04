/**
 * 
 */
package com.rhl.ml.attrition;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.GBTRegressionModel;
import org.apache.spark.ml.regression.GBTRegressor;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * @author U12044
 *
 */
public class EmployeeAttrition {

	public static void main(String[] args) {

		EmployeeAttrition ea = new EmployeeAttrition();
		ea.execute();
	}

	public void execute() {
		
		SparkSession spark = SparkSession
			      .builder().master("local")
			      .appName("JavaRandomForestClassifierExample")
			      .getOrCreate();
		
		//Dataset<Row> data = spark.read().format("libsvm").load("data/mllib/sample_libsvm_data.txt");
		Dataset<Row> data = spark.read().option("header", true).option("inferSchema", true).csv("D:\\Rahul\\Downloads\\WA_Fn-UseC_-HR-Employee-Attrition.csv");
		data.printSchema();
		
		StringIndexerModel businessTravelIndexer = new StringIndexer()
				  .setInputCol("BusinessTravel")
				  .setOutputCol("BusinessTravelIdx")
				  .fit(data);
		data = businessTravelIndexer.transform(data);
		
		StringIndexerModel departmentIndexer = new StringIndexer()
				  .setInputCol("Department")
				  .setOutputCol("DepartmentIdx")
				  .fit(data);
		data = departmentIndexer.transform(data);
		
		StringIndexerModel maritalStatusIndexer = new StringIndexer()
				  .setInputCol("MaritalStatus")
				  .setOutputCol("MaritalStatusIdx")
				  .fit(data);
		data = maritalStatusIndexer.transform(data);
		
		StringIndexerModel educationFieldIndexer = new StringIndexer()
				  .setInputCol("EducationField")
				  .setOutputCol("EducationFieldIdx")
				  .fit(data);
		data = educationFieldIndexer.transform(data);
		
		StringIndexerModel genderIndexer = new StringIndexer()
				  .setInputCol("Gender")
				  .setOutputCol("GenderIdx")
				  .fit(data);
		data = genderIndexer.transform(data);
		
		StringIndexerModel jobRoleIndexer = new StringIndexer()
				  .setInputCol("JobRole")
				  .setOutputCol("JobRoleIdx")
				  .fit(data);
		data = jobRoleIndexer.transform(data);
		
		StringIndexerModel overTimeIndexer = new StringIndexer()
				  .setInputCol("OverTime")
				  .setOutputCol("OverTimeIdx")
				  .fit(data);
		data = overTimeIndexer.transform(data);
		
		StringIndexerModel labelIndexer = new StringIndexer()
				  .setInputCol("Attrition")
				  .setOutputCol("AttritionIdx")
				  .fit(data);
		data = labelIndexer.transform(data);

				
		String[] vectorFeatures = {"BusinessTravelIdx", "DepartmentIdx", "MaritalStatusIdx", "EducationFieldIdx", "GenderIdx", "JobRoleIdx", "OverTimeIdx", "DailyRate", 
				"DistanceFromHome", "HourlyRate", "MonthlyIncome", "MonthlyRate", "PercentSalaryHike", "TotalWorkingYears"
				, "TrainingTimesLastYear", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"};
		VectorAssembler vectorAssembler = new VectorAssembler()
				.setInputCols(vectorFeatures)
				.setOutputCol("VectorFeatureCol");
		
		data = vectorAssembler.transform(data);
		
		// Automatically identify categorical features, and index them.
		// Set maxCategories so features with > 4 distinct values are treated as continuous.
	/*	VectorIndexerModel featureIndexer = new VectorIndexer()
		  .setInputCol("VectorFeatureCol")
		  .setOutputCol("VectorFeatureColIdx")
		  .setMaxCategories(4)
		  .fit(data);

		data = featureIndexer.transform(data); */
		
		// Train a RandomForest model.
		RandomForestRegressor rf = new RandomForestRegressor()
		  .setLabelCol("AttritionIdx")
		  .setFeaturesCol("VectorFeatureCol")
		  .setFeatureSubsetStrategy("0.3");

		GBTRegressor gbt = new GBTRegressor()
				  .setLabelCol("AttritionIdx")
				  .setFeaturesCol("VectorFeatureCol");
		
		// Convert indexed labels back to original labels.
		IndexToString labelConverter = new IndexToString()
		  .setInputCol("AttritionIdx")
		  .setOutputCol("AttritionLabel");

		
		
		
		ParamMap[] paramMapArray = new ParamGridBuilder()
				.addGrid(rf.maxBins() , new int[]{20, 25})
				.addGrid(rf.maxDepth(), new int[]{4, 6})
				.addGrid(rf.numTrees(), new int[]{100, 200})
				.build();
		
		ParamMap[] gbtParamMapArray = new ParamGridBuilder()
				.addGrid(gbt.maxBins() , new int[]{20})
				.addGrid(gbt.maxDepth(), new int[]{6})
				.addGrid(gbt.maxIter(), new int[]{100})
				.build();
		
		// Chain indexers and forest in a Pipeline
		Pipeline pipeline = new Pipeline()
		  .setStages(new PipelineStage[] { rf, labelConverter});

		Pipeline gbtPipeline = new Pipeline()
				.setStages(new PipelineStage[]{gbt, labelConverter});
		
		// Split the data into training and test sets (30% held out for testing)
		Dataset<Row>[] splits = data.randomSplit(new double[] {0.8, 0.2});
		Dataset<Row> trainingData = splits[0];
		Dataset<Row> testData = splits[1];

		// Select (prediction, true label) and compute test error	
		BinaryClassificationEvaluator bce = new BinaryClassificationEvaluator()
				  .setLabelCol("AttritionIdx")
				  .setRawPredictionCol("prediction")
				.setMetricName("areaUnderROC");
		
		RandomForestRegressionModel rfModel = runRegression(paramMapArray, pipeline, trainingData, testData, bce);
//		GBTRegressionModel gbrModel = runGBRegression(gbtParamMapArray, gbtPipeline, trainingData, testData, bce);
		
		
		/*
		 * Testing
		 */
		Vector inputFeatures = Vectors.dense(2.0, 1.0, 1.0, 0.0, 1.0, 2.0, 2.0, 1567.0, 12.0, 27.0, 1890.0, 18420.0, 14.0, 3.0, 3.0, 2.0, 1.0, 1.0, 1.0 );
		System.out.println(inputFeatures);
		double attritionProbability = rfModel.predict(inputFeatures);
		System.out.println("Attrition Probability - " + attritionProbability);
		
		
		spark.stop();
	}

	private RandomForestRegressionModel runRegression(ParamMap[] paramMapArray, Pipeline pipeline, Dataset<Row> trainingData,
			Dataset<Row> testData, BinaryClassificationEvaluator bce) {
		CrossValidator cv = new CrossValidator()
				.setEstimator(pipeline)
				.setEstimatorParamMaps(paramMapArray)
				.setEvaluator(bce)
				.setNumFolds(10);
		
		CrossValidatorModel model = cv.fit(trainingData);

		// Make predictions.
		Dataset<Row> predictions = model.transform(testData);

		System.out.println("Predictions");
		predictions.printSchema();
		
		// Select example rows to display.
//		predictions.select("prediction", "AttritionLabel", "VectorFeatureCol").show(50);
		
		double accuracy = bce.evaluate(predictions);
		System.out.println("Accuracy = " + accuracy);

		RandomForestRegressionModel rfModel = (RandomForestRegressionModel) ((PipelineModel)model.bestModel()).stages()[0];
		System.out.println("Learned classification forest model:\n" + rfModel.toDebugString());
		
		Vector featureImportanceVector = rfModel.featureImportances();
		System.out.println(featureImportanceVector);
		
		return rfModel;
	}
	
	private GBTRegressionModel runGBRegression(ParamMap[] paramMapArray, Pipeline pipeline, Dataset<Row> trainingData,
			Dataset<Row> testData, BinaryClassificationEvaluator bce) {
		CrossValidator cv = new CrossValidator()
				.setEstimator(pipeline)
				.setEstimatorParamMaps(paramMapArray)
				.setEvaluator(bce)
				.setNumFolds(10);
		
		CrossValidatorModel model = cv.fit(trainingData);

		// Make predictions.
		Dataset<Row> predictions = model.transform(testData);

		System.out.println("Predictions");
		predictions.printSchema();
		
		// Select example rows to display.
//		predictions.select("prediction", "AttritionLabel", "VectorFeatureCol").show(50);
		
		double accuracy = bce.evaluate(predictions);
		System.out.println("Accuracy = " + accuracy);

		GBTRegressionModel gbtModel = (GBTRegressionModel) ((PipelineModel)model.bestModel()).stages()[0];
		System.out.println("Learned classification forest model:\n" + gbtModel.toDebugString());
		
		Vector featureImportanceVector = gbtModel.featureImportances();
		System.out.println(featureImportanceVector);
		
		return gbtModel;
	}
}
