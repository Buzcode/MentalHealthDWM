import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.SerializationHelper;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.instance.SMOTE;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;

import java.io.File;
import java.util.Random;

public class FeatureSelectionAndPersistence {

    public static void main(String[] args) {
        try {
            System.out.println("==================================================");
            System.out.println("       MEMBER 2: FEATURE SELECTION, BALANCING & PERSISTENCE PIPELINE         ");
            System.out.println("==================================================\n");

            String datasetPath = "clean_mental_health.arff";
            File file = new File(datasetPath);
            if (!file.exists()) {
                System.out.println("Error: Dataset file not found -> " + datasetPath);
                return;
            }

            DataSource source = new DataSource(datasetPath);
            Instances rawData = source.getDataSet();

            int targetIdx = rawData.attribute("has_mental_health_condition_nominal").index();
            rawData.setClassIndex(targetIdx);

            System.out.println("Step 1: Loading Cleaned Dataset");
            System.out.println("Total Rows: " + rawData.numInstances());
            System.out.println("Total Attributes: " + rawData.numAttributes() + " (24 input features + 1 target class)");
            System.out.println("Target Class: " + rawData.classAttribute().name() + " {No, Yes}");

            System.out.println("\nStep 2: Evaluating Baseline Random Forest (All 24 Features)...");
            RandomForest baselineModel = new RandomForest();
            Evaluation evalBaseline = evaluateModel(baselineModel, rawData);

            System.out.printf("Baseline Accuracy : %.2f%%\n", evalBaseline.pctCorrect());
            System.out.printf("Baseline F1-Score : %.4f\n", evalBaseline.weightedFMeasure());
            System.out.printf("Baseline ROC-AUC  : %.4f\n", evalBaseline.weightedAreaUnderROC());

            System.out.println("\nStep 3: Feature Selection using Information Gain...");
            InfoGainAttributeEval infoGain = new InfoGainAttributeEval();
            Ranker ranker = new Ranker();
            ranker.setThreshold(0.0);

            infoGain.buildEvaluator(rawData);
            int[] rankedIndices = ranker.search(infoGain, rawData);

            int topN = 10;
            System.out.println("Top " + topN + " Features Ranked by Information Gain:");
            for (int i = 0; i < topN && i < rankedIndices.length; i++) {
                int attIdx = rankedIndices[i];
                double score = infoGain.evaluateAttribute(attIdx);
                System.out.printf("  %2d. %-30s (InfoGain: %.4f)\n", (i + 1), rawData.attribute(attIdx).name(), score);
            }

            AttributeSelection filter = new AttributeSelection();
            filter.setEvaluator(infoGain);
            Ranker searchMethod = new Ranker();
            searchMethod.setNumToSelect(topN);
            filter.setSearch(searchMethod);
            filter.setInputFormat(rawData);

            Instances selectedData = Filter.useFilter(rawData, filter);
            System.out.println("Reduced attributes from " + rawData.numAttributes() + " to " + selectedData.numAttributes() + " (10 features + 1 class).");

            RandomForest modelTop10 = new RandomForest();
            Evaluation evalTop10 = evaluateModel(modelTop10, selectedData);

            System.out.println("\nStep 4: Handling Class Imbalance with SMOTE...");
            int[] counts = selectedData.attributeStats(selectedData.classIndex()).nominalCounts;
            int countNo = counts[0];
            int countYes = counts[1];

            int minCount = Math.min(countNo, countYes);
            int majCount = Math.max(countNo, countYes);
            double smotePercentage = ((double) (majCount - minCount) / minCount) * 100.0;

            System.out.println("Minority Class Count: " + minCount);
            System.out.println("Majority Class Count: " + majCount);
            System.out.printf("Calculated SMOTE Percentage: %.2f%%\n", smotePercentage);

            SMOTE smoteFilter = new SMOTE();
            smoteFilter.setInputFormat(selectedData);
            smoteFilter.setPercentage(smotePercentage);
            smoteFilter.setRandomSeed(1);

            Instances balancedData = Filter.useFilter(selectedData, smoteFilter);
            System.out.println("Rows Before SMOTE: " + selectedData.numInstances());
            System.out.println("Rows After SMOTE : " + balancedData.numInstances());

            System.out.println("\nStep 5: Retraining Random Forest on Selected and Balanced Data...");
            RandomForest finalModel = new RandomForest();
            Evaluation evalFinal = evaluateModel(finalModel, balancedData);

            System.out.println("\n==================================================================");
            System.out.println("                  MODEL PERFORMANCE COMPARISON                    ");
            System.out.println("==================================================================");
            System.out.printf("%-22s | %-12s | %-12s | %-12s\n", "Experiment Stage", "Accuracy", "F1-Score", "ROC-AUC");
            System.out.println("------------------------------------------------------------------");
            System.out.printf("%-22s | %11.2f%% | %12.4f | %12.4f\n", "1. Baseline (Raw 24)", evalBaseline.pctCorrect(), evalBaseline.weightedFMeasure(), evalBaseline.weightedAreaUnderROC());
            System.out.printf("%-22s | %11.2f%% | %12.4f | %12.4f\n", "2. Top 10 Features", evalTop10.pctCorrect(), evalTop10.weightedFMeasure(), evalTop10.weightedAreaUnderROC());
            System.out.printf("%-22s | %11.2f%% | %12.4f | %12.4f\n", "3. Top 10 + SMOTE", evalFinal.pctCorrect(), evalFinal.weightedFMeasure(), evalFinal.weightedAreaUnderROC());
            System.out.println("==================================================================");

            finalModel.buildClassifier(balancedData);

            System.out.println("\nStep 6: Saving and Reloading the Trained Model...");
            String modelPath = "mental_health_best_model.model";
            SerializationHelper.write(modelPath, finalModel);
            System.out.println("Saved model file to: " + modelPath);

            RandomForest loadedModel = (RandomForest) SerializationHelper.read(modelPath);
            System.out.println("Successfully reloaded model from disk using SerializationHelper.");

            System.out.println("\nStep 7: Testing Live Prediction on a Sample Record...");
            Instance sample = new DenseInstance(balancedData.numAttributes());
            sample.setDataset(balancedData);

            for (int i = 0; i < balancedData.numAttributes(); i++) {
                if (i == balancedData.classIndex()) {
                    continue;
                }
                Attribute attr = balancedData.attribute(i);
                if (attr.isNominal()) {
                    sample.setValue(attr, attr.value(0));
                } else {
                    sample.setValue(attr, balancedData.meanOrMode(i));
                }
            }

            double predIndex = loadedModel.classifyInstance(sample);
            double[] prob = loadedModel.distributionForInstance(sample);
            String prediction = balancedData.classAttribute().value((int) predIndex);

            System.out.println("Prediction Result: " + prediction);
            System.out.printf("Prediction Confidence: [No: %.2f%%, Yes: %.2f%%]\n", prob[0] * 100.0, prob[1] * 100.0);

            System.out.println("\nMember 2 task completed successfully!");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static Evaluation evaluateModel(RandomForest classifier, Instances dataset) throws Exception {
        Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(classifier, dataset, 10, new Random(1));
        return eval;
    }
}