import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.filters.unsupervised.attribute.Normalize;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.SMO;

import java.io.*;
import java.util.ArrayList;
import java.util.Random;

public class SupervisedClassifier {

    public static void main(String[] args) {
        try {
            System.out.println("==================================================");
            System.out.println("   MEMBER 1: REALISTIC BINARY CLASSIFICATION      ");
            System.out.println("==================================================\n");

            // 1. LOCATE CSV DATASET
            File inputFile = new File("mental_health_tech_2024.csv");
            if (!inputFile.exists()) {
                inputFile = new File("survey.csv");
            }
            if (!inputFile.exists()) {
                System.err.println("ERROR: Could not find dataset file in project folder!");
                return;
            }

            // 2. SANITIZE RAW CSV QUOTES & APOSTROPHES
            System.out.println("1. Sanitizing raw CSV quotes and apostrophes...");
            File sanitizedFile = new File("temp_sanitized_survey.csv");
            try (BufferedReader reader = new BufferedReader(new FileReader(inputFile));
                 BufferedWriter writer = new BufferedWriter(new FileWriter(sanitizedFile))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    line = line.replace("'", "");
                    writer.write(line);
                    writer.newLine();
                }
            }
            sanitizedFile.deleteOnExit();

            // 3. LOAD DATASET INTO WEKA
            System.out.println("2. Loading dataset into WEKA...");
            CSVLoader loader = new CSVLoader();
            loader.setSource(sanitizedFile);
            loader.setBufferSize(10000);
            Instances data = loader.getDataSet();
            System.out.println("   -> Successfully loaded " + data.numInstances() + " rows with " + data.numAttributes() + " raw columns.");

            // 4. PREPROCESSING PIPELINE
            System.out.println("\n3. Applying WEKA Preprocessing Filters...");

            // (a) Identify and Remove ID + the 6 Data Leakage Columns
            StringBuilder removeIndices = new StringBuilder();
            for (int i = 0; i < data.numAttributes(); i++) {
                String name = data.attribute(i).name().toLowerCase();
                
                boolean isLeakage = name.contains("id") && i == 0 ||
                                    name.equals("mental_health_condition") ||
                                    name.equals("diagnosis_type") ||
                                    name.equals("sought_treatment") ||
                                    name.equals("used_mh_benefits") ||
                                    name.equals("mental_health_leave_taken") ||
                                    name.equals("disclosed_to_employer");

                if (isLeakage) {
                    if (removeIndices.length() > 0) removeIndices.append(",");
                    removeIndices.append(i + 1); // WEKA uses 1-based indexing
                    System.out.println("   [✓] Flagged leakage column for removal: " + data.attribute(i).name());
                }
            }

            Remove removeFilter = new Remove();
            removeFilter.setAttributeIndices(removeIndices.toString());
            removeFilter.setInputFormat(data);
            data = Filter.useFilter(data, removeFilter);
            System.out.println("   [✓] Successfully removed all leakage columns.");

            // (b) Format Target Class to Nominal {No, Yes}
            int targetIdx = data.attribute("has_mental_health_condition").index();
            
            ArrayList<String> classValues = new ArrayList<>();
            classValues.add("No");
            classValues.add("Yes");
            Attribute nominalClass = new Attribute("has_mental_health_condition_nominal", classValues);

            data.insertAttributeAt(nominalClass, data.numAttributes());
            int newClassIdx = data.numAttributes() - 1;

            for (int i = 0; i < data.numInstances(); i++) {
                double val = data.instance(i).value(targetIdx);
                if (val == 0.0) {
                    data.instance(i).setValue(newClassIdx, "No");
                } else {
                    data.instance(i).setValue(newClassIdx, "Yes");
                }
            }

            // Remove the old numeric target
            Remove removeOldTarget = new Remove();
            removeOldTarget.setAttributeIndices("" + (targetIdx + 1));
            removeOldTarget.setInputFormat(data);
            data = Filter.useFilter(data, removeOldTarget);

            // Set class index to the new nominal target (last column)
            data.setClassIndex(data.numAttributes() - 1);
            System.out.println("   [✓] Formatted target attribute to Nominal {No, Yes}.");

            // (c) Convert String attributes to Nominal
            StringToNominal strToNom = new StringToNominal();
            strToNom.setAttributeRange("first-last");
            strToNom.setInputFormat(data);
            data = Filter.useFilter(data, strToNom);
            System.out.println("   [✓] StringToNominal applied.");

            // (d) Handle Missing Values
            ReplaceMissingValues replaceMissing = new ReplaceMissingValues();
            replaceMissing.setInputFormat(data);
            data = Filter.useFilter(data, replaceMissing);
            System.out.println("   [✓] ReplaceMissingValues applied.");

            // (e) Normalize Numeric Attributes
            Normalize normalize = new Normalize();
            normalize.setInputFormat(data);
            data = Filter.useFilter(data, normalize);
            System.out.println("   [✓] Normalize applied.");

            System.out.println("\n4. Prediction Target: '" + data.classAttribute().name() + "' {No, Yes}");
            System.out.println("   Valid Features Used for Prediction: " + (data.numAttributes() - 1));

            // 5. SAVE CLEAN ARFF (For Member 2 and Member 3)
            File arffOutput = new File("clean_mental_health.arff");
            ArffSaver saver = new ArffSaver();
            saver.setInstances(data);
            saver.setFile(arffOutput);
            saver.writeBatch();
            System.out.println("5. Saved clean binary dataset to: '" + arffOutput.getName() + "'");
            System.out.println("   -> (Share 'clean_mental_health.arff' with Member 2 and Member 3)");

            // 6. TRAIN & COMPARE 3 CLASSIFIERS (10-FOLD CV)
            System.out.println("\n==================================================");
            System.out.println("   6. REALISTIC CLASSIFIER COMPARISON (10-FOLD CV)");
            System.out.println("==================================================");

            Classifier[] models = {
                new J48(),           // Decision Tree
                new RandomForest(),  // Random Forest
                new SMO()            // Support Vector Machine (SVM)
            };

            String[] modelNames = {"J48 (Decision Tree)", "Random Forest", "SMO (SVM)"};

            for (int i = 0; i < models.length; i++) {
                evaluateModel(models[i], modelNames[i], data);
            }

            System.out.println("\n[✓] Member 1 pipeline finished successfully with realistic results!");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void evaluateModel(Classifier classifier, String modelName, Instances data) throws Exception {
        System.out.println("\n--------------------------------------------------");
        System.out.println(" Training & Evaluating: " + modelName);
        System.out.println("--------------------------------------------------");

        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(classifier, data, 10, new Random(1));

        // Required Metrics
        System.out.printf(" Accuracy  : %.2f%%\n", eval.pctCorrect());
        System.out.printf(" Precision : %.4f\n", eval.weightedPrecision());
        System.out.printf(" Recall    : %.4f\n", eval.weightedRecall());
        System.out.printf(" F1-Score  : %.4f\n", eval.weightedFMeasure());
        System.out.printf(" ROC-AUC   : %.4f\n", eval.weightedAreaUnderROC());
        
        System.out.println("\n Confusion Matrix:");
        System.out.println(eval.toMatrixString());
    }
}