import java.io.File;
import weka.associations.Apriori;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

public class AssociationRuleMiner {

    public static void main(String[] args) {
        try {
            System.out.println("=================================================================");
            System.out.println("   CSE-4142: MENTAL HEALTH PREDICTIVE ANALYSIS & PATTERN MINING   ");
            System.out.println("              MEMBER 3: EDA & ASSOCIATION RULE MINING             ");
            System.out.println("=================================================================\n");

            // 1. Locate and Load Dataset
            String datasetPath = "clean_mental_health.arff";
            File file = new File(datasetPath);
            if (!file.exists()) {
                datasetPath = ".." + File.separator + "clean_mental_health.arff";
            }

            System.out.println(">> Loading dataset: " + datasetPath);
            DataSource source = new DataSource(datasetPath);
            Instances data = source.getDataSet();

            // Set the target class index to the last attribute
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            System.out.println(">> Total Instances: " + data.numInstances());
            System.out.println(">> Total Attributes: " + data.numAttributes());
            System.out.println(">> Target Class: " + data.classAttribute().name() + "\n");

            // -------------------------------------------------------------
            // TASK 1: EXPLORATORY DATA ANALYSIS (EDA)
            // -------------------------------------------------------------
            System.out.println("-----------------------------------------------------------------");
            System.out.println("                 EXPLORATORY DATA ANALYSIS (EDA)                 ");
            System.out.println("-----------------------------------------------------------------");

            // Class Distribution
            Attribute classAttr = data.classAttribute();
            AttributeStats classStats = data.attributeStats(data.classIndex());
            System.out.println("\n[Class Distribution: " + classAttr.name() + "]");
            for (int i = 0; i < classAttr.numValues(); i++) {
                int count = classStats.nominalCounts[i];
                double pct = (count / (double) data.numInstances()) * 100.0;
                System.out.printf("   - %-6s: %5d instances (%.2f%%)\n", classAttr.value(i), count, pct);
            }

            // Summary Statistics for Numeric Attributes
            System.out.println("\n[Numeric Attributes Summary Statistics (Normalized 0.0 - 1.0)]");
            System.out.printf("%-30s %-10s %-10s %-10s %-10s\n", "Attribute", "Mean", "StdDev", "Min", "Max");
            System.out.println("-------------------------------------------------------------------------");
            for (int i = 0; i < data.numAttributes(); i++) {
                Attribute attr = data.attribute(i);
                if (attr.isNumeric()) {
                    AttributeStats stats = data.attributeStats(i);
                    System.out.printf("%-30s %-10.3f %-10.3f %-10.3f %-10.3f\n",
                            attr.name(),
                            stats.numericStats.mean,
                            stats.numericStats.stdDev,
                            stats.numericStats.min,
                            stats.numericStats.max);
                }
            }

            // -------------------------------------------------------------
            // TASK 2: FEATURE DISCRETIZATION
            // -------------------------------------------------------------
            System.out.println("\n-----------------------------------------------------------------");
            System.out.println("                      FEATURE DISCRETIZATION                     ");
            System.out.println("-----------------------------------------------------------------");
            System.out.println(">> Apriori algorithm requires nominal data.");
            System.out.println(">> Applying WEKA's Discretize filter (3 equal-width bins per numeric attribute)...");

            Discretize discretizeFilter = new Discretize();
            discretizeFilter.setBins(3); // 3 bins: Low [0-0.33], Medium (0.33-0.67], High (0.67-1.0]
            discretizeFilter.setInputFormat(data);
            Instances discretizedData = Filter.useFilter(data, discretizeFilter);

            System.out.println(">> Discretization complete. All attributes are now nominal.\n");

            // -------------------------------------------------------------
            // TASK 3: GENERAL ASSOCIATION RULE MINING (RANKED BY LIFT)
            // -------------------------------------------------------------
            System.out.println("-----------------------------------------------------------------");
            System.out.println("   PART A: TOP GENERAL ASSOCIATION RULES (Workplace & Culture)   ");
            System.out.println("-----------------------------------------------------------------");

            Apriori generalApriori = new Apriori();
            generalApriori.setNumRules(10);
            generalApriori.setLowerBoundMinSupport(0.15); // Minimum 15% Support
            generalApriori.setMinMetric(1.10);             // Minimum Lift > 1.10
            generalApriori.setMetricType(new SelectedTag(1, Apriori.TAGS_SELECTION)); // 1 = Lift
            generalApriori.buildAssociations(discretizedData);

            System.out.println(generalApriori.toString());

            // -------------------------------------------------------------
            // TASK 4: CLASS-ASSOCIATION RULES (Predicting Target Class)
            // -------------------------------------------------------------
            System.out.println("-----------------------------------------------------------------");
            System.out.println("   PART B: CLASS-ASSOCIATION RULES (Predicting Target Class)     ");
            System.out.println("-----------------------------------------------------------------");

            Apriori classApriori = new Apriori();
            classApriori.setNumRules(10);
            classApriori.setLowerBoundMinSupport(0.05); // Minimum 5% Support
            classApriori.setCar(true);                  // Enable Class Association Rules
            // Note: In WEKA, CAR-mining metric type MUST be Confidence (Tag 0)
            classApriori.setMetricType(new SelectedTag(0, Apriori.TAGS_SELECTION)); 
            classApriori.setMinMetric(0.55);            // Minimum Confidence > 55%
            
            // Set target class index (1-based index in WEKA options)
            classApriori.setClassIndex(discretizedData.classIndex() + 1);
            classApriori.buildAssociations(discretizedData);

            System.out.println(classApriori.toString());

            System.out.println("=================================================================");
            System.out.println("                   MINING COMPLETED SUCCESSFULLY!                ");
            System.out.println("=================================================================");

        } catch (Exception e) {
            System.err.println("Error running association rule miner: " + e.getMessage());
            e.printStackTrace();
        }
    }
}