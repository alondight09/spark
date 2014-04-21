
/*** SimpleApp.java ***/
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;

public class SimpleApp {
    public static void main(String[] args) {
        //String logFile = "hdfs://localhost:8020:/user/bjxiao/input/stencil.tex"; // Should be some file on your system
        String logFile = "pom.xml"; // Should be some file on your system
        JavaSparkContext sc = new JavaSparkContext("local", "Simple App",
                System.getenv("SPARK_HOME"), new String[]{"target/simple-project-1.0.jar"});
        JavaRDD<String> logData = sc.textFile(logFile).cache();

        long numAs = logData.filter(new Function<String, Boolean>() {
                public Boolean call(String s) { return s.contains("a"); }
                }).count();

        long numBs = logData.filter(new Function<String, Boolean>() {
                public Boolean call(String s) { return s.contains("b"); }
                }).count();

        System.out.println("Lines with a: " + numAs + ", lines with b: " + numBs);
    }
}
