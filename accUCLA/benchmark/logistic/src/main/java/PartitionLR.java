/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.FlatMapFunction;

import java.io.Serializable;
import java.util.*;
import java.util.regex.Pattern;

/**
 * Logistic regression based classification.
 */
public final class PartitionLR {

  private static final int D = 10;   // Number of dimensions
  private static final int L = 1;   // Number of labels
  //private static final int D = 784;   // Number of dimensions
  //private static final int L = 10;   // Number of labels
  private static final Random rand = new Random(42);

  static class DataPoint implements Serializable {
    DataPoint(float[] x, float[] y) {
      this.x = x;
      this.y = y;
    }

    float[] x;
    float[] y;
  }

  static class ParsePoint extends Function<String, DataPoint> {
    private static final Pattern SPACE = Pattern.compile(" ");

    @Override
    public DataPoint call(String line) {
      String[] tok = SPACE.split(line);
      float[] y = new float[L];
      for (int i = 0; i < L; i++) {
        y[i] = Float.parseFloat(tok[i]);
      }
      float[] x = new float[D];
      for (int i = 0; i < D; i++) {
        x[i] = Float.parseFloat(tok[i + L]);
      }
      return new DataPoint(x, y);
    }
  }

  static class VectorSum extends Function2<float[][], float[][], float[][]> {
    @Override
    public float[][] call(float[][] a, float[][] b) {
      float[][] result = new float[L][D];
      for (int i = 0; i < L; i++) {
          for (int j = 0; j < D; j++) {
            result[i][j] = a[i][j] + b[i][j];
          }
      }
      return result;
    }
  }

  static class TestPrediction extends FlatMapFunction<Iterator<DataPoint>, Integer> {
    private final float[][] weights;

    TestPrediction(float[][] weights) {
        this.weights = weights;
    }

    @Override
    public Iterable<Integer> call(Iterator<DataPoint> p_iter) {
        List<Integer> result = new ArrayList<Integer>();
        int prediction = 0;
        while(p_iter.hasNext())
        {
            DataPoint p = p_iter.next();
            float max_possibility = Float.MIN_VALUE;
            int likely_class = 0;
            for(int i = 0; i < L; i++) {
                float dot = dot(weights[i], p.x);
                if(dot > max_possibility)
                {
                    max_possibility = dot;
                    likely_class = i;
                }
            }
            if( L <= 1 )
            {
                prediction += p.y[0] > 0 ?  ( max_possibility > 0 ? 1 : 0 ) : ( max_possibility < 0 ? 1 : 0 );
            }
            else
            {
                prediction += ( p.y[likely_class] > 0 ? 1 : 0 );
            }
        }
        result.add(prediction);
        return result;
    }
  }

  static class ComputeGradient extends FlatMapFunction<Iterator<DataPoint>, float[][]> {
    private final float[][] weights;

    ComputeGradient(float[][] weights) {
      this.weights = weights;
    }

    @Override
    public Iterable<float[][]> call(Iterator<DataPoint> p_iter) {
      List<float[][]> result = new ArrayList<float[][]>();
      float[][] gradient = new float[L][D];
      while(p_iter.hasNext())
      {
          DataPoint p = p_iter.next();
          for (int i = 0; i < L; i++) {
              float dot = dot(weights[i], p.x);
              for (int j = 0; j < D; j++) {
                gradient[i][j] += (1 / (1 + (float)(Math.exp(-p.y[i] * dot))) - 1) * p.y[i] * p.x[j];
              }
          }
      }
      result.add(gradient);
      return result;
    }
  }

  public static float dot(float[] a, float[] b) {
    float x = 0;
    for (int i = 0; i < D; i++) {
      x += a[i] * b[i];
    }
    return x;
  }

  public static void printWeights(float[][] a) {
    for(int i = 0; i < L; i++) {
        System.out.println(Arrays.toString(a[i]));
    }
  }

  public static void main(String[] args) {

    if (args.length < 3) {
      System.err.println("Usage: JavaHdfsLR <master> <file> <iters> (<testing file>)");
      System.exit(1);
    }

    JavaSparkContext sc = new JavaSparkContext(args[0], "PartitionLR",
        System.getenv("SPARK_HOME"), "target/simple-project-1.0.jar");
    JavaRDD<String> lines = sc.textFile(args[1]);
    //JavaRDD<String> lines = sc.textFile("lr_data.txt");
    JavaRDD<DataPoint> points = lines.map(new ParsePoint()).cache();
    int ITERATIONS = Integer.parseInt(args[2]);

    // Initialize w to a random value
    float[][] w = new float[L][D];
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < D; j++) {
          w[i][j] = 2 * rand.nextFloat() - 1;
        }
    }

    System.out.print("Initial w: ");
    printWeights(w);

    for (int k = 1; k <= ITERATIONS; k++) {
      System.out.println("On iteration " + k);

      float[][] gradient = points.mapPartitions(
        new ComputeGradient(w)
      ).reduce(new VectorSum());

      for (int i = 0; i < L; i++) {
        for (int j = 0; j < D; j++) {
          w[i][j] -= gradient[i][j];
        }
      }

    }

    System.out.print("Final w: ");
    printWeights(w);

    lines = sc.textFile( args.length < 4 ? args[1] : args[3] );
    points = lines.map(new ParsePoint());
    float error_rate = points.mapPartitions( new TestPrediction(w) ).reduce( new Function2<Integer, Integer, Integer>() {
      @Override
      public Integer call(Integer integer, Integer integer2) {
        return integer + integer2;
      }
    }) / (float)( points.count() );
    System.out.println("error rate: "+(error_rate*100)+"%");

    System.exit(0);
  }
}
