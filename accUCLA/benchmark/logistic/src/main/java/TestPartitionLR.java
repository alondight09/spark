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

import java.io.*;
import java.io.Serializable;
import java.util.*;
import java.util.regex.Pattern;

/**
 * Logistic regression based classification.
 */
public final class TestPartitionLR {

  private static final int D = 10;   // Number of dimensions
  private static final Random rand = new Random(42);

  static class DataPoint implements Serializable {
    DataPoint(float[] x, float y) {
      this.x = x;
      this.y = y;
    }

    float[] x;
    float y;
  }

  static class ParsePoint extends Function<String, DataPoint> {
    private static final Pattern SPACE = Pattern.compile(" ");

    @Override
    public DataPoint call(String line) {
      String[] tok = SPACE.split(line);
      float y = Float.parseFloat(tok[0]);
      float[] x = new float[D];
      for (int i = 0; i < D; i++) {
        x[i] = Float.parseFloat(tok[i + 1]);
      }
      return new DataPoint(x, y);
    }
  }

  static class VectorSum extends Function2<float[], float[], float[]> {
    @Override
    public float[] call(float[] a, float[] b) {
      float[] result = new float[D];
      for (int j = 0; j < D; j++) {
        result[j] = a[j] + b[j];
      }
      return result;
    }
  }

  static class ComputeGradient extends FlatMapFunction<Iterator<DataPoint>, float[]> {
    private final float[] weights;

    ComputeGradient(float[] weights) {
      this.weights = weights;
    }

    @Override
    public Iterable<float[]> call(Iterator<DataPoint> p_iter) {
      List<float[]> result = new ArrayList<float[]>();
      if(p_iter.hasNext())
      {
          try{
              DataPoint p = p_iter.next();
              PrintWriter writer = new PrintWriter("/tmp/partitionLR_"+p.y+".log");
              while(true)
              {
                  writer.println(p.y);
                  float[] gradient = new float[D];
                  for (int i = 0; i < D; i++) {
                    float dot = dot(weights, p.x);
                    gradient[i] = (1 / (1 + (float)(Math.exp(-p.y * dot))) - 1) * p.y * p.x[i];
                    writer.println(p.x[i]);
                  }
                  result.add(gradient);
                  if(p_iter.hasNext()==false) break;
                  p = p_iter.next();
              }
              writer.close();
          } catch(FileNotFoundException ex) {
          }
      }
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

  public static void printWeights(float[] a) {
    System.out.println(Arrays.toString(a));
  }

  public static void main(String[] args) {

    if (args.length < 3) {
      System.err.println("Usage: JavaHdfsLR <master> <file> <iters>");
      System.exit(1);
    }

    JavaSparkContext sc = new JavaSparkContext(args[0], "TestPartitionLR",
        System.getenv("SPARK_HOME"), "target/simple-project-1.0.jar");
    JavaRDD<String> lines = sc.textFile(args[1]);
    //JavaRDD<String> lines = sc.textFile("lr_data.txt");
    JavaRDD<DataPoint> points = lines.map(new ParsePoint()).cache();
    int ITERATIONS = Integer.parseInt(args[2]);

    // Initialize w to a random value
    float[] w = new float[D];
    for (int i = 0; i < D; i++) {
      w[i] = 2 * rand.nextFloat() - 1;
    }

    System.out.print("Initial w: ");
    printWeights(w);

    for (int i = 1; i <= ITERATIONS; i++) {
      System.out.println("On iteration " + i);

      float[] gradient = points.mapPartitions(
        new ComputeGradient(w)
      ).reduce(new VectorSum());

      for (int j = 0; j < D; j++) {
        w[j] -= gradient[j];
      }

    }

    System.out.print("Final w: ");
    printWeights(w);
    System.exit(0);
  }
}
