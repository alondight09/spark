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

import java.io.IOException;
import java.io.Serializable;
import java.io.PrintWriter;
import java.util.*;
import java.util.regex.Pattern;
import java.lang.System;

import accUCLA.api.ComputeLogisticGradient;
/**
 * Logistic regression based classification.
 */
public final class UserLR {

  private static final int D = 10;   // Number of dimensions
  private static final int L = 1;   // Number of labels
  //private static final int D = 784;   // Number of dimensions
  //private static final int L = 10;   // Number of labels
  private static final Random rand = new Random(42);

  static class DataPoint implements Serializable {
    DataPoint(Float[] x, Float[] y) {
      this.x = x;
      this.y = y;
    }

    Float[] x;
    Float[] y;
  }

  static class ParsePoint extends Function<String, DataPoint> {
    private static final Pattern SPACE = Pattern.compile(" ");

    @Override
    public DataPoint call(String line) {
      String[] tok = SPACE.split(line);
      Float[] y = new Float[L];
      for (int i = 0; i < L; i++) {
        y[i] = Float.parseFloat(tok[i]);
      }
      Float[] x = new Float[D];
      for (int i = 0; i < D; i++) {
        x[i] = Float.parseFloat(tok[i + L]);
      }
      return new DataPoint(x, y);
    }
  }

  static class VectorSum extends Function2<Float[][], Float[][], Float[][]> {
    @Override
    public Float[][] call(Float[][] a, Float[][] b) {
      Float[][] result = new Float[L][D];
      for (int i = 0; i < L; i++) {
          for (int j = 0; j < D; j++) {
            result[i][j] = a[i][j] + b[i][j];
          }
      }
      return result;
    }
  }

  static class ComputeGradient extends FlatMapFunction<Iterator<DataPoint>, Float[][]> {
    private final Float[][] weights;

    ComputeGradient(Float[][] weights) {
      this.weights = weights;
    }

    @Override
    public Iterable<Float[][]> call(Iterator<DataPoint> p_iter) throws IOException{
      
      int partition_size = 0;
      List<Float> data = new ArrayList<Float>();
      while (p_iter.hasNext()) {
          partition_size ++;
          DataPoint p = p_iter.next();
          data.addAll(Arrays.asList(p.y));
          data.addAll(Arrays.asList(p.x));
      }
      List<Float[][]> result = new ArrayList<Float[][]>();
      result.add(ComputeLogisticGradient.run(partition_size,L,D,weights,data.toArray(new Float[0])));
      return result;
    }
  }

  public static Float dot(Float[] a, Float[] b) {
    Float x = new Float(0.);
    for (int i = 0; i < D; i++) {
      x += a[i] * b[i];
    }
    return x;
  }

  public static void printWeights(Float[][] a) {
    for(int i = 0; i < L; i++) {
        System.out.println(Arrays.toString(a[i]));
    }
  }

  public static void main(String[] args) {

    if (args.length < 3) {
      System.err.println("Usage: JavaHdfsLR <master> <file> <iters>");
      System.exit(1);
    }

    JavaSparkContext sc = new JavaSparkContext(args[0], "UserLR",
        System.getenv("SPARK_HOME"), "target/simple-project-1.0.jar");
    JavaRDD<String> lines = sc.textFile(args[1]);
    //JavaRDD<String> lines = sc.textFile("lr_data.txt");
    JavaRDD<DataPoint> points = lines.map(new ParsePoint()).cache();
    int ITERATIONS = Integer.parseInt(args[2]);

    // Initialize w to a random value
    Float[][] w = new Float[L][D];
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < D; j++) {
          w[i][j] = 2 * rand.nextFloat() - 1;
        }
    }

    System.out.print("Initial w: ");
    printWeights(w);

    for (int k = 1; k <= ITERATIONS; k++) {
      System.out.println("On iteration " + k);

      Float[][] gradient = points.repartition(100).mapPartitions(
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
    System.exit(0);
  }
}
