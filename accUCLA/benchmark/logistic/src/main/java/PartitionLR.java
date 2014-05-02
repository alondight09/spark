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

import org.apache.commons.lang.ArrayUtils;
import java.lang.InterruptedException;
import java.io.IOException;
import java.io.Serializable;
import java.util.*;
import java.util.regex.Pattern;
import java.lang.System;

import accUCLA.api.BackwardKernel;
/**
 * Logistic regression based classification.
 */
public final class PartitionLR {

  private static int D = 784;   // Number of dimensions
  private static int L = 10;   // Number of labels
  private static int useFPGA = 1;

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

  static class ForwardLR extends FlatMapFunction<Iterator<DataPoint>, float[]> {
    private final float[][] weights;

    ForwardLR(float[][] weights) {
        this.weights = weights;
    }

    @Override
    public Iterable<float[]> call(Iterator<DataPoint> p_iter) {
        List<float[]> result = new ArrayList<float[]>();
        while(p_iter.hasNext())
        {
            DataPoint p = p_iter.next();
            float[] prediction = new float[L];
            for(int i = 0; i < L; i++) {
                float dot = dot(weights[i], p.x);
                prediction[i] = 1 / ( 1 + (float)(Math.exp(-dot)) );
            }
            result.add(prediction);
        }
        return result;
    }
  }

  static class BackwardLR extends FlatMapFunction<Iterator<DataPoint>, float[][]> {
    private final float[][] weights;

    BackwardLR(float[][] weights) {
      this.weights = weights;
    }

    @Override
    public Iterable<float[][]> call(Iterator<DataPoint> p_iter) throws IOException, InterruptedException {
      List<float[][]> result = new ArrayList<float[][]>();
      int partition_size = 0;
      List<float[]> data = new ArrayList<float[]>();
      while (p_iter.hasNext()) {
          partition_size ++;
          DataPoint p = p_iter.next();
          data.add(ArrayUtils.addAll(p.y,p.x));
      }
      result.add(BackwardKernel.run(useFPGA,L,D,partition_size,weights,data.toArray(new float[0][L+D])));

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

    if (args.length < 6) {
      System.err.println("Usage: JavaHdfsLR <master> <file> <iters> <L> <D> <use FPGA?> (<testing file>)");
      System.exit(1);
    }
    int ITERATIONS = Integer.parseInt(args[2]);
    System.out.println("iterations: "+ITERATIONS);
    L = Integer.parseInt(args[3]);
    System.out.println("L: "+L);
    D = Integer.parseInt(args[4]);
    System.out.println("D: "+D);
    useFPGA = Integer.parseInt(args[5]);
    System.out.println("use FPGA: "+useFPGA);

    JavaSparkContext sc = new JavaSparkContext(args[0], "PartitionLR",
        System.getenv("SPARK_HOME"), "target/simple-project-1.0.jar");
    JavaRDD<String> lines = sc.textFile(args[1]);
    JavaRDD<DataPoint> points = lines.map(new ParsePoint()).repartition(32).cache();

    float[][] w = new float[L][D];
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < D; j++) {
          w[i][j] = 0.0f;
        }
    }

    System.out.print("Initial w: ");
    printWeights(w);

    for (int k = 1; k <= ITERATIONS; k++) {
      System.out.println("On iteration " + k);

      long tic = System.nanoTime( );
      float[][] gradient = points.mapPartitions(
        new BackwardLR(w)
      ).reduce(new VectorSum());
      System.out.println("elapsed time: " + (System.nanoTime()-tic)/1e9);


      for (int i = 0; i < L; i++) {
        for (int j = 0; j < D; j++) {
          w[i][j] -= gradient[i][j];
        }
      }

    }

    System.out.print("Final w: ");
    printWeights(w);

    lines = sc.textFile( args.length < 7 ? args[1] : args[6] );
    System.out.println("first prediction");
    System.out.println(Arrays.toString(lines.map(new ParsePoint()).repartition(32).mapPartitions( new ForwardLR(w) ).first( )));

    System.exit(0);
  }
}
