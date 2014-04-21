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

import java.io.IOException;
import java.io.PrintWriter;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.net.Socket;
import java.net.InetAddress;
import java.net.ServerSocket;
/**
 * Logistic regression based classification.
 */
public final class SocketTest {

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

  public static int big2LittleEndianInt(int i) {
    int b0,b1,b2,b3;

    b0 = (i&0xff)>>0;
    b1 = (i&0xff00)>>8;
    b2 = (i&0xff0000)>>16;
    b3 = (i&0xff000000)>>24;

    return ((b0<<24)|(b1<<16)|(b2<<8)|(b3<<0));
  }
  public static byte[] big2LittleEndianFloat(float f) {
    int floatBits = Float.floatToIntBits(f);
    byte floatBytes[] = new byte[4];
    floatBytes[3] = (byte)(floatBits>>24 & 0xff);
    floatBytes[2] = (byte)(floatBits>>16 & 0xff);
    floatBytes[1] = (byte)(floatBits>>8 & 0xff);
    floatBytes[0] = (byte)(floatBits & 0xff);
    return floatBytes;
  }

  public static byte[] big2LittleEndianFloatArray(float[] f, int len) {
    byte floatBytes[] = new byte[4*len];
    for (int i = 0; i < len; i ++) {
      int floatBits = Float.floatToIntBits(f[i]);
      floatBytes[4*i + 3] = (byte)(floatBits>>24 & 0xff);
      floatBytes[4*i + 2] = (byte)(floatBits>>16 & 0xff);
      floatBytes[4*i + 1] = (byte)(floatBits>>8 & 0xff);
      floatBytes[4*i + 0] = (byte)(floatBits & 0xff);
    }
    return floatBytes;
  }

  static class ComputeGradient extends FlatMapFunction<Iterator<DataPoint>, float[]> {
    private final float[] weights;

    ComputeGradient(float[] weights) {
      this.weights = weights;
    }

    @Override
    public Iterable<float[]> call(Iterator<DataPoint> p_iter) throws IOException{
      //debug
      PrintWriter writer = new PrintWriter("/tmp/partition/");
      Iterator<DataPoint> p_start = p_iter;
      
      InetAddress addr = InetAddress.getByName("10.0.128.2"); 
      Socket socket = new Socket(addr, 5000);
      DataOutputStream o2 = 
           new DataOutputStream(socket.getOutputStream());

      o2.writeInt(big2LittleEndianInt(D));
      o2.flush();
      
      List<float[]> result = new ArrayList<float[]>();
      int partition_size = 0;
      int idx = 0;
      while (p_iter.hasNext()) {
          partition_size ++;
      }
      
      o2.writeInt(big2LittleEndianInt(partition_size));
      o2.flush();
      
      byte[] weight_byte = big2LittleEndianFloatArray(weights, D);
      o2.write(weight_byte, 0, 4*D);
      o2.flush();
      
      byte[] pf = new byte[4*partition_size*(D+1)];

      p_iter = p_start;
      while (p_iter.hasNext()) {
          DataPoint p = p_iter.next();
	      
          byte[] yf = new byte[4];
          yf = big2LittleEndianFloat(p.y);
          byte[] xf = new byte[4*D];
          xf = big2LittleEndianFloatArray(p.x, D);
          for (int j = 0; j < 4; j ++) {
            pf[idx ++] = yf[j];
	  }
          for (int j = 0; j < 4*D; j ++) {
            pf[idx ++] = xf[j];
	  }
          
          /*float[] gradient = new float[D];
          for (int i = 0; i < D; i++) {
            float dot = dot(weights, p.x);
            gradient[i] = (1 / (1 + (float)(Math.exp(-p.y * dot))) - 1) * p.y * p.x[i];
          }
          result.add(gradient);*/
      }
      o2.write(pf);
      o2.flush();
 
      DataInputStream in = new DataInputStream(socket.getInputStream());
      byte[] gradient_byte = new byte[D*4*partition_size];
      in.readFully(gradient_byte);
      ByteBuffer buf2 = ByteBuffer.wrap(gradient_byte).order(ByteOrder.LITTLE_ENDIAN);
      for (int i = 0; i < partition_size; i ++) {
        float[] gradient = new float[D];
        for (int j = 0; j < D; j ++) {
          gradient[j] = buf2.getFloat((i*D + j)*4);
        }
        result.add(gradient); 
      }
      o2.close();
      in.close();
      socket.close();
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

    JavaSparkContext sc = new JavaSparkContext(args[0], "SocketLR",
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
      
      float[] gradient = points.repartition(1).mapPartitions(
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
