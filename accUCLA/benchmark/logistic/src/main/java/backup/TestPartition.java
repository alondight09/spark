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
import org.apache.spark.api.java.function.DoubleFlatMapFunction;

import java.util.*;
import java.io.*;

/**
 * Logistic regression based classification.
 */
public final class TestPartition {

  public static void main(String[] args) {

    if (args.length < 2) {
      System.err.println("Usage: JavaHdfsLR <master> <file>");
      System.exit(1);
    }

    JavaSparkContext sc = new JavaSparkContext(args[0], "TestPartition",
        System.getenv("SPARK_HOME"), "target/simple-project-1.0.jar");
    List<Double> partial_sum = sc.textFile(args[1]).map(
      new Function<String, Double>() {
        @Override
        public Double call(String line) {
          return Double.parseDouble(line);
        }
      }
    ).mapPartitions(new DoubleFlatMapFunction< Iterator<Double> >() {
        @Override
        public Iterable<Double> call(Iterator<Double> dataList) {
          List<Double> result = new ArrayList<Double>();
          if(dataList.hasNext())
          {
              try{
                  double temp = dataList.next();
                  PrintWriter writer = new PrintWriter( "/tmp/partition_"+temp+".log" );
                  Double sum = 0.;
                  while(true){
                    sum += temp;
                    writer.println(temp);
                    if(dataList.hasNext()==false) break;
                    temp = dataList.next();
                  }
                  result.add(sum);
                  writer.close();
              } catch(FileNotFoundException ex){
              }
          }
          return result;
        }
    }).collect();
    for(Double d : partial_sum) {
        System.out.println("result="+d);
    }
    System.exit(0);
  }
}
