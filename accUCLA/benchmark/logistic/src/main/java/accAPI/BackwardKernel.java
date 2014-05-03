
package accUCLA.api;

import org.apache.spark.api.java.function.FlatMapFunction;

import java.util.*;
import java.io.IOException;
import java.lang.InterruptedException;
import java.io.PrintWriter;
import java.io.PrintStream;

public class BackwardKernel {
    private static final Boolean use_acc_manager = false;

    public static float[][] run(int useFPGA, int L, int D, int partition_size, float[][] weights, float[][] data) throws IOException, InterruptedException {
        //debug

        int callFPGA = useFPGA;

        if( use_acc_manager && useFPGA == 1)
        {
            Connector2FPGA conn_manager = new Connector2FPGA("farmer.cs.ucla.edu", 9989);
            while(true)
            {
                int[] request = {0,1};
                conn_manager.buildConnection( );
                conn_manager.send(request,2);
                int[] response = conn_manager.receive_int(3);
                conn_manager.closeConnection( );
                if(response[1]==1) break; // call FPGA
                //if(response[1]==2) // call CPU
                else
                {
                    callFPGA = 0;
                    break;
                }
                //Thread.sleep(response[2]*1000);
            }
        }
        if(callFPGA == 1)
        {
            //PrintWriter writer = new PrintWriter("/tmp/partition_log.txt");
            PrintStream writer = System.out;
            writer.println("Start call FPGA");

            Connector2FPGA conn = new Connector2FPGA("10.0.128.2", 5000);
            conn.buildConnection( );

            writer.println("Start writing L ");
            conn.send(L);

            writer.println("Start writing D ");
            conn.send(D);

            writer.println("Start writing partition_size ");
            conn.send(partition_size);

            writer.println("Start writing array weight ");
            conn.send(weights,L,D);

            writer.println("Start writing array data");
            conn.send(data,partition_size,(L+D));

            writer.println("data transferred");
            writer.close();

            float[][] gradient = conn.receive_float(L,D);
            conn.closeConnection();

            if( use_acc_manager )
            {
                Connector2FPGA conn_manager2 = new Connector2FPGA("farmer.cs.ucla.edu", 9990);
                conn_manager2.buildConnection( );
                conn_manager2.send(1);
                conn_manager2.closeConnection( );
            }
            return gradient;
        }
        else
        {
            float[][] gradient = new float[L][D];
            for(int k = 0; k < partition_size; k++)
            {
                for (int i = 0; i < L; i++) {
                    float dot = 0.f;
                    for(int j = 0; j < D; j++)
                    {
                        dot += weights[i][j] * data[k][j+L];
                    }
                    float coeff = (1 / (1 + (float)(Math.exp(-data[k][i] * dot))) - 1) * data[k][i];
                    for (int j = 0; j < D; j++) {
                        gradient[i][j] += coeff * data[k][j+L];
                    }
                }
            }
            return gradient;
        }
    }
}
