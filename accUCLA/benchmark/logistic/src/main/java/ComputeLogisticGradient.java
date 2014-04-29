
package accUCLA.api;

import java.io.IOException;
import java.lang.InterruptedException;
import java.io.PrintWriter;

public class ComputeLogisticGradient{

    private static final Boolean use_acc_manager = false;

    public static Float[][] run(int partition_size, int L, int D, Float[][] weights, Float[] data) throws IOException, InterruptedException
    {
      //debug
      PrintWriter writer = new PrintWriter("/tmp/partition_log.txt");

  if( use_acc_manager )
  {
      Connector2FPGA conn_manager = new Connector2FPGA("farmer.cs.ucla.edu", 9989);
      while(true)
      {
        int[] request = {0,1};
        conn_manager.buildConnection( );
        conn_manager.send(request,2);
        int[] response = conn_manager.receive_int(3);
        if(response[1]==1) break;
        Thread.sleep(response[2]*1000);
        conn_manager.closeConnection( );
      }
  }

      writer.println("Start call ");
      //Iterator<DataPoint> p_start = p_iter;
      
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
      conn.send(data,partition_size*(L+D));
 
      writer.println("data transferred");
      writer.close();

      Float[][] result = conn.receive_float(L,D);
      conn.closeConnection();

  if( use_acc_manager )
  {
      Connector2FPGA conn_manager2 = new Connector2FPGA("farmer.cs.ucla.edu", 9990);
      conn_manager2.buildConnection( );
      conn_manager2.send(1);
      conn_manager2.closeConnection( );
  }
      return result;
    }
}
