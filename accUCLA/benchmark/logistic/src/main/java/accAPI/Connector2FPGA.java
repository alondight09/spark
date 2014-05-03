
package accUCLA.api;

import java.util.*;
import java.io.IOException;
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


public class Connector2FPGA {
    private Socket socket;
    private DataOutputStream o2;
    private DataInputStream in;
    private final String ip;
    private final int port;
    private Boolean is_connected;

    public Connector2FPGA(String ip, int port)
    {
        this.ip = ip;
        this.port = port;
    }
    public void buildConnection( ) throws IOException
    {
        InetAddress addr = InetAddress.getByName(ip); 
        socket = new Socket(addr, port);
        o2 = new DataOutputStream(socket.getOutputStream());
        in = new DataInputStream(socket.getInputStream());
        is_connected = true;
    }

    public void send( int i ) throws IOException
    {
        o2.writeInt(big2LittleEndian.Int(i));
        o2.flush();
    }
    public void send( float[] float_array, int len ) throws IOException
    {
        o2.write(big2LittleEndian.floatArray(float_array, len));
        o2.flush();
    }
    public void send( float[][] float_array, int len1, int len2 ) throws IOException
    {
        o2.write(big2LittleEndian.floatArray(float_array, len1, len2));
        o2.flush();
    }
    public void send( int[] int_array, int len ) throws IOException
    {
        o2.write(big2LittleEndian.IntArray(int_array, len));
        o2.flush();
    }
    public int receive( ) throws IOException
    {
        return in.readInt( );
    }
    public int[] receive_int( int len ) throws IOException
    {
        byte[] byte_array = new byte[len*4];
        in.readFully(byte_array);
        ByteBuffer buf2 = ByteBuffer.wrap(byte_array).order(ByteOrder.LITTLE_ENDIAN);
        int[] result = new int[len];
        for(int i = 0; i < len; i++)
        {
            result[i] = buf2.getInt(i*4);
        }
        return result;
    }
    public float[] receive_float( int len ) throws IOException
    {
        byte[] byte_array = new byte[len*4];
        in.readFully(byte_array);
        ByteBuffer buf2 = ByteBuffer.wrap(byte_array).order(ByteOrder.LITTLE_ENDIAN);
        float[] result = new float[len];
        for(int i = 0; i < len; i++)
        {
            result[i] = buf2.getFloat(i*4);
        }
        return result;
    }
    public float[][] receive_float( int len1, int len2 ) throws IOException
    {
        float[][] result = new float[len1][len2];
        float[] data = receive_float( len1 * len2 );
        for( int i = 0; i < len1; i++ )
        {
                System.arraycopy(data,i*len2,result[i],0,len2);
        }
        return result;
    }
    public void closeConnection( ) throws IOException
    {
        o2.close();
        in.close();
        socket.close();
    }
}
