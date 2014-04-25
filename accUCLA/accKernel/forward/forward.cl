
__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void forward(
	int	L,
	int	D,
	__global float* global_weights,                        
        __global float* global_data,                        
        __global int* global_prediction)                        
{                                                   
                                                    
	//int gid = get_group_id(0);                      

	__local float weights[8192];
	__local float data[1024];

	int i, j;

	event_t e_memcpy[2];
	e_memcpy[0] = async_work_group_copy(weights, global_weights, (size_t)(D*L), e_memcpy[0]);
	e_memcpy[1] = async_work_group_copy(data, global_data, (size_t)(D+L), e_memcpy[1]);
	wait_group_events(2, e_memcpy);

	float max_possibility = -1e10;
	int likely_class = 0;
	int prediction;
	for( i = 0; i < L; i++ )
	{
		float dot = 0.;
		//__attribute__((xcl_pipeline_loop)) 
		for( j = 0; j < D; j++ )
		{
			dot += weights[i*D+j]*data[j+L];
		}
		if( dot > max_possibility )
		{
			max_possibility = dot;
			likely_class = i;
		}
	}
	if( L <= 1 )
	{
		prediction = data[0] > 0 ? ( max_possibility > 0 ? 1 : 0 ) : ( max_possibility < 0 ? 1 : 0 );
	}
	else
	{
		prediction = ( data[likely_class] > 0 ? 1 : 0 );
	}
	global_prediction += prediction;
}
