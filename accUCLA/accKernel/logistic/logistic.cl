
__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void logistic(
	int	L,
	int	D,
	__global float* global_weights,                        
        __global float* global_data,                        
        __global float* global_gradient)                        
{                                                   
                                                    
	//int gid = get_group_id(0);                      

	__local float weights[8192];
	__local float data[1024];
	__local float gradient[8192];

	int i, j;

	event_t e_memcpy[3];
	e_memcpy[0] = async_work_group_copy(weights, global_weights, (size_t)(D*L), e_memcpy[0]);
	e_memcpy[1] = async_work_group_copy(data, global_data, (size_t)(D+L), e_memcpy[1]);
	e_memcpy[2] = async_work_group_copy(gradient, global_gradient, (size_t)(D*L), e_memcpy[2]);
	wait_group_events(3, e_memcpy);

	for( i = 0; i < L; i++ )
	{
		float dot = 0.;
		//__attribute__((xcl_pipeline_loop)) 
		for( j = 0; j < D; j++ )
		{
			dot += weights[i*D+j]*data[j+L];
		}
		float coeff = (1./(1.+exp(-data[i]*dot ))-1.)*data[i];
		//__attribute__((xcl_pipeline_loop)) 
		for (j = 0; j < D; j++) 
		{ 
			gradient[i*D+j] += coeff*data[j+L];
		}
	}
	e_memcpy[0] = async_work_group_copy(global_gradient, gradient, (size_t)(D*L), e_memcpy[0]);
	wait_group_events(1, e_memcpy);
}
