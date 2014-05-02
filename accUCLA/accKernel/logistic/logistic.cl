#define LABEL_SIZE		10
#define FEATURE_SIZE	784

__kernel //__attribute__ ((reqd_work_group_size(1, 1, 1)))
void logistic(
	int	L,
	int	D,
	int n, 
	__global float* global_weights,                        
    __global float* global_data,                        
    __global float* global_gradient)                        
{                                                   
                                                    
	int gid = get_group_id(0);                      
	int gnum = get_num_groups(0);                      

	__local float data[FEATURE_SIZE+LABEL_SIZE];

	__local float weights[LABEL_SIZE*FEATURE_SIZE];
	__local float gradient[LABEL_SIZE*FEATURE_SIZE];

	int i, j, k;

	event_t e_memcpy[2];
	event_t e_memout[1];

	e_memcpy[0] = async_work_group_copy(weights, global_weights, D*L, e_memcpy[0]);
	e_memcpy[1] = async_work_group_copy(gradient, global_gradient+gid*D*L, D*L, e_memcpy[1]);
	wait_group_events(2, e_memcpy);

	for (k = gid; k < n; k += gnum) 
	{
		e_memcpy[0] = async_work_group_copy(data, global_data + k*(D+L), (size_t)(D+L), e_memcpy[0]);
		wait_group_events(1, e_memcpy);

		for (i = 0; i < L; i++ ) // L: label (1, 10)
		{
			float dot = 0.;

			__attribute__((xcl_pipeline_loop)) 
			for (j = 0; j < D; j+=2) // D: feature (10, 784 (28x28))
			{
				dot += weights[i*D+j+0]*data[j+L+0];
				dot += weights[i*D+j+1]*data[j+L+1];
			}

			float coeff = (1./(1.+exp(-data[i]*dot))-1.)*data[i];

			__attribute__((xcl_pipeline_loop)) 
			for (j = 0; j < D; j++) 
			{ 
				gradient[i*D+j+0] += coeff*data[j+L+0];
			}

		}
	}
	e_memout[0] = async_work_group_copy(global_gradient+gid*D*L, gradient, D*L, e_memout[0]);
	wait_group_events(1, e_memout);
}
