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

	__local float weights_ping[FEATURE_SIZE];
	__local float gradient_ping[FEATURE_SIZE];

	__local float weights_pong[FEATURE_SIZE];
	__local float gradient_pong[FEATURE_SIZE];

	int i, j, k;

	event_t e_memcpy[2];
	event_t e_memout[1];

	for (k = gid; k < n; k += gnum) 
	{
		e_memcpy[0] = async_work_group_copy(data, global_data + k*(D+L), (size_t)(D+L), e_memcpy[0]);
		wait_group_events(1, e_memcpy);

		e_memcpy[0] = async_work_group_copy(weights_ping, global_weights, D, e_memcpy[0]);
		e_memcpy[1] = async_work_group_copy(gradient_ping, global_gradient+gid*D*L, D, e_memcpy[1]);
		for (i = 0; i < L; i++ ) // L: label (1, 10)
		{
			wait_group_events(2, e_memcpy);

			__local float *weights;
			__local float *gradient;

			if (i < L-1) {
			if (i%2 == 1) {
				e_memcpy[0] = async_work_group_copy(weights_ping, global_weights+(i+1)*D, D, e_memcpy[0]);
				e_memcpy[1] = async_work_group_copy(gradient_ping, global_gradient+gid*D*L+(i+1)*D, D, e_memcpy[1]);

				weights = weights_pong;
				gradient = gradient_pong;
			}
			else {
				e_memcpy[0] = async_work_group_copy(weights_pong, global_weights+(i+1)*D, D, e_memcpy[0]);
				e_memcpy[1] = async_work_group_copy(gradient_pong, global_gradient+gid*D*L+(i+1)*D, D, e_memcpy[1]);

				weights = weights_ping;
				gradient = gradient_ping;
			}
			}

			float dot = 0.;

			__attribute__((xcl_pipeline_loop)) 
			for (j = 0; j < D; j+=2) // D: feature (10, 784 (28x28))
			{
				dot += weights[j+0]*data[j+L+0];
				dot += weights[j+1]*data[j+L+1];
			}

			float coeff = (1./(1.+exp(-data[i]*dot))-1.)*data[i];

			__attribute__((xcl_pipeline_loop)) 
			for (j = 0; j < D; j++) 
			{ 
				gradient[j+0] += coeff*data[j+L+0];
			}

			e_memout[0] = async_work_group_copy(global_gradient+gid*D*L+i*D, gradient, (D), e_memout[0]);
			wait_group_events(1, e_memout);
		}
	}
}
