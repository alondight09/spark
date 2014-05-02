#include "header.h"
//#include "Accelerator.h"
//#include "FPGA.h" 
#include "AccManager.h"
//
int main()
{
	AccManager * acc_mngr = new AccManager();
	
	acc_mngr -> init_socket(); 
	acc_mngr -> run();


	cout << "done" << endl;

	return 0;

}
