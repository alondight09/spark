export TARGET=logistic
PWD=${CURDIR}

HOSTS=hostlist
host_list=$(shell cat $(HOSTS))

accService=../accService/arm/accService_host/pkg/arm/zc706/bin/accService_host.exe 
accKernel=../accKernel/${TARGET}/${TARGET}_lpp/pkg/arm/zc706/bin/${TARGET}_lpp.xclbin

$(accService):
	make -C ../accService/arm

$(accKernel):
	make -C ../accKernel/${TARGET}

setup: $(accService) $(accKernel)
	for computer in $(host_list); do \
		echo processing $$computer; \
		ssh $$computer ssh root@fpga cp /mnt/embedded_root/platform.xml /mnt/acc; \
		ssh $$computer 'scp ${PWD}/$(accService) root@fpga:/mnt/acc'; \
		ssh $$computer 'scp ${PWD}/$(accKernel) root@fpga:/mnt/acc'; \
		ssh $$computer 'scp ${PWD}/*.sh root@fpga:/mnt/acc'; \
	done

start: $(accService) $(accKernel)
	for computer in $(host_list) ; do \
		echo "processing $$computer"; \
		ssh $$computer 'ssh root@fpga "cd /mnt/acc; sh stop.sh accService"'; \
		ssh $$computer 'ssh root@fpga "cd /mnt/acc; sh start.sh accService"'& \
	done

stop:
	for computer in $(host_list) ; do \
		echo "processing $$computer"; \
		ssh $$computer 'ssh root@fpga "cd /mnt/acc; sh stop.sh accService"'; \
	done
