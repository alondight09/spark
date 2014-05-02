
TARGET=logistic
DST=computer0

curr_dir=$(shell pwd)
src_dir=$(TARGET)_lpp/pkg/arm/zc706/bin

all: $(src_dir)/$(TARGET)_lpp.xclbin
	rm -rf ${TARGET}_lpp
	picasso ${TARGET}.tcl

install: $(src_dir)/$(TARGET)_lpp.xclbin
	ssh $(DST) scp $(curr_dir)/$(src_dir)/$(TARGET)_lpp.xclbin root@fpga:/mnt/acc
