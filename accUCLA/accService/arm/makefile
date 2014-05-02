
TARGET=accService
DST=computer0

curr_dir=$(shell pwd)
src_dir=$(TARGET)_host/pkg/arm/zc706/bin

all: $(src_dir)/$(TARGET)_host.exe

$(src_dir)/$(TARGET)_host.exe: $(TARGET).c
	rm -rf ${TARGET}_host
	picasso ${TARGET}.tcl

install: $(src_dir)/$(TARGET)_host.exe
	ssh $(DST) scp $(curr_dir)/$(src_dir)/$(TARGET)_host.exe root@fpga:/mnt/acc
