#!/bin/sh

export LANG=en_US.UTF8

echo "to kill $1_host.exe"
pid=`ps aux | grep $1_host.exe | grep -v grep | tr -s ' ' | cut -d ' ' -f2`
if [[ $pid != "" ]]; then
	echo "killing $pid"
	kill $pid
fi

