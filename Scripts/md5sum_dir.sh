#!/bin/bash
#~ https://stackoverflow.com/questions/1657232/how-can-i-calculate-an-md5-checksum-of-a-directory

nb_params=$#
if [ "$nb_params" -ge 1 ]; then
	if [ "$nb_params" -ge 2 ]
	then
		find "$1" -type f -name $2 -exec md5sum {} + | awk '{print $1}' | sort | md5sum
	else
		find "$1" -type f -exec md5sum {} + | awk '{print $1}' | sort | md5sum
	fi
fi
