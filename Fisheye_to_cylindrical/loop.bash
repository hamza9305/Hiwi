#!/bin/bash

direc1=/mnt/data/users/haahm/theostereo/*/
not_req='depth_exr_abs'
direc2=/mnt/data_on_nvme2/home/haahm/Dataset/THEOStereo_Cyc_New/*/

folders=($direc1 $direc2)

#for d in "${folders[@]}"; do
#  echo "$d"
#done



for dir in "${folders[@]}" ;
do
 #echo $dirs
  for subdir in $dir*/;
  do
   #basename $subdir
   #echo $subdir
   name1="$(basename "$subdir")"
   if [ "$name1" == "$not_req" ]
   then
   	echo 'Directory not required'
   else
   	echo $subdir
   fi
  done
done
