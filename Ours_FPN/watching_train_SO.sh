var=0
while [ $var -eq 0 ]
do
    count=0
    for i in $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    do
        if [ $i -lt 600 ]
        then
            echo 'GPU'$count' is avaiable'
			python train.py --device_id $count
            var=1
            break
        fi
        count=$(($count+1))
	done		
done