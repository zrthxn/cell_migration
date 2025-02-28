rm msd*.txt dac*.txt parameters*.txt len*.txt xx*.txt yy*.txt

for ((i = 0 ; i < 1000 ; i++))
do
 touch f$i.todo
done

export OMP_NUM_THREADS=1

make -j52 $(ls *.todo | sed s/.todo/.done/)

for i in $(ls *.done | sed "s/.done//") 
do 
 for s in msd.txt dac.txt parameters.txt len.txt msd_var.txt dac_var.txt len_var.txt xx.txt yy.txt
 do 
  cat $i/$s >> $s 
 done
done
  