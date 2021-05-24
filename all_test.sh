rm timing_raw.log timing_encrypt.log
for ((i=0; i< $1; i=i+1))
do
	./app -encrypt=0 >> timing_raw.log
done
for ((i=0; i< $1; i=i+1))
do
	./app -encrypt=1 >> timing_encrypt.log
done
echo "DONE. BRIEF(NO encrypt):"
cat timing_raw.log | grep time
echo "WITH encrypt enabled:"
cat timing_encrypt.log | grep time

