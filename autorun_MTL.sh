

#for inname in 'emotions' 'yeast' 'scene' 'enron' 'cal500' 'fp' 'cancer' 'medical' 'toy10' 'toy50'
for inname in 'toy10'
do
    nohup matlab -nodesktop -nosplash -r "run_MTL '$inname'" &
    sleep 2s
done
wait
rm nohup.out
