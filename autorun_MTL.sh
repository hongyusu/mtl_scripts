

for inname in {'emotions','yeast','scene','enron','cal500','fp','cancer','medical','toy10','toy50'}
for inname in {'toy10'}
do
    echo $inname
    nohup matlab -nodesktop -nosplash -r 'run_MTL.m $inname' $
done
wait
