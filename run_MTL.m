
function rtn = run_MTL(inname)

% add path of libsvm
addpath '~/softwares/libsvm-3.12/matlab/'
addpath '../shared_scripts/'

if nargin ==0
    names={'emotions','yeast','scene','enron','cal500','fp','cancer','medical','toy10','toy50'}
else
    names={inname}
end

for name=names
[sta,comres]=system('hostname');
if strcmp(comres(1:4),'dave')
    X=dlmread(sprintf('/fs/group/urenzyme/workspace/data/%s_features',name{1}));
    Y=dlmread(sprintf('/fs/group/urenzyme/workspace/data/%s_targets',name{1}));
else
    X=dlmread(sprintf('../shared_scripts/test_data/%s_features',name{1}));
    Y=dlmread(sprintf('../shared_scripts/test_data/%s_targets',name{1}));
end

rand('twister', 0);

%------------
%
% preparing     
%
%------------
% example selection with meaningful features
Xsum=sum(X,2);
X=X(find(Xsum~=0),:);
Y=Y(find(Xsum~=0),:);
% label selection with two labels
Yuniq=[];
for i=1:size(Y,2)
    if size(unique(Y(:,i)),1)>1
        Yuniq=[Yuniq,i];
    end
end
Y=Y(:,Yuniq);

% feature normalization (tf-idf for text data, scale and centralization for other numerical features)
if or(strcmp(name{1},'medical'),strcmp(name{1},'enron')) 
    X=tfidf(X);
elseif ~(strcmp(name{1}(1:2),'to'))
    X=(X-repmat(min(X),size(X,1),1))./repmat(max(X)-min(X),size(X,1),1);
end

% change Y from -1 to 0: labeling (0/1)
Y(Y==-1)=0;

% length of x and y
Nx = length(X(:,1));
Ny = length(Y(1,:));

% stratified cross validation index
nfold = 3;
Ind = getCVIndex(Y,nfold);

% performance
perf=[];

% get dot product kernels from normalized features or just read precomputed kernels
if or(strcmp(name{1},'fp'),strcmp(name{1},'cancer'))
    K=dlmread(sprintf('/fs/group/urenzyme/workspace/data/%s_kernel',name{1}));
else
    K = X * X'; % dot product
    K = K ./ sqrt(diag(K)*diag(K)');    %normalization diagonal is 1
end

%------------
%
% SVM, single label        
%
%------------
Ypred = [];
YpredVal = [];
% iterate on targets (Y1 -> Yx -> Ym)
for i=1:Ny
    % nfold cross validation
    Ycol = [];
    YcolVal = [];
    for k=1:nfold
        Itrain = find(Ind ~= k);
        Itest  = find(Ind == k);
        % training & testing with kernel
        if strcmp(name{1}(1:2),'to')
                svm_c=0.01;
        elseif strcmp(name{1},'cancer')
                svm_c=5;
        else
                svm_c=0.5;
        end
        model = svmtrain(Y(Itrain,i),[(1:numel(Itrain))',K(Itrain,Itrain)],sprintf('-b 1 -q -c %.2f -t 4',svm_c));
        [Ynew,acc,YnewVal] = svmpredict(Y(Itest,k),[(1:numel(Itest))',K(Itest,Itrain)],model,'-b 1');
        [Ynew] = svmpredict(Y(Itest,k),[(1:numel(Itest))',K(Itest,Itrain)],model);
        Ycol = [Ycol;[Ynew,Itest]];
        if size(YnewVal,2)==2
            YcolVal = [YcolVal;[YnewVal(:,abs(model.Label(1,:)-1)+1),Itest]];
        else
            YcolVal = [YcolVal;[zeros(numel(Itest),1),Itest]];
        end
    end
    Ycol = sortrows(Ycol,size(Ycol,2));
    Ypred = [Ypred,Ycol(:,1)];
    YcolVal = sortrows(YcolVal,size(YcolVal,2));
    YpredVal = [YpredVal,YcolVal(:,1)];
end
% performance of svm
[acc,vecacc,pre,rec,f1,auc1,auc2]=get_performance(Y,Ypred,YpredVal);
perf=[perf;[acc,vecacc,pre,rec,f1,auc1,auc2]];perf


%------------
%
% MTL
%
%------------
% parameter selection
if ~or(strcmp(name{1},'fp'),strcmp(name{1},'cancer'))
    K=X'; 
end
gammas=[100,10,5,1,0.5,0.1,0.01,0.001];
iterations=10;
if strcmp(name{1},'cancer')
    iterations=5
end
method_str='feat';
fname=sprintf('mtltmp_%s',name{1});
epsilon_init=1;
cv_size=1;
Dini=diag(repmat(1/size(K,1),size(K,1),1));
Dini(size(Dini,1),size(Dini,1))=Dini(size(Dini,1),size(Dini,1))+1-sum(sum(Dini));
Isel = randsample(1:size(K,2),ceil(size(K,2)*.03));
IselTrain=Isel(1:ceil(numel(Isel)/3*2));
IselTest=Isel(ceil(numel(Isel)/3*2+1):numel(Isel));
selRes=gammas*0;
for i=1:numel(gammas)
    gamma=gammas(i);
    trainx = K(:,repmat(IselTrain,1,size(Y,2)));
    testx = K(:,repmat(IselTest,1,size(Y,2)));
    task_indexes=[1:length(IselTrain):length(IselTrain)*size(Y,2)];
    task_indexes_test=[1:length(IselTest):length(IselTest)*size(Y,2)];
    Y_tr = Y(IselTrain,:);  Y_tr=reshape(Y_tr,numel(Y_tr),1);
    Y_ts = Y(IselTest,:); Y_ts=reshape(Y_ts,numel(Y_ts),1);
    % running
    rtn = code_example(gamma,trainx,Y_tr,testx,Y_ts,task_indexes,task_indexes_test,cv_size,Dini,iterations,method_str, epsilon_init, fname);
    selRes(i)=sum(sum(Y_ts==(rtn>=0.5)));
end
gamma=gammas(find(selRes==max(selRes)));
if numel(gamma) >1
    gamma=gamma(1);
end

pa=[selRes;gammas]
dlmwrite(sprintf('../parameters/%s_paraMTL',name{1}),pa)

% running
iterations=10;
Dini=diag(repmat(1/size(K,1),size(K,1),1));
Dini(size(Dini,1),size(Dini,1))=Dini(size(Dini,1),size(Dini,1))+1-sum(sum(Dini));
YpredVal=[];
for k=1:nfold
    Itrain = find(Ind ~= k);
    Itest  = find(Ind == k);
    trainx = K(:,repmat(Itrain,1,size(Y,2)));
    testx = K(:,repmat(Itest,1,size(Y,2)));
    task_indexes=[1:length(Itrain):length(Itrain)*size(Y,2)];
    task_indexes_test=[1:length(Itest):length(Itest)*size(Y,2)];
    Y_tr = Y(Itrain,:); Y_tr=reshape(Y_tr,numel(Y_tr),1);
    Y_ts = Y(Itest,:); Y_ts=reshape(Y_ts,numel(Y_ts),1);
    % running
    rtn = code_example(gamma,trainx,Y_tr,testx,Y_ts,task_indexes,task_indexes_test,cv_size,Dini,iterations,method_str, epsilon_init, fname);
    Ypred_ts_val=reshape(rtn,length(Itest),size(Y,2));
    YpredVal = [YpredVal;[Ypred_ts_val,Itest]];
end
YpredVal = sortrows(YpredVal,size(YpredVal,2));
YpredVal = YpredVal(:,1:size(Y,2));

% auc & roc mtl
[acc,vecacc,pre,rec,f1,auc1,auc2]=get_performance(Y,(YpredVal>=0.5),YpredVal);
perf=[perf;[acc,vecacc,pre,rec,f1,auc1,auc2]];perf
dlmwrite(sprintf('../predictions/%s_predValMTL',name{1}),YpredVal)
dlmwrite(sprintf('../predictions/%s_predBinMTL',name{1}),YpredVal>=0.5)

% save results
dlmwrite(sprintf('../results/%s_perfMTL',name{1}),perf)
end

rtn = [];
end





