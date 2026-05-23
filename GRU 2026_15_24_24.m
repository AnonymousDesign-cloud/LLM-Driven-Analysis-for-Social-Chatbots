clc;clear;close all;	
load('R_19_May_2026_15_24_24.mat')	
random_seed=G_out_data.random_seed ;  %界面设置的种子数 	
rng(random_seed)  %固定随机数种子 	
	
data_str=G_out_data.data_path_str ;  %读取数据的路径 	
dataO=readtable(data_str,'VariableNamingRule','preserve'); %读取数据 	
data1=dataO(:,2:end);test_data=table2cell(dataO(1,2:end));	
for i=1:length(test_data)	
      if ischar(test_data{1,i})==1	
          index_la(i)=1;     %char类型	
      elseif isnumeric(test_data{1,i})==1	
          index_la(i)=2;     %double类型	
      else	
        index_la(i)=0;     %其他类型	
     end 	
end	
index_char=find(index_la==1);index_double=find(index_la==2);	
 %% 数值类型数据处理	
if length(index_double)>=1	
    data_numshuju=table2array(data1(:,index_double));	
    index_double1=index_double;	
	
    index_double1_index=1:size(data_numshuju,2);	
    data_NAN=(isnan(data_numshuju));    %找列的缺失值	
    num_NAN_ROW=sum(data_NAN);	
    index_NAN=num_NAN_ROW>round(0.2*size(data1,1));	
    index_double1(index_NAN==1)=[]; index_double1_index(index_NAN==1)=[];	
    data_numshuju1=data_numshuju(:,index_double1_index);	
    data_NAN1=(isnan(data_numshuju1));  %找行的缺失值	
    num_NAN__COL=sum(data_NAN1');	
    index_NAN1=num_NAN__COL>0;	
    index_double2_index=1:size(data_numshuju,1);	
    index_double2_index(index_NAN1==1)=[];	
    data_numshuju2=data_numshuju1(index_double2_index,:);	
    index_need_last=index_double1;	
 else	
    index_need_last=[];	
    data_numshuju2=[];	
end	
%% 文本类型数据处理	
	
data_shuju=[];	
 if length(index_char)>=1	
  for j=1:length(index_char)	
    data_get=table2array(data1(index_double2_index,index_char(j)));	
    data_label=unique(data_get);	
    if j==length(index_char)	
       data_label_str=data_label ;	
    end    	
	
     for NN=1:length(data_label)	
            idx = find(ismember(data_get,data_label{NN,1}));  	
            data_shuju(idx,j)=NN; 	
     end	
  end	
 end	
label_all_last=[index_char,index_need_last];	
[~,label_max]=max(label_all_last);	
 if(label_max==length(label_all_last))	
     str_label=0; %标记输出是否字符类型	
     data_all_last=[data_shuju,data_numshuju2];	
     label_all_last=[index_char,index_need_last];	
 else	
    str_label=1;	
    data_all_last=[data_numshuju2,data_shuju];	
    label_all_last=[index_need_last,index_char];     	
 end	
 data=data_all_last;	
 data_biao_all=data1.Properties.VariableNames;	
 for j=1:length(label_all_last)	
    data_biao{1,j}=data_biao_all{1,label_all_last(j)};	
 end	
	
% 异常值检测	
	
 unique_index_ab=G_out_data.unique_index_ab; 	
 data(:,unique_index_ab)=[];	
 label_all_last(unique_index_ab)=[];	
 data_biao1=data_biao; data_biao1(unique_index_ab)=[]; 	
	
data=data;	
	
%%  特征处理 特征选择或者降维	
	
 A_data1=data;	
	
 select_feature_num=G_out_data.select_feature_num1;   %特征选择的个数	
index_name=data_biao1;	
print_index_name=[]; 	
[B,~] = lasso(A_data1(:,1:end-1),A_data1(:,end),'Alpha',1); 	
L_B=(B~=0);   SL_B=sum(L_B); [~,index_L1]=min(abs(SL_B-select_feature_num)); 	
feature_need_last=find(L_B(:,index_L1)==1);  	
data_select=[A_data1(:,feature_need_last),A_data1(:,end)];	
	
	
for NN=1:length(feature_need_last) 	
   print_index_name{1,NN}=index_name{1,feature_need_last(NN)};	
end 	
disp('选择特征');disp(print_index_name)  	
	
	
	
%% 数据划分	
x_feature_label=data_select(:,1:end-1);    %x特征	
y_feature_label=data_select(:,end);          %y标签	
index_label1=randperm(size(x_feature_label,1));	
index_label=G_out_data.spilt_label_data;  % 数据索引	
if isempty(index_label)	
     index_label=index_label1;	
end	
spilt_ri=G_out_data.spilt_rio;  %划分比例 训练集:验证集:测试集	
train_num=round(spilt_ri(1)/(sum(spilt_ri))*size(x_feature_label,1));          %训练集个数	
vaild_num=round((spilt_ri(1)+spilt_ri(2))/(sum(spilt_ri))*size(x_feature_label,1)); %验证集个数	
 %训练集，验证集，测试集	
train_x_feature_label=x_feature_label(index_label(1:train_num),:);	
train_y_feature_label=y_feature_label(index_label(1:train_num),:);	
vaild_x_feature_label=x_feature_label(index_label(train_num+1:vaild_num),:);	
vaild_y_feature_label=y_feature_label(index_label(train_num+1:vaild_num),:);	
test_x_feature_label=x_feature_label(index_label(vaild_num+1:end),:);	
test_y_feature_label=y_feature_label(index_label(vaild_num+1:end),:);	
%Zscore 标准化	
%训练集	
x_mu = mean(train_x_feature_label);  x_sig = std(train_x_feature_label); 	
train_x_feature_label_norm = (train_x_feature_label - x_mu) ./ x_sig;    % 训练数据标准化	
y_mu = mean(train_y_feature_label);  y_sig = std(train_y_feature_label); 	
train_y_feature_label_norm = (train_y_feature_label - y_mu) ./ y_sig;    % 训练数据标准化  	
%验证集	
vaild_x_feature_label_norm = (vaild_x_feature_label - x_mu) ./ x_sig;    %验证数据标准化	
vaild_y_feature_label_norm=(vaild_y_feature_label - y_mu) ./ y_sig;  %验证数据标准化	
%测试集	
test_x_feature_label_norm = (test_x_feature_label - x_mu) ./ x_sig;    % 测试数据标准化	
test_y_feature_label_norm = (test_y_feature_label - y_mu) ./ y_sig;    % 测试数据标准化  	
	
%% 参数设置	
num_pop=G_out_data.num_pop1;   %种群数量	
num_iter=G_out_data.num_iter1;   %种群迭代数	
method_mti=G_out_data.method_mti1;   %优化方法	
BO_iter=G_out_data.BO_iter;   %贝叶斯迭代次数	
min_batchsize=G_out_data.min_batchsize;   %batchsize	
max_epoch=G_out_data.max_epoch1;   %maxepoch	
hidden_size=G_out_data.hidden_size1;   %hidden_size	
attention_label=G_out_data.attention_label;   %注意力机制标签	
attention_head=G_out_data.attention_head;   %注意力机制设置	
	
%% 数据增强部分	
get_mutiple=G_out_data.get_mutiple;  %数据增加倍数	
methodchoose=1; 	
origin_data=[train_x_feature_label_norm;vaild_x_feature_label_norm]; 	
origin_data_label=[train_y_feature_label;vaild_y_feature_label]; 	
[SyntheticData,Synthetic_label]=generate_classdata(origin_data,origin_data_label,methodchoose,get_mutiple); 	
% 绘制生成后数据样本图	
figure_data_generate(origin_data,SyntheticData,origin_data_label,Synthetic_label)	
X_new_DATA=[origin_data;SyntheticData];             %生成的X特征数据	
Y_new_DATA=[origin_data_label;Synthetic_label];  %生成的Y标签数据	
	
syn_spilt=round(spilt_ri(1)/(spilt_ri(1)+spilt_ri(2))*length(Y_new_DATA));	
syn_index=randperm(length(Y_new_DATA));	
%以下将生成的数据随机分配到训练集和验证集中	
train_x_feature_label_norm=X_new_DATA(syn_index(1:syn_spilt),:);	
vaild_x_feature_label_norm=X_new_DATA(syn_index(syn_spilt+1:end),:);	
train_y_feature_label=Y_new_DATA(syn_index(1:syn_spilt),:);	
vaild_y_feature_label=Y_new_DATA(syn_index(syn_spilt+1:end),:);	
train_x_feature_label=train_x_feature_label_norm.*x_sig+x_mu;	
vaild_x_feature_label=vaild_x_feature_label_norm.*x_sig+x_mu;	
%数据生成输出数据	
train_x_feature_label_aug=(train_x_feature_label_norm.*x_sig)+x_mu;	
vaild_x_feature_label_aug=(vaild_x_feature_label_norm.*x_sig)+x_mu;	
%总体生成数据+原数据保存在以下的 augdata_all 数据里面	
augdata_all=[train_x_feature_label_aug,train_y_feature_label;vaild_x_feature_label_aug,vaild_y_feature_label;test_x_feature_label,test_y_feature_label];	
	
	
%% 算法处理块	
	
	
	
hidden_size=G_out_data.hidden_size1;    %神经网络隐藏层	
disp('GRU分类') 	
t1=clock; 	
max_epoch=G_out_data.max_epoch1;    %神经网络隐藏层	
for i = 1: size(train_x_feature_label,1)      %修改输入变成元胞形式	
    p_train1{i, 1} = (train_x_feature_label_norm(i,:))';	
end	
for i = 1 : size(test_x_feature_label,1)	
     p_test1{i, 1}  = (test_x_feature_label_norm(i,:))';	
	
end	
	
for i = 1 : size(vaild_x_feature_label,1)	
     p_vaild1{i, 1}  = (vaild_x_feature_label_norm(i,:))';	
end	
	
  layers = [sequenceInputLayer(size(train_x_feature_label_norm,2))               % 建立输入层    	
  gruLayer(hidden_size(1), 'OutputMode', 'sequence')      % LSTM层	
 reluLayer  	
 dropoutLayer(0.2)                                 % 防止过拟合  	
  gruLayer(hidden_size(2),'OutputMode','last')	
  dropoutLayer(0.2);                                % 回归层	
  fullyConnectedLayer(length(unique(train_y_feature_label)))	
  softmaxLayer  	
 classificationLayer];   	
	
  options = trainingOptions('adam','Shuffle','every-epoch',...	
  'MaxEpochs',max_epoch, ...,	
   'MiniBatchSize',min_batchsize,...	
  'InitialLearnRate',0.001,...	
  'ValidationFrequency',20, ...	
  'Plots','training-progress');	
   [Mdl,Loss] = trainNetwork(p_train1, categorical(train_y_feature_label), layers, options);	
 y_train_predict = double(classify(Mdl, p_train1));	
 y_vaild_predict =  double(classify(Mdl, p_vaild1));	
 y_test_predict =  double(classify(Mdl, p_test1));	
 t2=clock;	
 Time=t2(3)*3600*24+t2(4)*3600+t2(5)*60+t2(6)-(t1(3)*3600*24+t1(4)*3600+t1(5)*60+t1(6));	
	
graph= layerGraph(Mdl.Layers); figure; plot(graph) 	
analyzeNetwork(Mdl)	
	
	
figure	
subplot(2, 1, 1)	
	
plot(1 : length(Loss.TrainingAccuracy), Loss.TrainingAccuracy, '-', 'LineWidth', 1)	
xlabel('迭代次数'); ylabel('准确率');legend('训练集准确率');title ('训练集准确率迭代曲线');grid;set(gcf,'color','w')	
	
subplot(2, 1, 2)	
plot(1 : length(Loss.TrainingLoss), Loss.TrainingLoss, '-', 'LineWidth', 1)	
xlabel('迭代次数');ylabel('损失函数');legend('训练集损失值');title ('训练集损失函数曲线');grid;set(gcf,'color','w');	
	
	
	
	
disp(['运行时长: ',num2str(Time)])	
confMat_train = confusionmat(train_y_feature_label,y_train_predict);	
TP_train = diag(confMat_train);      TP_train=TP_train'; % 被正确分类的正样本 True Positives	
FP_train = sum(confMat_train, 1)  - TP_train;  %被错误分类的正样本 False Positives	
FN_train = sum(confMat_train, 2)' - TP_train;  % 被错误分类的负样本 False Negatives	
TN_train = sum(confMat_train(:))  - (TP_train + FP_train + FN_train);  % 被正确分类的负样本 True Negatives	
	
disp('训练集*******************************************************************************')	
accuracy_train = sum(TP_train) / sum(confMat_train(:)); accuracy_train(isnan(accuracy_train))=0; disp(['训练集accuracy：',num2str(mean(accuracy_train))])% Accuracy 	
precision_train = TP_train ./ (TP_train + FP_train); precision_train(isnan(precision_train))=0; disp(['训练集precision_train：',num2str(mean(precision_train))]) % Precision	
recall_train = TP_train ./ (TP_train + FN_train);recall_train(isnan(recall_train))=0; disp(['训练集recall_train：',num2str(mean(recall_train))])  % Recall / Sensitivity	
F1_score_train = 2 * (precision_train .* recall_train) ./ (precision_train + recall_train); F1_score_train(isnan(F1_score_train))=0;  disp(['训练集F1_score_train：',num2str(mean(F1_score_train))])   % F1 Score	
specificity_train = TN_train ./ (TN_train + FP_train); specificity_train(isnan(specificity_train))=0; disp(['训练集specificity_train：',num2str(mean(specificity_train))])  % Specificity	
	
disp('验证集********************************************************************************')	
confMat_vaild = confusionmat(vaild_y_feature_label,y_vaild_predict);	
TP_vaild = diag(confMat_vaild);      TP_vaild=TP_vaild'; % 被正确分类的正样本 True Positives	
FP_vaild = sum(confMat_vaild, 1)  - TP_vaild;  %被错误分类的正样本 False Positives	
FN_vaild = sum(confMat_vaild, 2)' - TP_vaild;  % 被错误分类的负样本 False Negatives	
TN_vaild = sum(confMat_vaild(:))  - (TP_vaild + FP_vaild + FN_vaild);  % 被正确分类的负样本 True Negatives	
accuracy_vaild = sum(TP_vaild) / sum(confMat_vaild(:)); accuracy_vaild(isnan(accuracy_vaild))=0; disp(['验证集accuracy：',num2str(accuracy_vaild)])% Accuracy 	
precision_vaild = TP_vaild ./ (TP_vaild + FP_vaild); precision_vaild(isnan(precision_vaild))=0; disp(['验证集precision_vaild：',num2str(mean(precision_vaild))]) % Precision	
recall_vaild = TP_vaild ./ (TP_vaild + FN_vaild); recall_vaild(isnan(recall_vaild))=0;  disp(['验证集recall_vaild：',num2str(mean(recall_vaild))])  % Recall / Sensitivity	
F1_score_vaild = 2 * (precision_vaild .* recall_vaild) ./ (precision_vaild + recall_vaild);  F1_score_vaild(isnan(F1_score_vaild))=0;  disp(['验证集F1_score_vaild：',num2str(mean(F1_score_vaild))])   % F1 Score	
specificity_vaild = TN_vaild ./ (TN_vaild + FP_vaild); specificity_vaild(isnan(specificity_vaild))=0; disp(['验证集specificity_vaild：',num2str(mean(specificity_vaild))])  % Specificity	
disp('测试集********************************************************************************') 	
confMat_test = confusionmat(test_y_feature_label,y_test_predict);	
TP_test = diag(confMat_test);      TP_test=TP_test'; % 被正确分类的正样本 True Positives	
FP_test = sum(confMat_test, 1)  - TP_test;  %被错误分类的正样本 False Positives	
FN_test = sum(confMat_test, 2)' - TP_test;  % 被错误分类的负样本 False Negatives	
TN_test = sum(confMat_test(:))  - (TP_test + FP_test + FN_test);  % 被正确分类的负样本 True Negatives	
	
accuracy_test = sum(TP_test) / sum(confMat_test(:)); accuracy_test(isnan(accuracy_test))=0; disp(['测试集accuracy：',num2str(accuracy_test)])% Accuracy	
precision_test = TP_test ./ (TP_test + FP_test);  precision_test(isnan(precision_test))=0; disp(['测试集precision_test：',num2str(mean(precision_test))]) % Precision	
recall_test = TP_test ./ (TP_test + FN_test); recall_test(isnan(recall_test))=0; disp(['测试集recall_test：',num2str(mean(recall_test))])  % Recall / Sensitivity	
F1_score_test = 2 * (precision_test .* recall_test) ./ (precision_test + recall_test); F1_score_test(isnan(F1_score_test))=0; disp(['测试集F1_score_test：',num2str(mean(F1_score_test))])   % F1 Score	
specificity_test = TN_test ./ (TN_test + FP_test); specificity_test(isnan(specificity_test))=0; disp(['测试集specificity_test：',num2str(mean(specificity_test))])  % Specificity	
	
disp('验证集+测试集 （没有用到优化可以直接当作整体的测试集）********************************************************************************') 	
test_y1=[vaild_y_feature_label;test_y_feature_label];y_test_predict1=[y_vaild_predict;y_test_predict];	
confMat_test1 = confusionmat(test_y1,y_test_predict1);	
TP_test1 = diag(confMat_test1);      TP_test1=TP_test1'; % 被正确分类的正样本 True Positives	
FP_test1 = sum(confMat_test1, 1)  - TP_test1;  %被错误分类的正样本 False Positives	
FN_test1 = sum(confMat_test1, 2)' - TP_test1;  % 被错误分类的负样本 False Negatives	
TN_test1 = sum(confMat_test1(:))  - (TP_test1 + FP_test1 + FN_test1);  % 被正确分类的负样本 True Negatives	
accuracy_test1 = sum(TP_test1) / sum(confMat_test1(:)); accuracy_test1(isnan(accuracy_test1))=0;  disp(['验证集+测试集accuracy：',num2str(accuracy_test1)])% Accuracy	
precision_test1 = TP_test1 ./ (TP_test1 + FP_test1);  precision_test1(isnan(precision_test1))=0;  disp(['验证集+测试集precision_test：',num2str(mean(precision_test1))]) % Precision	
recall_test1 = TP_test1 ./ (TP_test1 + FN_test1); recall_test1(isnan(recall_test1))=0;  disp(['验证集+测试集recall_test：',num2str(mean(recall_test1))])  % Recall / Sensitivity	
F1_score_test1 = 2 * (precision_test1 .* recall_test1) ./ (precision_test1 + recall_test1); F1_score_test1(isnan(F1_score_test1))=0; disp(['验证集+测试集F1_score_test：',num2str(mean(F1_score_test1))])   % F1 Score	
specificity_test1 = TN_test1 ./ (TN_test1 + FP_test1); specificity_test1(isnan(specificity_test1))=0; disp(['验证集+测试集specificity_test：',num2str(mean(specificity_test1))])  % Specificity	
	
%% 绘制ROC曲线	
layer = 'softmax';	
p_train2 = double(activations(Mdl,p_train1,layer,'OutputAs','rows')); %特征进行拓展	
p_vaild2 = double(activations(Mdl,p_vaild1, layer,'OutputAs','rows'));	
p_test2 = double(activations(Mdl,p_test1, layer,'OutputAs','rows'));	
%	
[X_ROC_train,Y_ROC_train,T_ROC_train,AUC_ROC_train] = perfcurve(train_y_feature_label,p_train2(:,1),1);	
rocObj_train = rocmetrics(train_y_feature_label,p_train2(:,1),1);	
figure	
plot(rocObj_train)	
title('Train ROC')	
%	
[X_ROC_vaild,Y_ROC_vaild,T_ROC_vaild,AUC_ROC_vaild] = perfcurve(vaild_y_feature_label,p_vaild2(:,1),1);	
rocObj_vaild = rocmetrics(vaild_y_feature_label,p_vaild2(:,1),1);	
figure	
plot(rocObj_vaild)	
title('Vaild ROC')	
%	
[X_ROC_test,Y_ROC_test,T_ROC_test,AUC_ROC_test] = perfcurve(test_y_feature_label,p_test2(:,1),1);	
rocObj_test = rocmetrics(test_y_feature_label,p_test2(:,1),1);	
figure	
plot(rocObj_test)	
title('Test ROC')	
