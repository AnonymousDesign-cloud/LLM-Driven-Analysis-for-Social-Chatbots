clc;clear;close all;	
load('R_19_May_2026_15_01_12.mat')	
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
	
	
	
	
disp('LightGBM分类')	
t1=clock; 	
 num_leaves1=10;           %叶子节点数	
 max_depth1=7;            %最大深度	
learning_rate1=0.9;      %学习率	
num_max_iter1=60;      %最大迭代次数   	
num_early_stopping1=30;    %早停次数   	
	
	
	
	
	
[LightGBM_importance,Mdl, best_iter, train_iter]=train_LGB_C(train_x_feature_label_norm,train_y_feature_label,vaild_x_feature_label_norm,vaild_y_feature_label,num_leaves1,max_depth1,learning_rate1,num_max_iter1,num_early_stopping1);   	
	
y_train_predict_gailv=predict_LGB(train_x_feature_label_norm,Mdl, best_iter);[~, y_train_predict] = max(y_train_predict_gailv, [], 2);  %训练集预测结果	
y_vaild_predict_gailv=predict_LGB(vaild_x_feature_label_norm,Mdl, best_iter); [~, y_vaild_predict] = max(y_vaild_predict_gailv, [], 2);  %验证集预测结果	
y_test_predict_gailv=predict_LGB(test_x_feature_label_norm,Mdl, best_iter);[~, y_test_predict] = max(y_test_predict_gailv, [], 2);  %测试集预测结果	
t2=clock;	
 Time=t2(3)*3600*24+t2(4)*3600+t2(5)*60+t2(6)-(t1(3)*3600*24+t1(4)*3600+t1(5)*60+t1(6));       	
%特征重要性绘图	
figure	
bar_plot_f1=bar(LightGBM_importance);   %  重要性衡量	
bar_plot_f1.FaceColor = 'flat';	
color_get=G_out_data.color_get;	
for i=1:length(LightGBM_importance)	
      bar_plot_f1.CData(i,:)=[color_get(1+i*(floor(length(color_get)/length(LightGBM_importance))-1),:)];	
end	
	
xticks([1:length(LightGBM_importance)])	
xticklabels(data_biao1())	
ylabel('LightGBM Importance')	
	
figure	
for i = 1: size(train_iter, 1)	
     subplot(1, size(train_iter, 1), i)	
     plot(squeeze(train_iter(i, :, :))', 'LineWidth', 1)	
     title('训练过程Log_Loss迭代曲线')	
     xline(best_iter+1,'--',{'best iter'})	
end	
 legend('train set', 'vaild set')	
grid;set(gcf,'color','w')	
xlabel('iter');  ylabel('RMSE')	
	
	
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
[score_train]=predict_LGB(train_x_feature_label_norm,Mdl, best_iter);  %训练集预测结果	
[score_vaild]=predict_LGB(vaild_x_feature_label_norm,Mdl, best_iter); %验证集预测结果	
[score_test]=predict_LGB(test_x_feature_label_norm,Mdl, best_iter); %测试集预测结果	
	
	
	
	
	
[X_ROC_train,Y_ROC_train,T_ROC_train,AUC_ROC_train] = perfcurve(train_y_feature_label,score_train(:,1),1);	
rocObj_train = rocmetrics(train_y_feature_label,score_train(:,1),1);	
	
figure	
plot(rocObj_train)	
title('Train ROC')	
%	
[X_ROC_vaild,Y_ROC_vaild,T_ROC_vaild,AUC_ROC_vaild] = perfcurve(vaild_y_feature_label,score_vaild(:,1),1);	
rocObj_vaild = rocmetrics(vaild_y_feature_label,score_vaild(:,1),1);	
	
figure	
plot(rocObj_vaild)	
title('Vaild ROC')	
%	
[X_ROC_test,Y_ROC_test,T_ROC_test,AUC_ROC_test] = perfcurve(test_y_feature_label,score_test(:,1),1);	
rocObj_test = rocmetrics(test_y_feature_label,score_test(:,1),1);	
figure	
plot(rocObj_test)	
title('Test ROC')	
	
	
%% K折验证	
x_feature_label_norm_all=(x_feature_label-x_mu)./x_sig;    %x特征	
y_feature_label_norm_all=y_feature_label;	
Kfold_num=G_out_data.Kfold_num;	
cv = cvpartition(size(x_feature_label_norm_all, 1), 'KFold', Kfold_num); % Split into K folds	
for k = 1:Kfold_num	
    trainingIdx = training(cv, k);	
    validationIdx = test(cv, k);	
     x_feature_label_norm_all_traink=x_feature_label_norm_all(trainingIdx,:);	
   y_feature_label_norm_all_traink=y_feature_label_norm_all(trainingIdx,:);	
	
   x_feature_label_norm_all_testk=x_feature_label_norm_all(validationIdx,:);	
   y_feature_label_norm_all_testk=y_feature_label_norm_all(validationIdx,:);	
	
   [~,Mdlkf, best_iter1, ~]=train_LGB_C(x_feature_label_norm_all_traink,y_feature_label_norm_all_traink,x_feature_label_norm_all_testk,y_feature_label_norm_all_testk,num_leaves1,max_depth1,learning_rate1,num_max_iter1,num_early_stopping1);	
	
   Mdl_kfold{1,k}=Mdlkf;	
   y_test_predict_norm_all_testk=predict_LGB(x_feature_label_norm_all_testk,Mdlkf,best_iter1); %测试集预测结果	
   [~, y_test_predict_all_testk] = max(y_test_predict_norm_all_testk, [], 2);	
   	
   test_kfold=sum((y_test_predict_all_testk==y_feature_label_norm_all_testk))/length(y_feature_label_norm_all_testk);% 采用的accuracy	
   AUC_kfold(k)=test_kfold;	
	
	
	
end	
	
	
	
% k折验证结果绘图	
figure('color',[1 1 1]);	
	
color_set=[0.4353    0.5137    0.7490];	
plot(1:length(AUC_kfold),AUC_kfold,'--p','color',color_set,'Linewidth',1.3,'MarkerSize',6,'MarkerFaceColor',color_set,'MarkerFaceColor',[0.3,0.4,0.5]);	
grid on;	
box off;	
grid off;	
ylim([0.92*min(AUC_kfold),1.2*max(AUC_kfold)])	
xlabel('kfoldnum')	
ylabel('accuracy')	
xticks(1:length(AUC_kfold))	
set(gca,'Xgrid','off');	
set(gca,'Linewidth',1);	
set(gca,'TickDir', 'out', 'TickLength', [.005 .005], 'XMinorTick', 'off', 'YMinorTick', 'off');	
yline(mean(AUC_kfold),'--')	
%小窗口柱状图的绘制	
axes('Position',[0.6,0.65,0.25,0.25],'box','on'); % 生成子图	
GO = bar(1:length(AUC_kfold),AUC_kfold,1,'EdgeColor','k');	
GO(1).FaceColor = color_set;	
xticks(1:length(AUC_kfold))	
xlabel('kfoldnum')	
ylabel('accuracy')	
disp('****************************************************************************************') 	
disp([num2str(Kfold_num),'折验证预测准确率accuracy结果：'])	
disp(AUC_kfold) 	
disp([num2str(Kfold_num),'折验证  ','accuracy均值为： ' ,num2str(mean(AUC_kfold)),'    accuracy标准差为： ' ,num2str(std(AUC_kfold))]) 	
	
	
	
	
%% SHAP分析	
num_set=200;    %这个值越大运行时间越长	
if size(test_x_feature_label_norm,1)>num_set	
    num_sample_get=num_set;	
    listshap_sample=round(1:size(test_x_feature_label_norm,1)/num_sample_get:size(test_x_feature_label_norm,1));	
else	
    listshap_sample=1:size(test_x_feature_label_norm,1);	
end	
 	
num_sample_get_train=length(listshap_sample)*4;	
if size(train_x_feature_label_norm,1)<num_sample_get_train	
      num_sample_get_train=size(train_x_feature_label_norm,1); 	
end 	
 listshap_sample_train=round(1:size(train_x_feature_label_norm,1)/num_sample_get_train:size(train_x_feature_label_norm,1)); 	
LimePredict_symble=G_out_data.LimePredict_symble;	
others=G_out_data.others;	
myPredict = @(x) LimePredict(Mdl,x,LimePredict_symble,others);	
	
	
shapValues=[];	
	
explainer = shapley(myPredict,train_x_feature_label_norm(listshap_sample_train,:),'QueryPoint',test_x_feature_label_norm(listshap_sample,:),'MaxNumSubsets', 200);	
ShapleyValues1=explainer.ShapleyValues;	
shapValues=table2array(ShapleyValues1(:,2:end))';	
	
index_name_plot=G_out_data.index_name_plot;	
color_get=G_out_data.color_get;	
figure('Position',[200,200,800,400])	
X_get=1:size(shapValues,2);	
X_get1=repmat(X_get,size(shapValues,1),1);	
for n=1:size(test_x_feature_label_norm,2)	
    train_x_shap=test_x_feature_label_norm(listshap_sample,n); train_x_shap1=train_x_shap;                	
    c_index=train_x_shap-min(train_x_shap1); c_index1=ceil(c_index/max(c_index)*length(color_get))+1;	
    if isnan(c_index1)	
        c_index1=ones(length(c_index1),1);	
    end	
    color_shap1(:,n)=c_index1;	
end	
 shapValues_imptance=mean(abs(shapValues));	
 [shapValues_imptance1,shapValues_sort]=sort(shapValues_imptance, 'descend');	
 yline(0,'-','LineWidth',1.1,'Color',[0.6,0.6,0.6])	
 hold on	
	
  for i=1:size(shapValues,2)	
     s(i)=swarmchart(X_get1(:,(i)),shapValues(:,shapValues_sort(i)),15,color_shap1(:,shapValues_sort(i)),'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5);	
    hold on	
     s(i).XJitterWidth = 0.7;	
 end	
	
 colormap(color_get)	
cbtick= linspace(1,256,2);	
colorbar_index=colorbar('Ticks',cbtick,'TickLabels',{'Low','High'});	
colorbar_index.Label.String = 'Feature value';	
colorbar_index.Label.FontSize = 12;	
xticks(1:size(shapValues,2))	
xticklabels(index_name_plot(shapValues_sort))	
set(gca,'LineWidth',1.2)	
ylabel('SHAP value (impact on model output) ')	
	
figure('Position',[200,200,600,350]) ;	
bar_plot_f=bar(shapValues_imptance1);   %  重要性衡量	
bar_plot_f.FaceColor = 'flat';	
for i=1:length(shapValues_imptance1)	
     bar_plot_f.CData(i,:)=[color_get(1+i*(floor(length(color_get)/length(shapValues_imptance))-1),:)];	
end	
	
xticks([1:length(shapValues_imptance1)])	
	
xticklabels(index_name_plot(shapValues_sort))	
title('SHAP analysis')	
ylabel('Predictor importance estimates');	
xlabel('Predictors');	
	
featureNames=index_name_plot;	
targetVarName=data_biao1{1,end};	
SHAP_test_x_feature_label=test_x_feature_label_norm(listshap_sample,:).*x_sig+x_mu;	
SHAP_test_y_feature_label=test_y_feature_label(listshap_sample,:);	
feature_show_max=12;	
[shapValues1]=SHAP_Optimized_plot(shapValues,SHAP_test_x_feature_label,SHAP_test_y_feature_label,featureNames,targetVarName,color_get,feature_show_max);	
	
