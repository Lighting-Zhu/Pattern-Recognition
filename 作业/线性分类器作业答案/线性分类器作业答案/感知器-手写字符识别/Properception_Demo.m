clc;
clear all;
close all;
%%
% 读取数据
load ('./mnist to matlab/train_images.mat');
load ('./mnist to matlab/train_labels.mat');
load ('./mnist to matlab/test_images.mat');
load ('./mnist to matlab/test_labels.mat');
%% Parameters
train_num = 300;
test_num = 500;
Maxepoch = 300;%最大迭代次数
lr = 0.1;%学习率
Selectednumber = [6;8];%识别6和8
N_train = size(train_labels1,2);
N_test = size(test_labels1,2);
%% generate training data
train_c = 1;
for i = 1:N_train
    if train_labels1(i)==Selectednumber(1) || train_labels1(i)==Selectednumber(2)
        Train_data(train_c,:) = reshape(train_images(:,:,i),1,numel(train_images(:,:,i)));
        Train_labels(train_c,1) = train_labels1(i);
        train_c = train_c + 1;
    end
    if train_c>train_num
        break;
    end
end
%% generate test data
test_c = 1;
for i = 1:N_test  
    if test_labels1(i)==Selectednumber(1) || test_labels1(i)==Selectednumber(2)
        Test_data(test_c,:) = reshape(test_images(:,:,i),1,numel(test_images(:,:,i)));
        Test_labels(test_c,1) = test_labels1(i);
        test_c = test_c + 1;
    end
    if test_c>test_num
        break;
    end
end
%% 规范化
% number 6 :w1
% number 8: w2
Train_data = [Train_data,ones(train_num,1)];
Train_data(Train_labels==Selectednumber(2),:) = -Train_data(Train_labels==Selectednumber(2),:);
%% Train Stage
a_cur=perceptionLearn(Train_data,lr,Maxepoch);

%% Test Stage
Test_data = [Test_data,ones(test_num,1)];
pred = a_cur*Test_data';
pred(pred>0)=6;
pred(pred<=0)=8;
pred = pred';

accu = sum(pred==Test_labels)/test_num;
fprintf('Based on the condition(Training number is %d, learning rate is %f),On %d testing samples,the accurate rate is: %f\n',train_num,lr,test_num,accu);
