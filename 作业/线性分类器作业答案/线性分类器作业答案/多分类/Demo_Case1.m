clc;
close all;
clear all;
%% 生成数据
randn('seed',2020);
mu1 = [0 3];
sigma1 = [0.5 0; 
         0 0.5];
data1 = mvnrnd(mu1,sigma1,300);

randn('seed',2021);
mu2 = [6 7];
sigma2 = [0.5 0; 
         0 0.5];
data2 = mvnrnd(mu2,sigma2,300);

randn('seed',2022);
mu3 = [5 -5];
sigma3 = [0.5 0; 
         0 0.5];
data3 = mvnrnd(mu3,sigma3,300);
%% 
figure(1),plot(data1(:,1),data1(:,2),'r+');hold on;
plot(data2(:,1),data2(:,2),'b*');hold on;
plot(data3(:,1),data3(:,2),'m^');hold on;
%%
Label1 = ones(length(data1),1);
Label2 = ones(length(data1),1)+1;
Label3 = ones(length(data1),1)+2;
Data = [data1;data2;data3];
Label = [Label1;Label2;Label3];
[xmin, ymin] = min(Data,[],1);
[xmax, ymax] = max(Data,[],1);
Data = [Data,ones(size(Data,1),1)];
%% Train Stage
[N,M]=size(Data);
% A = randn(M,3);
p = 1;
method = 'HK'; % HK Perceptron
MaxEpoch = 1000;%最大迭代次数
lr = 0.5;
C = max(Label); % class number
A = zeros(C,size(Data,2));
X = 1.2*xmin:0.1:1.2*xmax;
for i = 1:C
    Data_P = Data(Label==i,:);
    Data_N = Data(Label~=i,:);
    Data_N = -Data_N;%规范化
    Y = [Data_P;Data_N];
    if strcmp(method,'HK')
        tmp =  HKLearn(Y,lr); % HK
    else
        tmp =  perceptionLearn(Y,lr,MaxEpoch); % Perceptron
    end
    
    A(i,:) = tmp;
    %% Draw
    draw(tmp,X,(ymin:0.1:ymax));
end
if strcmp(method,'HK')
    title('HK for Multi-class classification-Case 1');
else
    title('Perceptron for Multi-class classification-Case 1');
end
%% Test Stage
test_pt=[0 3; 6 7; 5 -5;2.5 2.3;-2 -8;10 -1;3 8];
test_pt = [test_pt,ones(size(test_pt,1),1)];
Pred = A*test_pt';
Pred(Pred>0) = 1;
Pred(Pred<=0) = 0;
tmp = sum(Pred,1);
for i = 1:size(test_pt,1)
    scatter(test_pt(i,1),test_pt(i,2));
    if tmp(i)==1
        ind = find(Pred(:,i)==1);
        fprintf('Sample[ %f, %f] belongs to class %d.\n',test_pt(i,1),test_pt(i,2),ind);
    else
        fprintf('Sample[ %f, %f] is uncertain to be classified.\n',test_pt(i,1),test_pt(i,2));
    end
end

