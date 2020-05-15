clc;
clear all;
close all;
load('data.mat');
%% parameter
lr=0.5;%学习率
MaxIter = ceil(20000/lr);%最大迭代次数
Eps = 1e-5;


% plot(w1_data(:,1),w1_data(:,2),'b*');
% hold on
% plot(w2_data(:,1),w2_data(:,2),'r*');
% hold on

rand('seed',2020);
list=randperm(100);
W1=[w2_data(list(1:10),:,:);w1_data(list(11:end),:,:)];
W2=[w1_data(list(1:10),:,:);w2_data(list(11:end),:,:)];
w1_data=W1;
w2_data=W2;
plot(W1(:,1),W1(:,2),'b*');
hold on
plot(W2(:,1),W2(:,2),'r*');
hold on


Y=[w1_data;-w2_data];%待分类数据，w1在决策平面的正侧
b=rand(size(Y,1),1);
Y_flag=inv(Y'*Y)*Y';
% Y_flag=pinv(Y);

N=length(b);
C=0;%迭代次数
while(C < MaxIter)
    a=Y_flag*b;
    e=Y*a-b;
    zeronum = sum(e<Eps & e>-Eps);
    nenum = sum(e<0);
    if  zeronum==N;%all is 0
        fprintf('the sample is linear to be classified! and The iteration number is %d\n',C);
        break;
    elseif nenum ==N
        fprintf('the sample is non-linear to be classified! and The iteration number is %d\n',C);
        break;
    end
    delta=lr*(e+abs(e));
    b = b + delta;
    C=C+1;
end
if C ==MaxIter
    if sum(e>-Eps)==N % all is larger than or equal to 0.
        fprintf('It has cost all iterartions(%d), and all elements are larger than or equal to 0. The sample is linear to be classified!\n',MaxIter);
    else if sum(e<=Eps) ==N% all is less than or equal to 0.
            fprintf('It has cost all iterartions(%d), and all elements are less than or equal to 0. The sample is non-linear to be classified!\n',MaxIter);
        else
            fprintf('It has cost all iterartions(%d), the sample is uncertain to be classified!\n',MaxIter);
        end
    end
end
fprintf('===============================================\n');
fprintf('The learned weight and bias as following:\n');
a
fprintf('===============================================\n');
x1=(-4:0.1:2);
if abs(a(2))<1e-7
    x1=-a(3)/a(1);
    x2=(-1:0.1:1);
    x1=ones(size(x2))*x1;
else
    x2=(a(1)*x1+a(3))/(-a(2));
end
plot(x1,x2,'LineWidth',1);