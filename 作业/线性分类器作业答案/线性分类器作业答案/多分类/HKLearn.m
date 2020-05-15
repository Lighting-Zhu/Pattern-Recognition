function a = HKLearn(Y,lr)
Eps = 1e-5;
MaxIter = ceil(20000/lr);%最大迭代次数
%%
rand('seed',800);
%%
b=rand(size(Y,1),1);
Y_flag=inv(Y'*Y)*Y';
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
a = a';
end

