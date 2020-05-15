function w=  perceptionLearn(Y,lr,maxEpoch)
rand('seed',300);
a_cur = rand(1,size(Y,2));
iter = 0;
while(iter<maxEpoch)
    c = 0;
    for i = 1:size(Y,1)
        tmp = a_cur(end,:)*Y(i,:)';
        if tmp>0
            c = c + 1;
        else
            a_t = a_cur(end,:) + lr*Y(i,:);
            a_cur = [a_cur;a_t];
        end
    end
    iter = iter + 1;
    if c == size(Y,1)
        break;
    end
end
w = a_cur(end,:);
end

