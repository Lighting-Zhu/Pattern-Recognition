clc;
close all;
clear all;
%%
Y = [0 0 1;
    0 1 1;
    -1 0 -1;
    -1 -1 -1];
p = 1;% learning rate
a_cur = [0 0 0];
while(1)
    c = 0;
    for i = 1:size(Y,1)
        tmp = a_cur(end,:)*Y(i,:)';
        if tmp>0
            c = c + 1;
        else
            a_t = a_cur(end,:) + p*Y(i,:);
            a_cur = [a_cur;a_t];
        end
    end
    if c == size(Y,1)
        break;
    end
end
fprintf('The learned weight:\n');
a_cur