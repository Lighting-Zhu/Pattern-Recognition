function draw(a,x1,range)
if abs(a(2))<1e-7
    x1=-a(3)/a(1);
    x2=range;
    x1=ones(size(x2))*x1;
else
    x2=(a(1)*x1+a(3))/(-a(2));
end
plot(x1,x2,'k-.');hold on;
end

