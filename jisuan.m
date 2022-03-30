clear
for i=10
load gatweiyi11;
a(i,1)=per2.best_perf;
a(i,2)=per2.best_vperf;
a(i,3)=per2.best_tperf;
a(i,5)=min(yy(:));
%plot(yy)
%hold on

load dataweiyi166 input output
k=[1:166];
%k=1:185;
[m,n]=sort(k); 
input_train=input(n(1:166),:)';
output_train=output(n(1:166));
input_test=input(n(1:166),:)';
output_test=output(n(1:166));
[inputn,inputps]=mapminmax(input_train,-1,1);
[outputn,outputps]=mapminmax(output_train,-1,1);
inputn_test=mapminmax('apply',input_test,inputps);
an=sim(net,inputn_test);
test_simu=mapminmax('reverse',an,outputps);
error=test_simu-output_test;
 outputn_test=mapminmax('apply',output_test,outputps);
errors=an-outputn_test;
[~,len]=size(output_test);
mse3=error*error'/len;
errorsum=sum(abs(error));
mse1(i)=sum(errors*errors'/len);
mse2=mse(net,an,outputn_test);

d=test_simu./output_test;
c=mean(d);
t(i)=std(d)/mean(d); %����ϵ��
w=t/sum(t);
tempdata=(output_test-test_simu).^2;
   tempdata2=(output_test-mean(output_test)).^2;
  r2(i)=1 - ( sum(tempdata)/sum(tempdata2) );
a(i,4)=sqrt(abs(r2(i)));
%plot(test_simu,':og')
%hold on
%plot(output_test,'-*');
%legend('Ԥ�����','�������')
%title('BP����Ԥ�����','fontsize',12)
%ylabel('�������','fontsize',12)
%xlabel('����','fontsize',12)
end

