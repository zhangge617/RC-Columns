clear
tic
for jjj=1


%��ȡ����
load dataweiyi166 input output
k=[1:166];
%k=1:185;
[m,n]=sort(k); 
%�ҳ�ѵ�����ݺ�Ԥ������
input_train=input(n(1:166),:)';
output_train=output(n(1:166));
input_test=input(n(1:166),:)';
output_test=output(n(1:166));


%�ڵ����
inputnum=8;
hiddennum=11;
outputnum=1;



%ѡ����������������ݹ�һ��
[inputn,inputps]=mapminmax(input_train,-1,1);
[outputn,outputps]=mapminmax(output_train,-1,1);

%��������
%net=newff(inputn,outputn,hiddennum,{'logsig', 'tansig'}, 'trainlm', 'learngd');
net=newff(inputn,outputn,hiddennum,{'tansig', 'tansig'}, 'trainlm', 'learngd');
% ������ʼ��
%����Ⱥ�㷨�е���������
c1 = 1.49445;
c2 = 1.49445;

maxgen=100;   % ��������  
sizepop=30;   %��Ⱥ��ģ

Vmax=1;
Vmin=-1;
popmax=5;
popmin=-5;

for i=1:sizepop
    pop(i,:)=5*rands(1,inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);
    V(i,:)=rands(1,inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);
    fitness(i)=fun(pop(i,:),inputnum,hiddennum,outputnum,net,inputn,outputn);
end


% ���弫ֵ��Ⱥ�弫ֵ
[bestfitness bestindex]=min(fitness);
zbest=pop(bestindex,:);   %ȫ�����
gbest=pop;    %�������
fitnessgbest=fitness;   %���������Ӧ��ֵ
fitnesszbest=bestfitness;   %ȫ�������Ӧ��ֵ

%% ����Ѱ��
for i=1:maxgen
    i;
    
    for j=1:sizepop
        
        %�ٶȸ���
        V(j,:) = V(j,:) + c1*rand*(gbest(j,:) - pop(j,:)) + c2*rand*(zbest - pop(j,:));
        V(j,find(V(j,:)>Vmax))=Vmax;
        V(j,find(V(j,:)<Vmin))=Vmin;
        
        %��Ⱥ����
        pop(j,:)=pop(j,:)+0.2*V(j,:);
        pop(j,find(pop(j,:)>popmax))=popmax;
        pop(j,find(pop(j,:)<popmin))=popmin;
        
        %����Ӧ����
        pos=unidrnd(inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);
        if rand>0.95
            pop(j,pos)=5*rands(1,1);
        end
      
        %��Ӧ��ֵ
        fitness(j)=fun(pop(j,:),inputnum,hiddennum,outputnum,net,inputn,outputn);
    end
    
    for j=1:sizepop
    %�������Ÿ���
    if fitness(j) < fitnessgbest(j)
        gbest(j,:) = pop(j,:);
        fitnessgbest(j) = fitness(j);
    end
    
    %Ⱥ�����Ÿ��� 
    if fitness(j) < fitnesszbest
        zbest = pop(j,:);
        fitnesszbest = fitness(j);
    end
    
    end
    
    yy(i)=fitnesszbest;    
        
end

%% �������
%plot(yy)
%title(['��Ӧ������  ' '��ֹ������' num2str(maxgen)]);
%xlabel('��������');ylabel('��Ӧ��');

x=zbest;
%% �����ų�ʼ��ֵȨֵ��������Ԥ��
% %���Ŵ��㷨�Ż���BP�������ֵԤ��

w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=B2;

%% BP����ѵ��
%�����������
net.trainParam.epochs=1000;
net.trainParam.lr=0.1;
net.trainParam.goal=1e-5;
net.trainParam.max_fail=50;
%����ѵ��
[net,per2]=train(net,inputn,outputn);
an=sim(net,inputn);

error=sum(abs(an-outputn));
%% BP����Ԥ��
%���ݹ�һ��
%inputn_test=mapminmax('apply',input_test,inputps);
%an=sim(net,inputn_test);
%test_simu=mapminmax('reverse',an,outputps);
%error=test_simu-output_test;
% outputn_test=mapminmax('apply',output_test,outputps);
%errors=an-outputn_test;
%[~,len]=size(output_test);
%mse3=error*error'/len;
%errorsum=sum(abs(error));
%mse1(jjj)=sum(errors*errors'/len);
%mse2=mse(net,an,outputn_test);
%plotregression(BPoutput,output_test);
%a=test_simu./output_test;
%b=mean(a);
%t(jjj)=std(a)/mean(a); %����ϵ��
%w=t/sum(t);
%tempdata=(output_test-test_simu).^2;
 %   tempdata2=(output_test-mean(output_test)).^2;
  %  r2=1 - ( sum(tempdata)/sum(tempdata2) );
%figure(jjj)
%plot(test_simu,':og')
%hold on
%plot(output_test,'-*');
%legend('Ԥ�����','�������')
%title('BP����Ԥ�����','fontsize',12)
%ylabel('�������','fontsize',12)
%xlabel('����','fontsize',12)

eval(['save(''psot',num2str(jjj),'''',',''net''',',''per2''',',''yy'')']);
clear
end
toc