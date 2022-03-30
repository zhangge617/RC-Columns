clear
tic
for jjj=1


%读取数据
load dataweiyi166 input output
k=[1:166];
%k=1:185;
[m,n]=sort(k); 
%找出训练数据和预测数据
input_train=input(n(1:166),:)';
output_train=output(n(1:166));
input_test=input(n(1:166),:)';
output_test=output(n(1:166));


%节点个数
inputnum=8;
hiddennum=11;
outputnum=1;



%选连样本输入输出数据归一化
[inputn,inputps]=mapminmax(input_train,-1,1);
[outputn,outputps]=mapminmax(output_train,-1,1);

%构建网络
%net=newff(inputn,outputn,hiddennum,{'logsig', 'tansig'}, 'trainlm', 'learngd');
net=newff(inputn,outputn,hiddennum,{'tansig', 'tansig'}, 'trainlm', 'learngd');
% 参数初始化
%粒子群算法中的两个参数
c1 = 1.49445;
c2 = 1.49445;

maxgen=100;   % 进化次数  
sizepop=30;   %种群规模

Vmax=1;
Vmin=-1;
popmax=5;
popmin=-5;

for i=1:sizepop
    pop(i,:)=5*rands(1,inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);
    V(i,:)=rands(1,inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);
    fitness(i)=fun(pop(i,:),inputnum,hiddennum,outputnum,net,inputn,outputn);
end


% 个体极值和群体极值
[bestfitness bestindex]=min(fitness);
zbest=pop(bestindex,:);   %全局最佳
gbest=pop;    %个体最佳
fitnessgbest=fitness;   %个体最佳适应度值
fitnesszbest=bestfitness;   %全局最佳适应度值

%% 迭代寻优
for i=1:maxgen
    i;
    
    for j=1:sizepop
        
        %速度更新
        V(j,:) = V(j,:) + c1*rand*(gbest(j,:) - pop(j,:)) + c2*rand*(zbest - pop(j,:));
        V(j,find(V(j,:)>Vmax))=Vmax;
        V(j,find(V(j,:)<Vmin))=Vmin;
        
        %种群更新
        pop(j,:)=pop(j,:)+0.2*V(j,:);
        pop(j,find(pop(j,:)>popmax))=popmax;
        pop(j,find(pop(j,:)<popmin))=popmin;
        
        %自适应变异
        pos=unidrnd(inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);
        if rand>0.95
            pop(j,pos)=5*rands(1,1);
        end
      
        %适应度值
        fitness(j)=fun(pop(j,:),inputnum,hiddennum,outputnum,net,inputn,outputn);
    end
    
    for j=1:sizepop
    %个体最优更新
    if fitness(j) < fitnessgbest(j)
        gbest(j,:) = pop(j,:);
        fitnessgbest(j) = fitness(j);
    end
    
    %群体最优更新 
    if fitness(j) < fitnesszbest
        zbest = pop(j,:);
        fitnesszbest = fitness(j);
    end
    
    end
    
    yy(i)=fitnesszbest;    
        
end

%% 结果分析
%plot(yy)
%title(['适应度曲线  ' '终止代数＝' num2str(maxgen)]);
%xlabel('进化代数');ylabel('适应度');

x=zbest;
%% 把最优初始阀值权值赋予网络预测
% %用遗传算法优化的BP网络进行值预测

w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=B2;

%% BP网络训练
%网络进化参数
net.trainParam.epochs=1000;
net.trainParam.lr=0.1;
net.trainParam.goal=1e-5;
net.trainParam.max_fail=50;
%网络训练
[net,per2]=train(net,inputn,outputn);
an=sim(net,inputn);

error=sum(abs(an-outputn));
%% BP网络预测
%数据归一化
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
%t(jjj)=std(a)/mean(a); %变异系数
%w=t/sum(t);
%tempdata=(output_test-test_simu).^2;
 %   tempdata2=(output_test-mean(output_test)).^2;
  %  r2=1 - ( sum(tempdata)/sum(tempdata2) );
%figure(jjj)
%plot(test_simu,':og')
%hold on
%plot(output_test,'-*');
%legend('预测输出','期望输出')
%title('BP网络预测输出','fontsize',12)
%ylabel('函数输出','fontsize',12)
%xlabel('样本','fontsize',12)

eval(['save(''psot',num2str(jjj),'''',',''net''',',''per2''',',''yy'')']);
clear
end
toc