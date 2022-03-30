% ��ջ���
clc
clear
% 
tic
%% ����ṹ����
%��ȡ����
%��ȡ����
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
dim=inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum;
maxgen=100 ;   % ��������  
sizepop=30;   %��Ⱥ��ģ

popmax=5;
popmin=-5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P_percent = 0.2;    % The population size of producers accounts for "P_percent" percent of the total population size       

pNum = round( sizepop *  P_percent );    % The population size of the producers   


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:sizepop
    pop(i,:)=5*rands(1,inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);
%     V(i,:)=rands(1,21);
     fitness(i)=fun(pop(i,:),inputnum,hiddennum,outputnum,net,inputn,outputn); 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pFit = fitness;                      
[ fMin, bestI ] = min( fitness );      % fMin denotes the global optimum fitness value
bestX = pop( bestI, : );             % bestX denotes the global optimum position corresponding to fMin


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%��ȸ�����㷨 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for t = 1 : maxgen 
 [ ans, sortIndex ] = sort( pFit );% Sort.
     
  [fmax,B]=max( pFit );
   worse= pop(B,:);  
         
   r2=rand(1);
if(r2<0.8)
 
    for i = 1 : pNum                                                   % Equation (3)
         r1=rand(1);
        pop( sortIndex( i ), : ) = pop( sortIndex( i ), : )*exp(-(i)/(r1*maxgen));
       
        fitness(sortIndex( i ))=fun(pop(sortIndex( i ),:),inputnum,hiddennum,outputnum,net,inputn,outputn); 
    end
  else
  for i = 1 : pNum   
          
  pop( sortIndex( i ), : ) = pop( sortIndex( i ), : )+randn(1)*ones(1,dim);
  
  fitness(sortIndex( i ))=fun(pop(sortIndex( i ),:),inputnum,hiddennum,outputnum,net,inputn,outputn); 
       
  end
      
end
[ fMMin, bestII ] = min( fitness );      
  bestXX = pop( bestII, : );  
for i = ( pNum + 1 ) : sizepop                     % Equation (4)
     
         A=floor(rand(1,dim)*2)*2-1;
         
          if( i>(sizepop/2))
           pop( sortIndex(i ), : )=randn(1)*exp((worse-pop( sortIndex( i ), : ))/(i)^2);
          else
          pop( sortIndex( i ), : )=bestXX+(abs(( pop( sortIndex( i ), : )-bestXX)))*(A'*(A*A')^(-1))*ones(1,dim);  

         end  
        
        fitness(sortIndex( i ))=fun(pop(sortIndex( i ),:),inputnum,hiddennum,outputnum,net,inputn,outputn);
        
end
   c=randperm(numel(sortIndex));
   b=sortIndex(c(1:3));
    for j =  1  : length(b)      % Equation (5)

    if( pFit( sortIndex( b(j) ) )>(fMin) )

        pop( sortIndex( b(j) ), : )=bestX+(randn(1,dim)).*(abs(( pop( sortIndex( b(j) ), : ) -bestX)));

        else

        pop( sortIndex( b(j) ), : ) =pop( sortIndex( b(j) ), : )+(2*rand(1)-1)*(abs(pop( sortIndex( b(j) ), : )-worse))/ ( pFit( sortIndex( b(j) ) )-fmax+1e-50);

    end
       
       
       fitness(sortIndex( i ))=fun(pop(sortIndex( i ),:),inputnum,hiddennum,outputnum,net,inputn,outputn);
    end
    
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   for i = 1 : sizepop 
        if ( fitness( i ) < pFit( i ) )
            pFit( i ) = fitness( i );
             pop(i,:) = pop(i,:);
        end
        
        if( pFit( i ) < fMin )
           fMin= pFit( i );
            bestX =pop( i, : );
         
            
        end
    end
 
 

    yy(t)=fMin;    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ����Ѱ��
x=bestX;
%% �������
plot(yy)
title(['��Ӧ������  ' '��ֹ������' num2str(maxgen)]);
xlabel('��������');ylabel('��Ӧ��');
%% �����ų�ʼ��ֵȨֵ��������Ԥ��
% %����ȸ�����㷨�Ż���BP�������ֵԤ��
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=B2;

%% ѵ��
%�����������
net.trainParam.epochs=100;
net.trainParam.lr=0.1;
net.trainParam.goal=0.00001;

%����ѵ��
[net,tr]=train(net,inputn,outputn);

%%Ԥ��
%���ݹ�һ��
inputn_test=mapminmax('apply',input_test,inputps);
an=sim(net,inputn_test);
test_simu=mapminmax('reverse',an,outputps);
error=test_simu-output_test;
figure(2)
plot(error)
title('����Ԥ�����','fontsize',12);
xlabel('�������','fontsize',12);ylabel('���ٷ�ֵ','fontsize',12);

jjj=11;

per2=tr;
eval(['save(''ssaweiyit',num2str(jjj),'''',',''net''',',''per2''',',''yy'')']);
toc