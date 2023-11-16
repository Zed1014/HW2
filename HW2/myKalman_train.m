% 
function kalman = myKalman_train(trainX, trainY)
    
    X1=trainY(:,1:end-1);
    X2=trainY(:,2:end);
    
   %%确定参数A，H，W，Q
    kalman.A=(X2*X1')*inv(X1*X1'); %状态转移矩阵
    kalman.H=trainX*trainY'*inv(trainY*trainY'); %测量转移矩阵
  
    %%利用A和H估算W和Q
    [~,n]=size(trainY);
    kalman.Q=(X2-kalman.A*X1)*(X2-kalman.A*X1)'/(n-1);%系统噪声
    kalman.R=(trainX-kalman.H*trainY)*(trainX-kalman.H*trainY)'/n;%观测噪声

end