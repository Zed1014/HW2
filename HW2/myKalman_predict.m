function [CC, MSE, R2, prediction] = myKalman_predict(kalman, testX, testY)
    
    %初始化估计矩阵
    m = size(testY,1);
    prediction=zeros(m,size(testX,3));
    prediction(:,1) = [0;0;0]; % 10*(rand(2,1)-0.5); % testY(:,1); % 
    %prediction(:,1) =testY(:,1);
    P = zeros(m);
    
    for j=2:size(testX,2)
        Xn=kalman.A*prediction(:,j-1);%一步预测   上一时刻的最优估计
        P_=kalman.A*P*kalman. A'+kalman.Q;%一步预测误差方差阵   当前先验估计误差的协方差
        %一步预测误差方差阵
        K=P_*kalman.H'*pinv(kalman.H*P_*kalman.H'+kalman.R);%滤波增益矩阵（权重）
        prediction(:,j)=Xn+K*(testX(:,j)-kalman.H*Xn);%状态误差方差阵估计   用这一时刻的先验估计、观测值、卡曼增益得到这一时刻的最优估计
        P=(eye(m)-K*kalman.H)*P_;   %用这一时刻的kalman增益和先验估计误差的协方差更新后验估计误差
    end 
     x_cc=corrcoef(testY(1,:),prediction(1,:));
     y_cc=corrcoef(testY(2,:),prediction(2,:));
     z_cc=corrcoef(testY(3,:),prediction(3,:));
     if all(~isnan(x_cc(:))) && size(x_cc,2) > 1
        CC = [x_cc(1,2) y_cc(1,2) z_cc(1,2)];
     else
         CC = 0;
     end
     MSE = [mse(prediction(1,:)-testY(1,:)) mse(prediction(2,:)-testY(2,:)) mse(prediction(3,:)-testY(3,:))];
     R2=[1-(sum((prediction(1,:)-testY(1,:)).^2)/sum((testY(1,:)-mean(testY(1,:))).^2)) 1-(sum((prediction(2,:)-testY(2,:)).^2)/sum((testY(2,:)-mean(testY(2,:))).^2)) 1-(sum((prediction(3,:)-testY(3,:)).^2)/sum((testY(3,:)-mean(testY(3,:))).^2))];
end