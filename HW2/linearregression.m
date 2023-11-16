%% decoding
load('indy_20160411_02/bined_spk.mat');
load('indy_20160411_02/trial_pos.mat');
load('indy_20160411_02/trial_velocity.mat');
load('indy_20160411_02/trial_acceleration.mat');
CC = zeros(10,3); 
MSE = zeros(10,3);
R2 = zeros(10,3);
prediction = cell(10,1); 
bined_spk = [bined_spk;ones(1,size(bined_spk,2))];
% 执行10折交叉验证  
cv = cvpartition(size(bined_spk,2),'KFold',10);  
for i = 1:cv.NumTestSets  
    % 训练数据和测试数据
    disp(['Fold :',num2str(i)])

    train_data = bined_spk(:,cv.training(i));
    train_outputs = trial_acceleration(cv.training(i),:)';

    test_data = bined_spk(:,cv.test(i));
    test_outputs = trial_acceleration(cv.test(i),:)';

    [b1,bint1,r1,rint1,stats1] = regress(train_outputs(1,:)',train_data');
    [b2,bint2,r2,rint2,stats2] = regress(train_outputs(2,:)',train_data');
    [b3,bint3,r3,rint3,stats3] = regress(train_outputs(2,:)',train_data');

    prediction{i} = [test_data'*b1 test_data'*b2 test_data'*b3;]';
    x_cc=corrcoef(test_data(1,:),prediction{i}(1,:));
    y_cc=corrcoef(test_data(2,:),prediction{i}(2,:));
    z_cc=corrcoef(test_data(3,:),prediction{i}(3,:));
    if all(~isnan(x_cc(:))) && size(x_cc,2) > 1
       CC(i,:) = [x_cc(1,2) y_cc(1,2) z_cc(1,2)];
    else
       CC(i,:) = 0;
    end
    MSE(i,:) = [mse(prediction{i}(1,:)-test_data(1,:)) mse(prediction{i}(2,:)-test_data(2,:)) mse(prediction{i}(3,:)-test_data(3,:))];
    R2(i,:)=[1-(sum((prediction{i}(1,:)-test_data(1,:)).^2)/sum((test_data(1,:)-mean(test_data(1,:))).^2)) 1-(sum((prediction{i}(2,:)-test_data(2,:)).^2)/sum((test_data(2,:)-mean(test_data(2,:))).^2)) 1-(sum((prediction{i}(3,:)-test_data(3,:)).^2)/sum((test_data(3,:)-mean(test_data(3,:))).^2))];
end
save('indy_20160411_02/线性回归/acceleration/CC.mat','CC');
save('indy_20160411_02/线性回归/acceleration/MSE.mat','MSE');
save('indy_20160411_02/线性回归/acceleration/R2.mat','R2');