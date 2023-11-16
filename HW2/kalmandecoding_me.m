%% decoding
CC = zeros(10,3); 
MSE = zeros(10,3);
R2 = zeros(10,3);
prediction = cell(10,1); 
% 执行10折交叉验证  
cv = cvpartition(size(bined_spk,2),'KFold',10);  

for i = 1:cv.NumTestSets  
    % 训练数据和测试数据
    disp(['Fold :',num2str(i)])

    train_data = bined_spk(:,cv.training(i));
    train_outputs = trial_pos(cv.training(i),:)';

    test_data = bined_spk(:,cv.test(i));
    test_outputs = trial_pos(cv.test(i),:)';
 
    % Kalman 
    decoding_algrithm = 'Kalman';
    model_train = @myKalman_train;
    model_predict = @myKalman_predict;


    model = model_train(train_data,train_outputs);
    [CC(i,:), MSE(i,:), R2(i,:),prediction{i}] = model_predict(model, test_data, test_outputs);

    
end


