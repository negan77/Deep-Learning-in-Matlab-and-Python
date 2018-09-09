function [W1, W2] = BackpropXOR(W1, W2, X, D)
% 以神经网络的权重和训练数据作为输入，返回调整后的权重
% 其中W1和W2为相应层的权重矩阵；X和D分别是训练数据的输入和标准输入
  alpha = 0.9;
  
  N = 4;  
  for k = 1:N
    x = X(k, :)';
    d = D(k);
    
    v1 = W1*x;
    y1 = Sigmoid(v1);    
    v  = W2*y1;
    y  = Sigmoid(v);
    
    e     = d - y;
    delta = y.*(1-y).*e;

    e1     = W2'*delta;    % 反向传播
    delta1 = y1.*(1-y1).*e1; 
    
    dW1 = alpha*delta1*x';
    W1  = W1 + dW1;
    
    dW2 = alpha*delta*y1';    
    W2  = W2 + dW2;
  end
end