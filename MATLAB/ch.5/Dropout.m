function ym = Dropout(y, ratio)
% y是输出向量；ratio是输出向量Dropout的比例
  [m, n] = size(y);  
  ym     = zeros(m, n);
  
  num     = round(m*n*(1-ratio));
  idx     = randperm(m*n, num); % ym元素的索引
  ym(idx) = 1 / (1-ratio);
end