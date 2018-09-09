function ym = Dropout(y, ratio)
% y�����������ratio���������Dropout�ı���
  [m, n] = size(y);  
  ym     = zeros(m, n);
  
  num     = round(m*n*(1-ratio));
  idx     = randperm(m*n, num); % ymԪ�ص�����
  ym(idx) = 1 / (1-ratio);
end