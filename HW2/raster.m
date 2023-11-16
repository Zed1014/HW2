load('indy_20170124_01.mat');

% 初始化一个空的向量来存储变化的索引  
changed_indices = [];  
changes(:,1)= diff(target_pos(:, 1));  
changes(:,2)= diff(target_pos(:, 2)); 
    
index = find(changes ~= 0)+1; 
index = index(1:440);
changes1 = diff(index);  
indices = find(changes1==1);
index(indices) = [];
trial_target_pos =  target_pos(index,:);
for i=2:size(trial_target_pos,1)
    trial_angles(i-1) = findAngle(trial_target_pos(i-1,1), trial_target_pos(i-1,2),trial_target_pos(i,1), trial_target_pos(i,2)); 
end

save('trial_angles.mat','trial_angles');
save('trial_target_pos.mat','trial_target_pos');

function [angle] = findAngle(x1, y1, x2, y2)
  dy = (y2-y1);
  dx = (x2-x1);
  if(x2>=x1&y2>y1)
    angle =  atand(dy/dx);
  elseif(x2<x1&y2>=y1) 
    angle = 180 + atand(dy/dx); 
  elseif(x2<=x1&y2<y1) 
    angle =  atand(dy/dx)+180;
  elseif(x2>x1&y2<=y1) 
    angle =  360+atand(dy/dx);
  else
    angle =  0; 
  end
end
