% Implementing polynomial codes to compute
% (A^T)B. Using 2 sources, N workers, and 1 master node.
% The stages are: 
% 1. Construct the polynomial code and `send` evaluations to each worker. 
% 2. Each worker computes the necessary products
% 3. Master reconstructs from worker computations. 
clc; 
p = 17; N = 6; num_columns = 3; 
m = N/num_columns; n = N/num_columns; 

% A = randi(N,N); A_size = size(A);
% B = randi(N,N); B_size = size(B);

A = randi(N,N);  B = eye(N); 
A_size = size(A); B_size = size(B);

% If matrices cannot be multiplied, return
    if (A_size(2) ~= B_size(1) ) 
        return 
    end
% If there are not enough workers, return 
    num_worker = m*n
    
% partitioning each matrix .
index = 1;
for i=1:m:N
    A_part(:,:,index) = A(:, i:i+m-1);
    B_part(:,:,index) = B(:, i:i+m-1);
    index = index + 1;
end

% Constructing each share for each worker 
f_A = zeros(N,m,num_worker);
f_B = zeros(N,m,num_worker);
for i=1:num_worker
    for j=1:N
        f_A(:,:,i) = f_A(:,:,i) + A_part(:,:,i) * i^(j); 
        f_B(:,:,i) = f_B(:,:,i) + B_part(:,:,i) * i^(m*j);
    end
end
 f_A = mod(f_A, p);
 f_B = mod(f_B, p);
disp(f_A); disp (f_B)
