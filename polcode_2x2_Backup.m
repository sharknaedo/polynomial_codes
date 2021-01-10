% Implementing polynomial codes to compute
% (A^T)B. Using 2 sources, N workers, and 1 master node.
% The stages are: 
% 1. Construct the polynomial code and `send` evaluations to each worker. 
% 2. Each worker computes the necessary products
% 3. Master reconstructs from worker computations. 
clc; 
clear all; 
p = 11; 
A = randi(2,2); A_size = size(A);
B = randi(2,2); B_size = size(B);

% If matrices cannot be multiplied, return
if (A_size(2) ~= B_size(1) ) 
    return 
end
% If there are not enough workers, return 
num_workers = 5; 
if (A_size(2) > num_workers)
    return
end

% Constructing the shares of each worker
% using a polynomial code of the columns.
for i=1:num_workers
    f_A(:,i) = mod(A(:,1) + i*A(:,2),p); 
    f_B(:,i) = mod(B(:,1) + (i^2)*B(:,2),p);
end

% now, each worker computes the local product
for i=1:num_workers
   C(i) = mod( f_A(:,i)'*f_B(:,i), p); 
end

% The workers who actually return a value to the
% master node are considered. 
considered_workers = [1,2,4,5]
C_considered = C(considered_workers)
disp (C_considered)

% Getting master node ready for reconstruction.
reshaped_worker_product = reshape (C_considered, [], 1)
m = fliplr ( vander(considered_workers) ); 

d = floor(mod(det(m),11));
[G,C] = gcd(d,p);
dinv = mod (C, p)

% Master node reconstruction 
ATB = mod( det(m)*inv(m)*reshaped_worker_product, p);
ATB =  mod(dinv * reshape (ATB, 2,2), 11)

% Printing for comparision
A'*B

