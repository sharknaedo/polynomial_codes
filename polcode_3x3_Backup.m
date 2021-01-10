% Implementing polynomial codes to compute
% (A^T)B. Using 2 sources, N workers, and 1 master node.
% The stages are: 
% 1. Construct the polynomial code and `send` evaluations to each worker. 
% 2. Each worker computes the necessary products
% 3. Master reconstructs from worker computations. 
clc; 
clear all; 
p = 11; N = 3; %m = 3; n = 3; 
 A = randi(N,N); A_size = size(A);
 B = randi(N,N); B_size = size(B);

% If matrices cannot be multiplied, return
if (A_size(2) ~= B_size(1) ) 
    return 
end
% If there are not enough workers, return 
num_workers = N^2; 
if (A_size(2) > num_workers)
    return
end

% Constructing the shares of each worker
% using a polynomial code of the columns.
tempA = 0; tempB = 0;
for i=1:num_workers
    for j=1:N
        tempA = tempA + (i^(j-1))*A(:,j);
        tempB = tempB + i^(N*(j-1))*B(:,j);
    end
    f_A(:,i) = mod(tempA, p); tempA = 0; 
    f_B(:,i) = mod(tempB, p); tempB = 0; 
end

% now, each worker computes the local product
for i=1:num_workers
   C(i) = mod( f_A(:,i)'*f_B(:,i), p); 
end

% The workers who actually return a value to the
% master node are considered. 
considered_workers = 1:num_workers;
C_considered = C(considered_workers);

% Getting master node ready for reconstruction.
reshaped_worker_product = reshape (C_considered, [], 1);
m = mod(fliplr(vander(considered_workers)), p); 

d = round(mod(det(m),p));
[G,rem] = gcd(d,p);
dinv = mod (rem, p);

% Master node reconstruction 
ATB = mod( det(m)*inv(m)*reshaped_worker_product, p);
ATB =  mod(dinv * reshape (ATB, N,N), p)

% Printing for comparision
mod(A'*B, p)
norm ( ATB - ans )
