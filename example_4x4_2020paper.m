%% Example Code for Secure, Distributed Matrix Multiplication.
% This code goes through the calculations of multiplying 2 $ 4 \times 4 $
% matrices, each split into submatrices of size $4 \times 2$ each. All
% calculations are in the finite field $F_{p}$.
% There are 4 workers to distribute among, and we specify $t = 1$. 
% NOTE: This version is based on Example 5, from (Najarkolaei, 2020)!

% Formatting options
rng('default'); 
showOn = false; 
format compact;

%making matrix A
a1 = [[1,2];[2,1]]; 
a2 = a1 + fliplr(eye(2)); 
a3 = a2 + fliplr(eye(2)); 
a4 = a3 + fliplr(eye(2)); 
A = [ [a1,a2];[a3,a4] ];

% matrix B
B = eye(4);

% other parameters :
N = 13;         % Number of workers needed. 
p = 13;         % All computations are modulo this.
m = 2;          % splitting each matrix into 2 submatrices, columnwise.

if (showOn == true)
    disp(mat_A);
end
 
%% Stage 1: polynomial codes of A and B
% We construct the direct polynomial code to share A and B to the worker
% nodes. Here, the $deg(A) = deg(B) = 6$. Then, each worker receives a 
% sharing of `polA` and `polB` respectively. 
polA = @(x) A(:,1:2) + A(:,3:4)*x +   randi(p,4,2)*x^4 + randi(p,4,2)*x^5 + randi(p,4,2)*x^6;
polB = @(x) B(:,1:2) + B(:,3:4)*x + randi(p,4,2)*x^4 + randi(p,4,2)*x^5 + randi(p,4,2)*x^6;

% each worker `n` receives A(n) and B(n)
for n=1:N
    share_A(:,:,n) = mod( polA(n), p);
    share_B(:,:,n) = mod( polB(n), p);
end

if (showOn == true)
    fprintf("Shares of A:\n"); 
    disp (share_A)
    fprintf("\nShares of B:\n"); 
    disp (share_B)
end

%% Stage 2: Computation phase
% Here, each worker multiplies the shares of A and B that they have
% received. Then, they Shamir-share their product shares to each other
% worker.

% Products first ... 
for n = 1:N
    share_AB(:,:,n) = mod( share_A(:,:,n)' * share_B(:,:,n), p); 
end

% Now Shamir-sharing the shares. For that, we need the vandermonde matrix
% of size `N`, where the n'th row will be used by the corresponding worker.

% First, we find the coefficients r_{ij}^{n} that we will need for each
% worker. 
considered_workers = 1:N;
vander_mat =  mod( vander(considered_workers), p); 
inv_vander = round( mod( inv(vander_mat)*det(vander_mat), p) );
dinv_temp = mod((1:p)*det(vander_mat), p);
dinv = find ( fix(dinv_temp) == 1);
inv_vander = mod(inv_vander*dinv, p); 

% Now, each worker can form their polynomial with which they can 
% share their product. 
for n = considered_workers
    pol_subshare_AB{n} = ... 
        @(x) [inv_vander(1,n)*share_AB(:,:,n); inv_vander(2,n)*share_AB(:,:,n) ] + ...
             [inv_vander(3,n)*share_AB(:,:,n); inv_vander(4,n)*share_AB(:,:,n) ].*x.^2 + ...
             randi(p,4,2).*x.^4 + randi(p,4,2).*x.^5 + randi(p,4,2).*x.^6; 
         
    % Each worker `n` now sends shares to every other worker `ndash`
    for ndash = 1:N
        subshare_AB(:,:,n,ndash) = mod( pol_subshare_AB{n}(ndash), p); 
    end
end

%% Stage 2.5: Local Worker Computation
% Each worker computes the sum of the `A^TB` subshares they received from
% every other worker.
for ndash = 1:N
    % Structure of the subshare matrix is: 
    %     (share_col_1, share_col_2, sender_worker, receiver_worker).
    % Here, we're summing over each share receieved by worker `ndash`.
    final_subshare_AB(:,:,ndash) = sum (subshare_AB(:,:,:,ndash), 3); 
end

%% Stage 3: Reconstruction
% Each worker sends their share of the final product, as computed in Stage
% 2.5, to the master node. Master node has to reconstruct the product from
% these shares. 