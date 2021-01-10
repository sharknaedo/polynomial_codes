showOn = false; 
clc
%making matrix A
a1 = [[1,2];[2,1]]; 
a2 = a1 + fliplr(eye(2)); 
a3 = a2 + fliplr(eye(2)); 
a4 = a3 + fliplr(eye(2)); 
A = [ [a1,a2];[a3,a4] ];

% matrix B
B = eye(4); 

% other parameters :
N = 4; t = 1; m = 2; z = 4; p = 11;

% Stage 1: polynomial codes of A and B
A = @(x) [a1;a3] + [a2;a4]*x + [[3,2];[1,4];[4,2];[2,3]]*x^2;
B = @(x) B(:,1:2) + B(:,3:4)*x + [[5,6];[1,3];[2,4];[1,2]]*x^2;

% each worker `n` receives A(n) and B(n)
for i=1:4
    share_A(:,:,i) = mod( A(i), p);
    share_B(:,:,i) = mod( B(i), p);
end
if (showOn == true)
    fprintf("Shares of A:\n"); 
    disp (share_A)
    fprintf("\nShares of B:\n"); 
    disp (share_B)
end

% Stage 2(A): subsharing A
A_1 = @(x) share_A(:,:,1) + [[9,7];[10,2];[2,4];[0,7]]*x + ...
            [[0,0];[0,6];[2,9];[11,2];]*(x^2); 
A_2 = @(x) share_A(:,:,2) + [[7,6];[7,3];[3,10];[4,3]]*x + ...
            [[3,4];[2,11];[3,5];[5,3];]*(x^2);
A_3 = @(x) share_A(:,:,3) + [[10,3];[11,5];[5,7];[2,3]]*x + ...
            [[7,4];[8,4];[3,5];[2,6];]*(x^2);
A_4 = @(x) share_A(:,:,4) + [[1,0];[3,9];[9,6];[1,7]]*x + ...
            [[3,6];[6,3];[0,6];[7,7];]*(x^2);

subshare_poly_A = {A_1, A_2, A_3, A_4};
for n=1:4
    for ndash = 1:4
        subshares_A(:,:,n,ndash) = mod ( subshare_poly_A{n}(ndash), p);
    end
end
if (showOn == true)
    fprintf("Subshares of A:\n"); 
    disp (subshares_A)
end

%Stage 2(B): subsharing B
B_10 = @(x) share_B(1:2,:,1) + [[3,7];[5,0]]*x + [[5,4];[0,8]]*(x^2);
B_11 = @(x) share_B(3:4,:,1) + [[8,8];[6,8]]*x;

B_20 = @(x) share_B(1:2,:,2) + [[1,10];[7,8]]*x + [[3,6];[5,0]]*(x^2);
B_21 = @(x) share_B(3:4,:,2) + [[2,8];[10,5]]*x;

B_30 = @(x) share_B(1:2,:,3) + [[7,5];[3,7]]*x + [[3,7];[4,3]]*(x^2);
B_31 = @(x) share_B(3:4,:,3) + [[10,9];[0,4]]*x;

B_40 = @(x) share_B(1:2,:,4) + [[9,7];[3,1]]*x + [[5,2];[4,2]]*(x^2);
B_41 = @(x) share_B(3:4,:,4) + [[7,10];[2,10]]*x;

subshare_poly_B = { {B_10, B_11}, {B_20,B_21}, {B_30, B_31}, {B_40,B_41} };
for n = 1:4
    for j = 1:2
        for ndash = 1:4
            intermediate_subshares_B(:,:,n,j,ndash) = mod(subshare_poly_B{n}{j}(ndash),p);
        end
    end
end
if (showOn == true)
    fprintf("Received intermediate subshares of B:\n"); 
    disp (intermediate_subshares_B)
end

% Each worker calculates his subshare of B i.e. B_n(ndash)
% Verified.
subshares_B = zeros(m,m,N,N);
for n=1:4
    for ndash = 1:4
        subshares_B(:,:,n,ndash) = mod(... 
            intermediate_subshares_B(:,:,n,1,ndash) + ndash * intermediate_subshares_B(:,:,n,2,ndash), ...
            p);
    end
end
if (showOn == true)
    fprintf("Recombined subshares of B_n(x), locally computed:\n"); 
    disp(subshares_B)
end

% Stage 2(B)(hidden): Each party `n` also forms B_n(x).
% This is the polynomial that from which the shares in  
% recom_subshares_B are created. Importantly, those shares 
% are independently and locally calculated. Here, we need 
% B_n(x) for Stage i.e. to calculate the O polynomials. 
B_1 = @(x) B_10(x) + B_11(x)*x; 
B_2 = @(x) B_20(x) + B_21(x)*x;
B_3 = @(x) B_30(x) + B_31(x)*x;
B_4 = @(x) B_40(x) + B_41(x)*x;

if (showOn == true)
    syms x; assume(x, 'real')
    fprintf ("B_n(x) for n = 1,2,3,4: \n \n"); 
    disp(expand(B_1(x))); disp( expand(B_2(x)) ); 
    disp(expand(B_4(x))); disp(expand(B_4(x))); 
end

% Stage 3: Computing on subshares 
syms x; assume(x, 'real');
AB_1 = expand(A_1(x)*B_1(x)');
AB_2 = expand(A_2(x)*B_2(x)');
AB_3 = expand(A_3(x)*B_3(x)');
AB_4 = expand(A_4(x)*B_4(x)');

poly_AB = {AB_1, AB_2, AB_3, AB_4}; 

% taking each coefficients modulo `p` 
for n=1:4 % for each AB_n
    for i = 1:4 % each row of the code
        for j = 1:2 % each column of code
            poly_AB{n}(i,j) = poly2sym( mod(coeffs(poly_AB{n}(i,j)), p) ); 
        end
    end
    % display if appropriate setting is true 
    if(showOn == true) 
        fprintf ("Subsharing polynomial of AB' held by %d\n", n); 
        disp ( poly_AB{n} ); 
    end
end
% Constructing the local shares of AB' at each worker.
subshares_AB = zeros(4,2,4,4);
for n=1:4
    for ndash = 1:4
%           x=ndash;  % evaluate the coming polynomial at x=n'
          subshares_AB(:,:,n,ndash) = mod( ... 
              subshares_A(:,:,n,ndash) * subshares_B(:,:,n,ndash)',p);
    end
end

% Constructing the O polynomials. Hardcoded for now. 
% Each O^(n)_j (x) = 
O_11 = @(x) [[9,7];[10,2];[2,4];[0,7]] + [[0,0];[6,8];[3,6];[8,9]]*x; 
O_10 = @(x) [[0,0];[0,6];[2,9];[0,2]] + [[8,4];[10,4];[4,6];[9,3]]*x; 

O_21 = @(x) [[8,8];[9,2];[9,8];[5,1]] + [[6,4];[4,9];[3,10];[8,8]]*x; 
O_20 = @(x) [[4,8];[1,4];[2,0];[10,1]] + [[6,1];[0,0];[8,2];[2,5]]*x;

O_31 = @(x) [[5,3];[5,6];[9,5];[9,8]] + [[0,2];[0,2];[10,10];[10,2]]*x; 
O_30 = @(x) [[8,8];[9,2];[4,2];[8,6]] + [[1,2];[1,9];[2,3];[0,9]]*x;

O_41 = @(x) [[0,9];[4,3];[7,6];[3,8]] + [[8,2];[1,0];[9,9];[10,6]]*x; 
O_40 = @(x) [[10,2];[0,3];[7,10];[2,3]] + [[1,10];[4,1];[8,9];[4,10]]*x;

poly_O = {{O_11,O_10},{O_21,O_20},{O_31,O_30},{O_41,O_40}};
subshare_O = zeros(4,2,4,2,4); 
for n = 1:4
    for j = 1:2
        for ndash = 1:4
            subshare_O(:,:,n,j,ndash) = mod ( poly_O{n}{j}(ndash), p ); 
        end
    end
end
if (showOn == true)
    fprintf ("Each worker now receives the O shares: \n\n"); 
    disp ( subshare_O ); 
end

% Each worker can now calculate their subshare of C, 
% using the shares they receieved of A_n, B_n and the Os.
syms x; assume(x,'real'); 
for n=1:4
    for ndash=1:4
        x = ndash;
        subshares_C(:,:,n,ndash) = mod(subs(poly_AB{n}) ... % subshares_AB(:,:,n,ndash) ... 
            - (ndash^2) * ( subshare_O(:,:,n,2,ndash) + ... 
                    ndash*subshare_O(:,:,n,1,ndash) ) ...
            ,p);
    end
end
if (showOn == true)
    fprintf ("Each worker's subshares of C:\n"); 
    subshares_C
end

% Stage(3)(Hidden)
% Now we can form the polynomial forms of the 
% subshares of C, given by the formula
% A_n(x)B_n(x) - x^m (O^n_0 + xO^n_1)
% This is for this case only. Theoretically, 
% each worker should have subshare_poly_C(:,:,n)(ndash)
syms x; assume(x,'real'); 
for n=1:4
    subshare_poly_C(:,:,n) = expand ( poly_AB{n} - ... 
        x^2 * ( poly_O{n}{2}(x) + x*poly_O{n}{1}(x) ) );
end
if ( showOn == true )
    fprintf ("Polynomial form of the shares of C at each worker:\n"); 
    subshare_poly_C
end
    
% for n=1:4
%     for ndash = 1:4
%         x = ndash;
%         eval_C(:,:,n,ndash) = mod( subs(subshare_poly_C(:,:,n)), p ); 
%     end
% end
% syms x

% Stage 4: The recombination at the master node. 
considered_workers = 1:3; 
m = fliplr(vander(considered_workers));
minv = floor(mod(inv(m) * det(m),p));

lambda = minv (1,:)
share_C = zeros(4,2,4);
for n=1:3
    for ndash = 1:3
        share_C(:,:,n) = mod( share_C(:,:,n) + ... 
            lambda(ndash)*subshares_C(:,:,n,ndash), p);
    end
end
% NOTES: try: syms y; assume(y,'real'); expand(A_1(y)*B_1(y)')