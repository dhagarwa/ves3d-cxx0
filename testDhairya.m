load fullOp.mat
load opMatrices.mat

n = size(DivStokesFs,1);

% Generate some data
sigma = rand(n,1);

% Generate RHS
divU = DivStokesFs * sigma;

% Solve
[x, flag, relres, iter, resvec] = gmres(DivStokesFs, divU, 100, 1e-7);
resvec