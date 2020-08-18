function [] = readMat()

    fid = fopen('MatVecDivStokes_sh16.txt','rt');
    C = textscan(fid,'%f');
    fclose(fid);
    %celldisp(C);
    A = cell2mat(C);
    DivStokes = reshape(A, 544, 3*544);
    rank(DivStokes)
    [U, S, V] = svd(DivStokes);
    size(diag(S));
    %y = gmres(B, rand(144, 1), 100, 1e-5, 300 ); 
    
       fid = fopen('MatVecStokes_sh16.txt','rt');
    C = textscan(fid,'%f');
    fclose(fid);
    %celldisp(C);
    A = cell2mat(C);
    Stokes = reshape(A, 3*544, 3*544);
    rank(Stokes)
    [U, S, V] = svd(Stokes);
    size(diag(S));
    %x = 1:544;
    %scatter(x, diag(S));
    
        fid = fopen('MatVecSigmaPure7_sh16.txt','rt');
    C = textscan(fid,'%f');
    fclose(fid);
    %celldisp(C);
    A = cell2mat(C);
    DivStokesFs = reshape(A, 544, 544);
    rank(DivStokesFs)
    [U, S, V] = svd(DivStokesFs);
    size(diag(S));
    %x = 1:544;
    %scatter(x, diag(S));
    
    
    fid = fopen('test_vesicle_sh16.txt','rt');
    ves = textscan(fid,'%f');
    ves = cell2mat(ves);
    ves_mat = reshape(ves, 544, 3);
    scatter3(ves_mat(:, 1), ves_mat(:, 2),  ves_mat(:, 3) );
    save('fullOp.mat', 'DivStokesFs');
    
    
 
end