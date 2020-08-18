function [] = surfLap()

%---------------USING PARAMETRIC-------------------------------------
%     m = @(u,v) {0.2*cos(u), 0.1*sin(u).*sin(v), 0.01*sin(u).*cos(v)};
%     f = @(u,v) 12*sin(3*v).*(sin(u).^3);
%     L = LapBel(f,m);
%     [u,v] = ndgrid(linspace(0,pi,200), linspace(0, 2*pi, 200));
%     M = m(u,v);
%     L(pi, pi/2)
%     figure, surf(M{1},M{2},M{3},L(u,v))
%     axis equal, grid on, shading interp, camlight, colormap(prism(55))

%--------------NOW USING CARTESIAN COORDINATES--------
    fid = fopen('vesicle3_sh8.txt','rt');
    ves = textscan(fid,'%f');
    ves = cell2mat(ves);
    ves_mat = reshape(ves, 144, 3);
    %scatter3(ves_mat(:, 1), ves_mat(:, 2),  ves_mat(:, 3) );
    
    a = 2; b = 1.5; c = 1;
    r = 0.15;
    m = @(x,y,z) ((x)/a).^2 + ((y)/b).^2 + ((z)/c).^2;
    f = @(x,y,z) x;
    [x,y,z] = meshgrid(linspace(0,.71),linspace(0,1),linspace(0,.5));
    L = LapBel(f,m);
    L(ves_mat(:, 1), ves_mat(:, 2), ves_mat(:, 3))
    %figure, isosurface(x,y,z,m(x,y,z),1,L(x,y,z),'noshare')
    %colormap(jet(11)), axis equal, grid on, camlight, view(130,30)

end