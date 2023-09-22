clear all
close all
clc

%% Initialize Parameters and Initial Conditions
x_0 = 100;%[m]
z_0 = 200;%[m]
v_x_0 = 2;%[m/s]
v_z_0 = 1;%[m/s]
theta_0 = 0.01;%[rad]
theta_dot_0 = 0.01;%[rad\s]

P = diag([10, 15, 2, 3, 0.08]).^2;
R = diag([8, 4]).^2;
H = [1, 0, 0, 0, 0; 0, 1, 0, 0, 0];

dt = 1e-2;
T = 40;
t_span = 0:dt:T + dt;

%% Solve Differential Equations
X_real = solve_differential_equations(x_0, v_x_0, z_0, v_z_0, theta_0, theta_dot_0, t_span);

%% Define GPS Vector
gps = define_gps_vector(X_real, R);

%% Perform Extended Kalman Filter
[S, X, P, K, Residual_m, Residual_k, std_k] = perform_extended_kalman_filter...
    (dt,H,R,gps,t_span,P,X_real,x_0,v_x_0,z_0,v_z_0,theta_0,theta_dot_0);

%% Plot Results
plot_results(gps, X, X_real);

% Definitions of the extracted functions are below:

function X_real = solve_differential_equations...
    (x_0, v_x_0, z_0, v_z_0, theta_0, theta_dot_0, t_span)
    % Defining differential equations
    syms x(t) z(t) theta(t) Y
    Eq1 = ...
    diff(z,2) == 0.2*abs(cos(t))*cos(theta) - 0.4*abs(sin(t))*sin(theta);
    Eq2 = ...
    diff(x,2) == 0.4*abs(sin(t))*cos(theta) + 0.2*abs(cos(t))*sin(theta);
    Eq3 = ...
    diff(theta,1) == theta_dot_0; 

    % Turn equations into a vector form
    [Eqs_vector, ~] = odeToVectorField(Eq1, Eq2, Eq3);
    Eqs_Fcn = matlabFunction(Eqs_vector, 'vars', {'t', 'Y'});
    initCond = [x_0, v_x_0, z_0, v_z_0, theta_0];
    [T, Y] = ode45(Eqs_Fcn, t_span, initCond);

    X_real = zeros(5, length(T));
    X_real(1,:) = Y(:,1); % x real
    X_real(2,:) = Y(:,3); % z real
    X_real(3,:) = Y(:,2); % v_x real
    X_real(4,:) = Y(:,4); % v_z real
    X_real(5,:) = Y(:,5); % theta real
end

function gps = define_gps_vector(X_real, R)
    % Initalize gps vector
    gps = zeros(2, length(X_real(1,:)));
    
    % Define the gps matrix
    gps(1,:) = X_real(1,:) + sqrt(R(1,1)) .* randn(1, length(X_real(1,:)));
    gps(2,:) = X_real(2,:) + sqrt(R(2,2)) .* randn(1, length(X_real(2,:)));
end

function [S, X, P, K, Residual_m, Residual_k, std_k] = perform_extended_kalman_filter...
    (dt,H,R,gps,t_span,P,X_real,x_0,v_x_0,z_0,v_z_0,theta_0,theta_dot_0)
    %initialize
    X(:,1) = [x_0,z_0,v_x_0,v_z_0,theta_0]+dt.*[v_x_0,v_z_0,...
        0.4*abs(sin(0))*cos(theta_0)+0.2*abs(cos(0))*sin(theta_0),...
        0.2*abs(cos(0))*cos(theta_0) - 0.4*abs(sin(0))*sin(theta_0),theta_dot_0];
    std_k(:,1) = sqrt(diag(P));
    for i = 1:1:(length(t_span)-1)
         % Time Vector Setting
            T_k(i+1) = t_span(i+1);
            % Prediction of states for next time step
            X(:,i+1) = X(:,i)+dt.*[X(3,i),X(4,i),...
            0.4*abs(sin(t_span(i)))*cos(X(5,i))+0.2*abs(cos(t_span(i)))*sin(X(5,i)),...
            0.2*abs(cos(t_span(i)))*cos(X(5,i)) - 0.4*abs(sin(t_span(i)))*sin(X(5,i)),...
            theta_dot_0].';
            % Jacobian Evaluation at predicted state
            J = [1,0,dt,0,0;...
                 0,1,0,dt,0;...
                 0,0,1,0,dt*(0.2*abs(cos(t_span(i)))*cos(X(5,i)) - 0.4*abs(sin(t_span(i)))*sin(X(5,i)));...
                 0,0,0,1,dt*(-0.4*abs(sin(t_span(i)))*cos(X(5,i))-0.2*abs(cos(t_span(i)))*sin(X(5,i)));...
                 0,0,0,0,1];
            % Covariance Prediction
            P = J*P*J';
        % Measurement Residual
        Residual_m(:,i+1) = gps(:,i+1) - H * X(:,i+1);
        
        % Kalman Filter Residual
        Residual_k(:,i+1) = X_real(:,i+1) - X(:,i+1);
        
        % Std 
        std_k(:,i+1) = sqrt(diag(P));
        
        % Residual Covariance
        S = H * P * H' + R;
        
        % Kalman Gain 
        K = P * H' / S; 
        
        % Update estimate
        X(:,i+1) = X(:,i+1) + K * Residual_m(:,i+1);
        
        % Update Covariance
        P = (eye(5) - K * H) * P; 
    end        

    end
    
function plot_results(gps, X, X_real) 
        
    %X Vs Z
    figure('Name', 'Postion result');

    plot(gps(1,:),gps(2,:),'o',X(1,:),X(2,:),'*',X_real(1,:),X_real(2,:),'-k','LineWidth',0.8)
    xlabel('$X [m]$', 'Interpreter','latex')
    ylabel('$Z [m]$', 'Interpreter','latex')
    set(gca,'fontsize',16)
    box on
    legend('GPS Location','EKF Prediction','Trajectory')
    legend('Location','southeast')
    pbaspect([1 1 1])
end
