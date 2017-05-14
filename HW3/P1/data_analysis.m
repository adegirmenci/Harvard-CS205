%PxPy = ['1x1'; '1x2'; '1x4'; '1x8'; '1x16'; '1x32'; '2x1'; '2x2'; '2x4';...
%        '2x8'; '2x16'; '2x32'; '4x1'; '4x2'; '4x4'; '4x8'];
clc
clear all
close all

%% Initialize varaibles
n = 6;
PxLim = 2^(n-1); %32
PyLim = 2^(n-1); %32

nTests = n*(n+1)/2; %21
PxPy = zeros(nTests,1);
PxPyInd = cell(nTests,1);
k = 1;
for i = 1:n
    for j = 1:n
        Px = 2^(i-1);
        Py = 2^(j-1);
        if Px*Py <= 32
            PxPy(k) = Px*Py;
            PxPyInd{k} = sprintf('%dx%d',Px,Py);
            k = k+1;
        end
    end
end

%% Data transfer
tOverallSerial = 4.272136;
tSerial = 0.000957;

timeOverallA = [4.335910; 2.809708; 2.050327; 1.731541; 1.356757; 1.648241;...
    2.541640; 1.967389; 1.874303; 1.430505; 1.271296;...
    1.909365; 1.690745; 1.430541; 1.336865;...
    1.643125; 1.252393; 1.280916;...
    1.184587; 1.073866;...
    1.069406];
timeA = [0.000972; 0.000630; 0.000460; 0.000388; 0.000304; 0.000369;...
    0.000570; 0.000441; 0.000420; 0.000321; 0.000285;...
    0.000428; 0.000379; 0.000321; 0.000300;...
    0.000368; 0.000281; 0.000287;...
    0.000265; 0.000241;...
    0.000240];

timeOverallB = [4.371471; 4.698833; 3.783983; 1.788928; 1.333211; 1.812561;...
    2.641686; 2.030914; 1.863173; 1.380506; 1.263434;...
    3.211366; 2.283479; 1.924114; 1.392740;...
    1.639082; 1.239383; 1.270946;...
    1.292314; 1.085043;...
    0.987773];
timeB = [0.000980; 0.001053; 0.000848; 0.000401; 0.000299; 0.000406;...
    0.000592; 0.000455; 0.000418; 0.000309; 0.000283;...
    0.000720; 0.000512; 0.000431; 0.000312;...
    0.000367; 0.000278; 0.000285;...
    0.000290; 0.000243;...
    0.000221];

%% Plot iteration time

idx = [1  2 7  3 8 12  4 9 13 16  5 10 14 17 19  6 11 15 18 20 21];
tA = [tSerial; timeA(idx)];
tB = [tSerial; timeB(idx)];
xtick = 1:22;
xTLabel = {'Serial', PxPyInd{idx}};

figure
hold on
bar([tA, tB])
colormap summer
box on
grid on
set(gcf, 'Color', 'w');
set(gca, 'fontSize', 14)
set(gca, 'xLim', [0.5 22.5], 'yLim', [0,1.1e-3])
set(gca, 'xTick', xtick,'xTickLabel', xTLabel)
legend('P1A', 'P1B', 0)
xlabel('Number of processes')
ylabel('Average Time per Iteration (s)')
title('P1: Average Time per Iteration vs. Number of processes')
hold off

%% Plot speedup

idx = [1  2 7  3 8 12  4 9 13 16  5 10 14 17 19  6 11 15 18 20 21];
spA = tSerial./timeA;
spB = tSerial./timeB;
xtick = 1:21;
xTLabel = {PxPyInd{idx}};

figure
hold on
bar([spA(idx), spB(idx)])
colormap summer
box on
grid on
set(gcf, 'Color', 'w');
set(gca, 'fontSize', 14)
set(gca, 'xLim', [0.5 21.5], 'yLim', [0,4.5])
set(gca, 'xTick', xtick,'xTickLabel', xTLabel)
legend('P1A', 'P1B', 0)
xlabel('Number of processes')
ylabel('Speedup (s/s)')
title('P1: Speedup vs. Number of processes')
hold off

%% Plot efficiency

idx = [1  2 7  3 8 12  4 9 13 16  5 10 14 17 19  6 11 15 18 20 21];
effA = spA./PxPy;
effB = spB./PxPy;
xtick = 1:21;
xTLabel = {PxPyInd{idx}};

figure
hold on
bar([effA(idx), effB(idx)])
colormap summer
box on
grid on
set(gcf, 'Color', 'w');
set(gca, 'fontSize', 14)
set(gca, 'xLim', [0.5 21.5], 'yLim', [0,1])
set(gca, 'xTick', xtick,'xTickLabel', xTLabel)
legend('P1A', 'P1B', 0)
xlabel('Number of processes')
ylabel('Efficiency')
title('P1: Efficiency vs. Number of processes')
hold off