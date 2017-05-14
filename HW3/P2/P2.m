clc
clear all
close all

ser_time = 0.001091; %s
% 8 threads, 32768 blocks
ave_gpu_time(1) = 0.000348; %s

% 64 threads, 4096 blocks
ave_gpu_time(2) = 0.000070; %s

% 128 threads, 2048 blocks
ave_gpu_time(3) = 0.000061; %s

% 256 threads, 1024 blocks
ave_gpu_time(4) = 0.000061; %s

% 512 threads, 512 blocks
ave_gpu_time(5) = 0.000061; %s
% 512 threads, 1024 blocks
ave_gpu_time(6) = 0.000065; %s

% 1024 threads, 256 blocks
ave_gpu_time(7) = 0.000071; %s
% 1024 threads, 512 blocks
ave_gpu_time(8) = 0.000081; %s
% 1024 threads, 1024 blocks
ave_gpu_time(9) = 0.000101; %s

%%

latTo= 297.784805; %usec
latFrom = 112.473965; %usec
bwTo = 8307.884282; %MBPS = Bpus
bwFrom = 4516.147420; %MBPS

cpuTn = ser_time*1e3; %msec

N = 2^18;

cpuTime = zeros(1, 25);
gpuTime = zeros(1, 25);
k = 1;
for M = 1:25
    cpuTime(k) = cpuTn*M;
    gpuTime(k) = (3*(latTo + N*4/bwTo) + M*ave_gpu_time(4)*1e6 + ...
        (latFrom + N*4/bwFrom))*1e-3;
    k = k+1;
end
figure
hold on
M = 1:25;
plot(M, cpuTime, 'r', 'LineWidth', 2)
plot(M, gpuTime, 'g', 'LineWidth', 2)
box on
grid on
set(gcf, 'Color', 'w')
set(gca, 'fontSize', 14)
xlabel('M operations')
ylabel('Time (ms)')
title('CPU vs. GPU time')
legend('CPU', 'GPU',2)
hold off
legend