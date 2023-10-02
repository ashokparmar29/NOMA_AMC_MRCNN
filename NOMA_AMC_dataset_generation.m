% author : Ashok Parmar
clc; 
clear variables; 
close all;
N = 800;

% symbols sets
symbol_bpsk = [-1+0i, 1+0i];
theta1 = cos(pi/8)+1i*sin(pi/8);
theta2 = cos(pi/16)+1i*sin(pi/16);
symbol_qpsk = theta1 * [0.7071+0.7071i,-0.7071+0.7071i, 0.7071-0.7071i,-0.7071-0.7071i];
symbol_8psk = [1, 0.7071+0.7071i, 1i, -0.7071+0.7071i, -1, -0.7071-0.7071i,-1i,0.7071-0.7071i];
qam = [-1.0000 - 1.0000i
1.0000 - 1.0000i
1.0000 + 1.0000i
-1.0000 + 1.0000i
-3.0000 - 1.0000i
-1.0000 - 3.0000i
1.0000 - 3.0000i
3.0000 - 1.0000i
3.0000 + 1.0000i
1.0000 + 3.0000i
-1.0000 + 3.0000i
-3.0000 + 1.0000i
-3.0000 - 3.0000i
3.0000 - 3.0000i
3.0000 + 3.0000i
-3.0000 + 3.0000i];
qam16 = transpose(qam);

sqrt_mean = sqrt(mean(abs(qam16).^2));
symbol_16qam = qam16/sqrt_mean;

Pf = 0.8; Pn = 0.4;

%select type of modulation at far and near user randomly with equal
%probability

vriance = 0.1; %variance of RV for rayleigh fading, it is tested with dif. values below and above 0.1, curve is smoother for 0.1
setof_symbs = {symbol_bpsk,symbol_qpsk,symbol_8psk,symbol_16qam};
true_lbls  = zeros(1,50000);
data_y = zeros(50000,N);
all_snrs = zeros(1,50000);
all_h = zeros(1,50000);
Mf_lik = zeros(4,16000); 
snrs = [];
data_Y= [];
true_Mods =[];
w=waitbar(0,'Why so impatient, I am workin...');
for snr_db = -10:2:20
    snr = 10^(snr_db/10);
    for iter = 1:50000
        
        xr = randn(1);
        yi = randn(1);
        h =sqrt(vriance*(xr^2+yi^2)); %channel rayleigh,
        all_h(iter) = h;    
        i = randi(4);
        true_lbls(iter) = i-1; 
        j = randi(4);
        
        %now we have i and j i.e. type of modulation for far and near user
        %select symbols
        symb_far = setof_symbs{i};
        symb_near = setof_symbs{j};
        xf = randsample(symb_far,N,true);
        xn = randsample(symb_near,N,true);
        y0 = h*(sqrt(Pf)*xf+ sqrt(Pn)*xn);
        ps = sum(abs(y0).^2)/numel(y0);
        
        noisepower = ps/snr;
        z = sqrt(noisepower*0.5)*complex(randn(size(y0)),randn(size(y0))); %AWGN
        pn = sum(abs(z).^2)/numel(z);
        y = y0+z;
        snr_y = ps/pn;
        all_snrs(iter) = snr_y;
        data_y(iter,:) = y;
        snr_itr(iter) = snr_db;
        waitbar(iter/800000);
    end
    
    snrs = [snrs snr_itr];
    data_Y = [data_Y; data_y];
    true_Mods = [true_Mods true_lbls];
    
end
close(w);
save dataset_NOMA_N_800_1_2.mat data_Y true_Mods snrs;


