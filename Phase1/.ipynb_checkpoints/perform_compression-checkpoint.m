%#==========================================
%# Perform dynamic range compression given CRs and soft gains
%# Author: Nasim Alamdari
%# Last modification:   Dec. 2020
%#==========================================
function out_compressed = perform_compression (audio, CRs, fs , soft_G)


RelT      = 1000e-3;   % Release time (sec)
AttT      = 1e-2;     % Attack time (sec)
CT        = 60;

%% perform compression

DRC_params = [CRs(1), CT-105, AttT, RelT, soft_G(1);...
              CRs(2), CT-105, AttT, RelT, soft_G(2);...
              CRs(3), CT-105, AttT, RelT, soft_G(3);...
              CRs(4), CT-105, AttT, RelT, soft_G(4);...
              CRs(5), CT-105, AttT, RelT, soft_G(5)];

[out_compressed, unCompressed_audio] = DRCFiveBand(double(audio(:)), double(fs), DRC_params, 12) ;


end
