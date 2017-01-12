function prepare_dicoms(dcm_dir)

% get rid of nasty jpeg2000 compression (creates 'converted' dir)
decompressdcm(dcm_dir);

% split dicoms into separate directory per scan
splitSiemensScans(fullfile(dcm_dir,'converted')); 

end

