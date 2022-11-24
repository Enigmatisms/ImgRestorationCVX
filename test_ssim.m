function result = test_ssim(img,ref)
%% 分别计算rgb三通道，取平均
img = double(img);
ref = double(ref);
img_r = img(:,:,1);
img_g = img(:,:,2);
img_b = img(:,:,3);
ref_r = ref(:,:,1);
ref_g = ref(:,:,2);
ref_b = ref(:,:,3);
ssim_r = ssim(img_r,ref_r);
ssim_g = ssim(img_g,ref_g);
ssim_b = ssim(img_b,ref_b);
ssim_result = (ssim_r+ssim_g+ssim_b)/3;

%% 转ycbcr，计算y通道

% img = double(img);
% ref = double(ref);
% img_ycbcr = rgb2ycbcr(img);
% ref_ycbcr = rgb2ycbcr(ref);
% img_y = img_ycbcr(:,:,1);
% ref_y = ref_ycbcr(:,:,1);
% ssim_result = ssim(img_y,ref_y);

result = ssim_result;