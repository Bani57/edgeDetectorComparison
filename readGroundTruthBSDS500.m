clear variables; close all; clc;

bsds500_ground_truth_folder_path='../BSR/BSDS500/data/groundTruth/';

train_images_folder_path=strcat(bsds500_ground_truth_folder_path,'train/');
train_images_paths=dir(strcat(train_images_folder_path,'*.mat'));
num_train_images=size(train_images_paths, 1);
for i = 1:num_train_images
    extractGroundTruthImage(train_images_paths(i).name,train_images_folder_path);
end

validation_images_folder_path=strcat(bsds500_ground_truth_folder_path,'val/');
validation_images_paths=dir(strcat(validation_images_folder_path,'*.mat'));
num_validation_images=size(validation_images_paths, 1);
for i = 1:num_validation_images
    extractGroundTruthImage(validation_images_paths(i).name,validation_images_folder_path);
end

test_images_folder_path=strcat(bsds500_ground_truth_folder_path,'test/');
test_images_paths=dir(strcat(test_images_folder_path,'*.mat'));
num_test_images=size(test_images_paths, 1);
for i = 1:num_test_images
    extractGroundTruthImage(test_images_paths(i).name,test_images_folder_path);
end


function extractGroundTruthImage(image_source_path, image_destination_path)
    image_mat_file=load(image_source_path,'-mat');
    image_regions_and_edges=image_mat_file.groundTruth;
    num_samples=size(image_regions_and_edges, 2);
    image_size=size(image_regions_and_edges{1}.Boundaries);
    average_human_edges=zeros(image_size);
    for i=1:num_samples
        image_edges=image_regions_and_edges{i}.Boundaries;
        average_human_edges=average_human_edges+image_edges;
    end
    average_human_edges=average_human_edges/num_samples;
    [~,image_id,~] = fileparts(image_source_path);
    image_file_new_path=strcat(image_destination_path,image_id,'.png');
    imwrite(average_human_edges,image_file_new_path,'PNG');
end