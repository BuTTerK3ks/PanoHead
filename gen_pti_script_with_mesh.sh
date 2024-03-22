#!/usr/bin/env bash

model="easy-khair-180-gpc0.8-trans10-025000.pkl"

input_dir="models"
output_dir="pti_out"

target_img="misc/images_cropped"
target_seg="misc/images_segmented"

# Count the number of images in the target_img directory
num_images=$(ls -1q $target_img/* | wc -l)

# Loop over the number of images
for ((i=0; i<num_images-1; i++)); do
    # Perform the PTI and save w
    python projector_withseg.py --num-steps=1000 --num-steps-pti=1000 --outdir="${output_dir}" --target_img="${target_img}" --target_seg="${target_seg}" --network "${input_dir}/${model}" --idx "${i}"
    # Generate .mp4 before finetune
    python gen_videos_proj_withseg.py --output="${output_dir}/${model}/${i}/PTI_render/pre.mp4" --latent="${output_dir}/${model}/${i}/projected_w.npz" --trunc 0.7 --network "${input_dir}/${model}" --cfg Head
    # Generate .mp4, .ply mesh and frame images after finetune
    python gen_videos_proj_withseg.py --output="${output_dir}/${model}/${i}/PTI_render/post.mp4" --latent="${output_dir}/${model}/${i}/projected_w.npz" --trunc 0.7 --network "${output_dir}/${model}/${i}/fintuned_generator.pkl" --cfg Head --shapes True --frames True --level 42
done