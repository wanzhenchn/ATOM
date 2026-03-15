# docker.io/rocm/vllm-private:rocm7.2.1_kineto_amd_dev_vllm_custom_2026-03-09-15-49-18
# rocm/vllm-dev:nightly_main_20260118

IMAGE="docker.io/rocm/vllm-private:preview_0.17.0_rocm7.2.1RC5_build78_20260306"

if ! podman image exists "$IMAGE"; then
    podman pull --storage-opt ignore_chown_errors=true "$IMAGE"
fi

podman run --rm -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --network host \
    --shm-size=16G \
    -v /shared/amdgpu/home/hattie_wu_qle:/home/hatwu \
    "$IMAGE" \
    /bin/bash

# docker.io/rocm/vllm-private:rocm7.2.1_kineto_amd_dev_vllm_custom_2026-03-09-15-49-18
# docker exec -it vllm_atom_hattie /bin/bash
# podman exec -it vllm_atom_hattie /bin/bash
# tmux new -s mysession
# tmux attach -t mysession
# --name vllm_atom_hattie \
# rocm/vllm-dev:nightly_main_20260118 \
