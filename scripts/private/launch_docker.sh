# podman pull --storage-opt ignore_chown_errors=true docker.io/rocm/vllm-private:preview_0.17.0_rocm7.2.1RC5_build78_20260306

podman run --rm -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --network host \
    --shm-size=16G \
    --name vllm_atom_hattie \
    -v /shared/amdgpu/home/hattie_wu_qle:/home/hatwu \
    docker.io/rocm/vllm-private:preview_0.17.0_rocm7.2.1RC5_build78_20260306 \
    /bin/bash

# docker exec -it vllm_atom_hattie /bin/bash
# podman exec -it vllm_atom_hattie /bin/bash
# tmux new -s mysession
# tmux attach -t mysession
# rocm/vllm-dev:nightly_main_20260118 \
