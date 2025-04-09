# hybridNanoGPT
testing hybrid models ((mamba2, linear attentions, BASED) + transformer) with the nanogpt objective of reaching a val_loss of 3.28 on FineWeb in the shortest possible time on 8xH100

See: https://github.com/KellerJordan/modded-nanogpt


Current code achieves 3.28 in 181.02s. There are various optimizations needed to bridge the gap between this hybrid speedrun and the current nanoGPT WR. 
I would strongly recommend running this code on a torch build from march 11 2025
pip install --pre torch==2.7.0.dev20250311+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128 --upgrade

Make sure to adjust the cuda version to your machine's specs, I have tested this code on both 12.6 and 12.8. The code will likely fail if you run with the latest torch nightly update, I will update the training code with the next torch stable release.
