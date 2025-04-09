# hybridNanoGPT
testing hybrid models ((mamba2, linear attentions, BASED) + transformer) with the nanogpt objective of reaching a val_loss of 3.27 on FineWeb in the shortest possible time on 8xH100

See: https://github.com/KellerJordan/modded-nanogpt


Current code achieves 3.27 in about 6 minutes. There are various optimizations needed to bridge the gap between this hybrid speedrun and the current nanoGPT WR. 
