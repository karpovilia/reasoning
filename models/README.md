# Included weights

`MANIFEST.json` pins the included lexical observer and GRU/MLP action-value baselines. Their architecture and preprocessing are implemented in `reasoning_control`. They are not weights of the segment Transformer, Qwen predictor or recurrent prefix detector.

Use `git lfs pull` to materialize these files, then `make weights` to verify them. Large-model training and inference require a checkpoint supplied explicitly by the caller. Base pretrained model revisions are specified in `configs`.
