# Integration Example

This directory shows how to connect your own decoder output to the SMPL head and SMPL layer.

The example is intentionally minimal. It assumes you already have a model that produces:

```text
hidden_states:  (num_decoder_layers, batch_size, num_queries, hidden_dim)
cam_intrinsics: (batch_size, 1, 3, 3)
```

## What To Modify

- Replace the dummy `hidden_states` input with your decoder output.
- Replace `input_size` and `default_focal` with your project's camera setup.
- Replace paths loaded from `config/smpl_paths.py` with your own config system if needed.
