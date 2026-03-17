Approach:

1. Use same read pipeline, encoder, and decoder as in class
2. Encode pixels one by one, as done in class
3. Uses 256 independent adaptive `VectorCountSymbolModel<u8>` contexts, one per possible predicted value. Conditioning on the prediction sharpens each model since pixels with similar predicted brightness have similar residual distributions, thus leading to lower entropy for each context's model.
4. For predictions, use both temporal neighbor from previous frame (temporal) and the left/top spatial neighbors in the current frame. At row/column boundaries, average available neighbors with the prior-frame value.
6. Updates counts for each residual symbol as data is processed (fully adaptive, no pre-training pass).

```bash
cargo run --release --bin assgn1 -- -count 50 [-in 'data/file.mp4']
```
**bourne.mp4** (54.72 MB, 5:22): ```50 frames encoded, average size (bits): 4126257, compression ratio: 4.02```

**chuckyy_poetry.mp4** (80.25 MB, 3:14): ```50 frames encoded, average size (bits): 4449731, compression ratio: 3.73```

**brokeboi.mp4** (13.00 MB, 3:49): ```50 frames encoded, average size (bits): 798729, compression ratio: 2.31```

