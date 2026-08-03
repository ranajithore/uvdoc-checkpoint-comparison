# UVDoc Checkpoint Comparison

Byte-level comparison of **PaddlePaddle's UVDoc release** against the **original ETH Zurich UVDoc**, plus a determinism proof for the model up to the 2D grid.

Three questions, answered from the shipped artifacts:

1. Did PaddlePaddle train UVDoc, or convert the original weights?
2. Does any learnable layer sit after the 2D grid prediction?
3. Does the same image always produce the same grid, bit for bit?

**Results: all 8,011,911 stored weights are bit-for-bit identical, the graph has zero parameter-bearing operations after the grid-producing convolution, and the grid is exactly reproducible across runs with zero tolerance.**

---

## The two models

| | Original UVDoc | PaddlePaddle UVDoc |
|---|---|---|
| Repo | [tanguymagne/UVDoc](https://github.com/tanguymagne/UVDoc) | [PaddlePaddle/PaddleX](https://github.com/PaddlePaddle/PaddleX) |
| Weights | `model/best_model.pkl` | [huggingface.co/PaddlePaddle/UVDoc](https://huggingface.co/PaddlePaddle/UVDoc) |
| Architecture source | [`model.py`](https://github.com/tanguymagne/UVDoc/blob/main/model.py) | [`modeling/uvdoc.py`](https://github.com/PaddlePaddle/PaddleX/blob/release/3.7/paddlex/inference/models/image_unwarping/modeling/uvdoc.py) |
| Framework | PyTorch | PaddlePaddle |
| License | MIT, © 2023 Tanguy MAGNE | Apache-2.0 (packaging) |
| Paper | [UVDoc: Neural Grid-based Document Unwarping](https://arxiv.org/abs/2302.02887), SIGGRAPH Asia 2023 | — |

---

## Input / output formats

### Original UVDoc (PyTorch)

```
INPUT   RGB, float32, /255, resized to 488x712 (WxH), NCHW
        shape (1, 3, 712, 488)
        demo.py: cv2.imread -> cvtColor(BGR2RGB) -> /255 -> resize -> transpose(2,0,1)

MODEL   returns a TUPLE of two grids, no image
        point_positions2D  (1, 2, 45, 31)   the unwarping grid
        point_positions3D  (1, 3, 45, 31)   auxiliary 3D mesh, training only

UNWARP  utils.bilinear_unwarping(warped_img, point_positions, img_size)
        a standalone helper, NOT part of the model:
            F.interpolate(grid, size=(712, 488), mode="bilinear", align_corners=True)
            F.grid_sample(image, grid.transpose(1,2).transpose(2,3), align_corners=True)

OUTPUT  712x488 image, converted RGB -> BGR, uint8
```

### PaddlePaddle UVDoc (exported PIR model)

```
INPUT   name "image", float32, NCHW, shape (-1, 3, -1, -1)   [dynamic H, W]
        predictor: ReadImage(format="BGR") -> Normalize(mean=0, std=1, scale=1/255)
                   -> ToCHWImage -> ToBatch
        NOTE: no BGR->RGB conversion; Normalize is a pure per-channel scale+shift

MODEL   UVDocNet.forward() inlines the unwarp:
            image = x                                    # full-res input kept
            x = F.interpolate(x, size=[712, 488], align_corners=True)
            out = head(backbone(x))                      # (N, 2, 45, 31)  the grid
            bm  = F.interpolate(out, size=(h_ori, w_ori), align_corners=True)
            result = F.grid_sample(image, bm.transpose([0,2,3,1]), align_corners=True)

OUTPUT  single tensor, shape (-1, 3, -1, -1) -- the warped image at the INPUT's
        native resolution; the grid is not exposed
        postprocess: squeeze -> HWC -> *255 -> channel reverse -> uint8
```

### Two behavioural differences

The weights are identical, but the shipping pipelines are not:

| | Original | PaddlePaddle |
|---|---|---|
| Channel order into the network | **RGB** | **BGR** (no conversion applied) |
| Output resolution | always 712×488 | input's native resolution |

The network was trained on RGB, so the first difference means the Paddle pipeline feeds the network channel-swapped input. Measured over eight colour documents, the swap changes the warped output by a mean RMS of **0.048** on [0,1] pixel values, up to 1.0 locally — roughly 160× the float32 kernel noise. It has no effect on grayscale inputs, where R=G=B. The second difference raises output resolution above the original's fixed 712×488.

---

## Running

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy torch paddlepaddle==3.0.0 pillow
```

Download the checkpoints (not committed — 64 MB of third-party binaries):

```bash
mkdir -p checkpoints
curl -fsSL -o checkpoints/best_model.pkl \
  "https://github.com/tanguymagne/UVDoc/raw/main/model/best_model.pkl"
curl -fsSL -o checkpoints/inference.pdiparams \
  "https://huggingface.co/PaddlePaddle/UVDoc/resolve/main/inference.pdiparams"
curl -fsSL -o checkpoints/inference.json \
  "https://huggingface.co/PaddlePaddle/UVDoc/resolve/main/inference.json"
```

```bash
.venv/bin/python compare.py              # weights + architecture
.venv/bin/python verify_determinism.py   # same image -> same grid, bit for bit
```

Both exit 0 when all checks pass.

Verified artifacts:

| File | Bytes | SHA-256 (first 16) |
|---|---:|---|
| `best_model.pkl` | 32,158,393 | `7e90861b8a516eb4` |
| `inference.pdiparams` | 32,054,311 | `810488899520e0da` |
| `inference.json` | 190,986 | `2c2bc3e0f15e782c` |

---

## Report

### Part 1 — Weights (bit-wise)

```
pytorch tensors         : 304
  excluded (BN counters): 45  dtype={'int64'}  — training-only, never read at inference
  comparable            : 259
paddle parameters       : 259
dtypes                  : pytorch={'float32'}  paddle={'float32'}
byte order              : {'='}  (native: True)
non-finite values       : 0  (NaN/Inf would invalidate ==)

paired 1:1              : 259/259
paddle unpaired         : 0
pytorch unpaired        : 0
elements compared       : 8,011,911
elements equal          : 8,011,911   (differing: 0)
max |a - b|  (float64)  : 0.0
bytes compared          : 32,047,644
bytes NOT examined      : paddle 0, pytorch 0

>>> WEIGHTS BIT-WISE IDENTICAL
```

Three independent methods, all agreeing: raw-byte identity, `np.array_equal`, and `max|a−b|` in float64. **Zero bytes unexamined on either side.**

### Part 2 — Architecture (structural)

```
paddle params declared  : 259
paddle params consumed  : 251
paddle params UNUSED    : 8   (shipped but never wired into the graph)
    batch_norm2d_44.b_0    (32,)              <- out_point_positions3D.1.bias
    batch_norm2d_44.w_0    (32,)              <- out_point_positions3D.1.weight
    batch_norm2d_44.w_1    (32,)              <- out_point_positions3D.1.running_mean
    batch_norm2d_44.w_2    (32,)              <- out_point_positions3D.1.running_var
    conv2d_45.w_0          (32, 128, 5, 5)    <- out_point_positions3D.0.weight
    conv2d_46.b_0          (3,)               <- out_point_positions3D.3.bias
    conv2d_46.w_0          (3, 32, 5, 5)      <- out_point_positions3D.3.weight
    p_re_lu_1.w_0          (1,)               <- out_point_positions3D.2.weight

learnable layer census:
  paddle  :  90   {'conv2d': 45, 'batch_norm': 44, 'prelu': 1}
  pytorch :  90   {'conv2d': 45, 'batch_norm': 44, 'prelu': 1}

per-type ordered sequence (order-invariant test):
  batch_norm   paddle= 44  pytorch= 44  ORDERED EQUAL: True
  conv2d       paddle= 45  pytorch= 45  ORDERED EQUAL: True
  prelu        paddle=  1  pytorch=  1  ORDERED EQUAL: True

final learnable op      : pd_op.conv2d, kernel (2, 32, 5, 5)  -> 2-channel grid
ops after the grid      : ['builtin.combine', 'pd_op.bilinear_interp',
                           'pd_op.transpose', 'pd_op.grid_sample', 'pd_op.fetch']
of which parameter-bearing : 0
model outputs           : 1

>>> 2D PATH STRUCTURALLY IDENTICAL; nothing learnable after the grid
```

Layer sequences are compared **per type**, not positionally interleaved: a PyTorch `state_dict` serialises in `__init__` definition order while a Paddle graph is in execution order, so the interleaving legitimately differs.

### Part 3 — Grid determinism (zero tolerance)

The grid is not exposed by the exported graph, and requesting an intermediate tensor via `fetch_list` stalls the PIR interpreter. The coordinate probe makes this exact rather than a workaround: on a coordinate-ramp input (ch0 = x, ch1 = y) bilinear interpolation reproduces the ramp exactly, so the warped output **is** the grid, and bit-equality of the output *is* bit-equality of the grid.

```
CHECK 1  —  GRID, run 5x per resolution (output IS the grid)
  17 resolutions from 170x296 to 1150x720, every one bit-identical
  >>> GRID FULLY DETERMINISTIC   (17 resolutions x 5 runs)

CHECK 2  —  REAL IMAGES, run 5x each
  18 document photographs x 5 runs, 91,107,624 bytes compared per repeat
  >>> OUTPUTS FULLY DETERMINISTIC

>>> EXACT. Zero tolerance, zero differing bits.
```

Every comparison is raw-byte equality; a single differing bit fails. Check 1 covers the grid at every input shape the corpus uses. Check 2 confirms it on real content — a differing grid would sample different coordinates and change the output on any textured image.

This guarantee is absolute in a way no cross-implementation comparison can be: one implementation run repeatedly *must* agree bit for bit, whereas two different implementations of convolution round differently and never can. It holds for a fixed model, runtime and machine. **Re-run after changing the paddlepaddle version, the CPU architecture, the thread count, or the execution provider** — each can select a different kernel, and a different kernel means different bits.

---

## Findings

### 1. PaddlePaddle did not train UVDoc

All 8,011,911 weights are bit-identical to the ETH Zurich checkpoint. No training run — not even a single fine-tuning step — leaves weights bit-identical. This is a format conversion.

What PaddlePaddle did to the weights:

| # | Operation |
|---|---|
| 1 | Copied all 8,011,911 values verbatim — zero numerical change |
| 2 | Dropped the 45 `num_batches_tracked` int64 counters (Paddle BN has no equivalent field) |
| 3 | Renamed keys to construction order — `resnet_head.1.weight` → `batch_norm2d_0.w_0` |
| 4 | Restructured `nn.Sequential` indices into named submodules |
| 5 | Kept the 3D head (8 tensors, ~410 KB) in the file but left it unwired |
| 6 | **Did not** fold BatchNorm into conv — 44 `batch_norm_` ops survive in the graph |
| 7 | **Did not** quantise — everything is float32 |
| 8 | Inlined `utils.bilinear_unwarping` into `forward()` |

Item 6 is notable: BatchNorm folding is a standard, near-free inference optimisation, and it was not applied. Together with items 1 and 7, this is a format-and-naming translation with no numerical modification.

The file-size difference is entirely container format:

| | bytes |
|---|---:|
| identical weight payload | 32,047,644 |
| PyTorch ZIP overhead (306 entries, headers, 64-byte alignment, `data.pkl` with all 304 key names) | 110,749 |
| Paddle overhead (259 × ~26 B per-tensor header; names live in `inference.json`) | 6,667 |

### 2. Nothing learnable follows the 2D grid

Exactly one convolution produces ≤3 output channels — `conv2d_44.w_0`, shape **(2, 32, 5, 5)**, i.e. the (x, y) grid. Everything after it and its bias epilogue is parameter-free: `bilinear_interp → transpose → grid_sample → fetch`. The model's sole output is the result of `grid_sample`.

Corroboration: the graph has 45 convolutions but only 44 BatchNorms. The one convolution with no BatchNorm and no activation is the grid head, exactly as `out_point_positions2D: [[128, 32], [32, 2]]` specifies.

This is by design — in the original repo the unwarp is a standalone utility function, not a network layer. PaddleX only relocated it into `forward()`.

**Practical consequence:** the grid alone fully determines the output. Returning the grid instead of the warped image, applying it with `cv2.remap`, composing it with another homography, or blending it toward identity cannot lose learned behaviour, because there is none downstream. Changing the *pre*-block — the `/255` scaling or the 712×488 resize — would break it, because that feeds the frozen weights.

### 3. Attribution

The weights are the MIT-licensed original, © 2023 Tanguy MAGNE, verified above as bit-identical. PaddleOCR redistributes them under Apache-2.0, with no citation in the source file, the model card, or the documentation. The MIT licence requires the copyright notice to be retained in redistributions.

---

## Scope and limitations

| Claim | Status | Established by |
|---|---|---|
| Weights bit-wise identical | **Proven** | `compare.py` |
| PaddlePaddle did not train UVDoc | **Proven** | follows from the above |
| 2D path structurally identical | **Proven** | `compare.py` |
| Nothing learnable after the grid | **Proven** | `compare.py` |
| Same image yields the same grid, bit for bit | **Proven** | `verify_determinism.py` |
| The two shipping *pipelines* behave identically | **False** | see *Two behavioural differences* |

`compare.py` is a static analysis and does not execute either model, so it cannot detect differences in settings a checkpoint does not store — BatchNorm epsilon, conv `padding_mode` edge semantics, `align_corners`. Those were confirmed to match by reading both sources; the channel-order difference is documented above.

Scope is limited to the two artifacts listed above. PaddlePaddle also distributes UVDoc via a BOS bucket and a safetensors variant, which were not examined.

---

## Repository layout

```
uvdoc-checkpoint-comparison/
├── README.md
├── LICENSE                  MIT
├── compare.py               weights + architecture, static analysis
├── verify_determinism.py    same image -> same grid, bit for bit
├── images/                  18 document photographs, 2.1 MB, committed
├── .gitignore
└── checkpoints/             not committed — download with the commands above
```

`images/` holds camera-captured documents, receipts and forms in several scripts, 170×296 to 1150×720, from [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) `doc/imgs` (Apache-2.0) plus PaddleX's own demo image for this model. They are committed so the determinism test runs offline and reproducibly.

---

## License

MIT — see [LICENSE](LICENSE). The third-party material this operates on keeps its own
terms: the UVDoc weights and architecture are MIT, © 2023 Tanguy MAGNE; PaddleX and the
document photographs in `images/` are Apache-2.0, © PaddlePaddle Authors.
