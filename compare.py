#!/usr/bin/env python3
"""
================================================================================
UVDoc checkpoint comparison  —  PaddlePaddle release  vs.  original ETH Zurich
================================================================================

PURPOSE
-------
Determine, from the shipped artifacts alone:
  1. Did PaddlePaddle train UVDoc, or convert the original weights?
  2. Does any learnable layer sit after the 2D grid prediction?

INPUTS
------
  checkpoints/best_model.pkl        github.com/tanguymagne/UVDoc   PyTorch, MIT,
                                    (c) 2023 Tanguy MAGNE
  checkpoints/inference.json        huggingface.co/PaddlePaddle/UVDoc  PIR graph
  checkpoints/inference.pdiparams   huggingface.co/PaddlePaddle/UVDoc  weights

DEPENDENCIES
------------
  numpy, torch (CPU is fine), paddlepaddle==3.0.0

USAGE
-----
  python compare.py                 # exit code 0 = all checks passed

WHAT THIS SCRIPT PROVES
-----------------------
  [PROVEN]     Weight identity. Every stored float is compared three
               independent ways: raw-byte equality, np.array_equal, and
               max|a-b| in float64. 100% of weight bytes on both sides are
               examined, and the script reports how many bytes it did NOT
               examine (expected: zero). Bit-identical weights are only
               achievable by copying -- no training run, however short,
               reproduces them.

  [PROVEN]     Structural identity of the 2D path: the ordered, per-type
               sequence of conv / batchnorm / prelu shapes, plus the absence of
               any parameter-bearing op after the grid-producing conv.

  [OUT OF     Settings a checkpoint does not store. This is a static analysis
   SCOPE]     and does not execute either model, so it cannot inspect BatchNorm
              epsilon, conv padding_mode edge semantics, align_corners or input
              channel order. Those were checked by reading both sources.

              For run-to-run reproducibility of the grid, see
              verify_determinism.py.

METHOD
------
  Weights      Tensors are paired 1:1 on (shape, raw bytes) and each pair is
               then re-checked with np.array_equal and with max|a-b| computed
               in float64. Byte totals are reconciled so that any tensor left
               unexamined is reported rather than silently skipped.

               The 45 `num_batches_tracked` int64 scalars in the PyTorch
               checkpoint are excluded: they are BatchNorm batch counters used
               only to derive a cumulative moving average during training, are
               never read at inference, and have no equivalent field in
               Paddle's BatchNorm2D. The exclusion is reported in the output.

  Architecture Layer sequences are compared per type rather than positionally
               interleaved, because a PyTorch state_dict serialises in
               __init__ definition order while a Paddle graph is in execution
               order.
"""

import sys
from collections import Counter, defaultdict

import numpy as np
import paddle
import torch

PYTORCH_CHECKPOINT_PATH = "checkpoints/best_model.pkl"
PADDLE_MODEL_PREFIX = "checkpoints/inference"

# Paddle op names for the parameter-bearing (learnable) layer types.
LEARNABLE_OP_NAMES = ("pd_op.conv2d", "pd_op.batch_norm_", "pd_op.prelu")

# PyTorch BatchNorm batch counters: training bookkeeping, not weights.
BATCH_COUNTER_SUFFIX = "num_batches_tracked"


# ================================================================== loading
def load_pytorch_checkpoint():
    """Return {tensor_name: np.ndarray} for every tensor in best_model.pkl."""
    checkpoint = torch.load(
        PYTORCH_CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
    state_dict = (
        checkpoint["model_state"]
        if isinstance(checkpoint, dict) and "model_state" in checkpoint
        else checkpoint
    )
    # ascontiguousarray guarantees .tobytes() reflects logical element order
    # even if a tensor happened to be stored non-contiguously.
    return {
        name: np.ascontiguousarray(tensor.detach().cpu().numpy())
        for name, tensor in state_dict.items()
        if hasattr(tensor, "detach")
    }


def load_paddle_inference_model():
    """Load the exported PIR model.

    Returns (parameters, graph_ops, program, model_outputs). `program` is
    returned only so the caller can keep it alive: the Op and Value handles in
    `graph_ops` are views into it and go empty if it is garbage-collected.
    """
    paddle.enable_static()
    executor = paddle.static.Executor(paddle.CPUPlace())
    program, _input_names, model_outputs = paddle.static.load_inference_model(
        PADDLE_MODEL_PREFIX, executor
    )
    graph_ops = list(program.global_block().ops)
    scope = paddle.static.global_scope()

    parameters = {}
    for op in graph_ops:
        if op.name() == "builtin.parameter":
            parameter_name = op.attrs()["parameter_name"]
            variable = scope.find_var(parameter_name)
            if variable is not None:
                parameters[parameter_name] = np.ascontiguousarray(
                    np.array(variable.get_tensor())
                )
    return parameters, graph_ops, program, model_outputs


def parameter_name_for_value(value):
    """Parameter name if `value` is produced by builtin.parameter, else None.

    Never key a dict on a PIR Value; resolve through its defining op instead.
    """
    try:
        defining_op = value.get_defining_op()
        if defining_op is not None and defining_op.name() == "builtin.parameter":
            return defining_op.attrs()["parameter_name"]
    except Exception:
        pass
    return None


def parameters_consumed_by(op, parameters):
    """Parameter names that `op` reads, in operand order."""
    consumed = []
    for operand in op.operands_source():
        name = parameter_name_for_value(operand)
        if name is not None and name in parameters:
            consumed.append(name)
    return consumed


# ========================================================== part 1: weights
def compare_weights(pytorch_tensors, paddle_parameters):
    """Bit-wise comparison of every stored weight.

    Returns (weights_are_identical, name_mapping) where name_mapping maps a
    Paddle parameter name to its PyTorch counterpart. Failures are collected
    and reported rather than raised, so a verdict always prints.
    """
    print("=" * 78)
    print("PART 1  —  WEIGHTS  (bit-wise)")
    print("=" * 78)

    failures = []

    # --- exclude BatchNorm batch counters (see METHOD in module docstring) --
    pytorch_weights = {
        name: tensor
        for name, tensor in pytorch_tensors.items()
        if not name.endswith(BATCH_COUNTER_SUFFIX)
    }
    batch_counters = {
        name: tensor
        for name, tensor in pytorch_tensors.items()
        if name.endswith(BATCH_COUNTER_SUFFIX)
    }
    counter_dtypes = {str(t.dtype) for t in batch_counters.values()} or {"-"}

    print(f"  pytorch tensors         : {len(pytorch_tensors)}")
    print(f"    excluded (BN counters): {len(batch_counters)}  dtype={counter_dtypes}"
          f"  — training-only, never read at inference")
    print(f"    comparable            : {len(pytorch_weights)}")
    print(f"  paddle parameters       : {len(paddle_parameters)}")

    # --- guard: byte comparison is only valid for matching dtype/byte order -
    pytorch_dtypes = {str(t.dtype) for t in pytorch_weights.values()}
    paddle_dtypes = {str(t.dtype) for t in paddle_parameters.values()}
    print(f"  dtypes                  : pytorch={pytorch_dtypes}  "
          f"paddle={paddle_dtypes}")
    if pytorch_dtypes != {"float32"} or paddle_dtypes != {"float32"}:
        failures.append(
            f"unexpected dtypes: pytorch={pytorch_dtypes} paddle={paddle_dtypes}"
        )

    byte_orders = {
        t.dtype.byteorder
        for t in list(pytorch_weights.values()) + list(paddle_parameters.values())
    }
    byte_order_is_native = byte_orders <= {"=", "|"}
    print(f"  byte order              : {byte_orders}  "
          f"(native: {byte_order_is_native})")
    if not byte_order_is_native:
        failures.append(f"non-native byte order present: {byte_orders}")

    # --- guard: NaN/Inf make == and array_equal misbehave -------------------
    non_finite_count = sum(
        int(np.sum(~np.isfinite(t))) for t in paddle_parameters.values()
    )
    print(f"  non-finite values       : {non_finite_count}  "
          f"(NaN/Inf would invalidate ==)")
    if non_finite_count:
        failures.append(f"{non_finite_count} non-finite values present")

    # --- 1:1 pairing on (shape, raw bytes) ----------------------------------
    # pop() makes the mapping a bijection: each PyTorch tensor can be claimed
    # once, so coincidentally-identical tensors cannot be double-counted.
    pytorch_by_content = defaultdict(list)
    for name, tensor in pytorch_weights.items():
        pytorch_by_content[(tuple(tensor.shape), tensor.tobytes())].append(name)

    name_mapping = {}
    unpaired_paddle = []
    max_absolute_difference = 0.0
    equal_element_count = 0
    total_element_count = 0
    compared_byte_count = 0

    for paddle_name, paddle_tensor in paddle_parameters.items():
        content_key = (tuple(paddle_tensor.shape), paddle_tensor.tobytes())  # raw bits
        if not pytorch_by_content.get(content_key):
            unpaired_paddle.append(paddle_name)
            continue

        pytorch_name = pytorch_by_content[content_key].pop()
        pytorch_tensor = pytorch_weights[pytorch_name]

        if paddle_tensor.shape != pytorch_tensor.shape:                    # shape
            failures.append(
                f"shape mismatch {paddle_name} {paddle_tensor.shape} "
                f"vs {pytorch_name} {pytorch_tensor.shape}"
            )
        if not np.array_equal(paddle_tensor, pytorch_tensor):              # elementwise
            failures.append(f"array_equal failed for {paddle_name} / {pytorch_name}")

        difference = (                                                     # numeric
            float(
                np.max(
                    np.abs(
                        paddle_tensor.astype(np.float64)
                        - pytorch_tensor.astype(np.float64)
                    )
                )
            )
            if paddle_tensor.size
            else 0.0
        )
        max_absolute_difference = max(max_absolute_difference, difference)

        equal_element_count += int(np.sum(paddle_tensor == pytorch_tensor))
        total_element_count += paddle_tensor.size
        compared_byte_count += paddle_tensor.nbytes
        name_mapping[paddle_name] = pytorch_name

    unpaired_pytorch = [n for names in pytorch_by_content.values() for n in names]
    paddle_total_bytes = sum(t.nbytes for t in paddle_parameters.values())
    pytorch_total_bytes = sum(t.nbytes for t in pytorch_weights.values())

    print(f"\n  paired 1:1              : "
          f"{len(name_mapping)}/{len(paddle_parameters)}")
    print(f"  paddle unpaired         : {len(unpaired_paddle)}  {unpaired_paddle[:5]}")
    print(f"  pytorch unpaired        : {len(unpaired_pytorch)}  "
          f"{unpaired_pytorch[:5]}")
    print(f"  elements compared       : {total_element_count:,}")
    print(f"  elements equal          : {equal_element_count:,}   "
          f"(differing: {total_element_count - equal_element_count})")
    print(f"  max |a - b|  (float64)  : {max_absolute_difference!r}")
    print(f"  bytes compared          : {compared_byte_count:,}")
    print(f"  bytes NOT examined      : "
          f"paddle {paddle_total_bytes - compared_byte_count}, "
          f"pytorch {pytorch_total_bytes - compared_byte_count}")

    if unpaired_paddle or unpaired_pytorch:
        failures.append(
            f"{len(unpaired_paddle)} paddle / {len(unpaired_pytorch)} pytorch unpaired"
        )
    if total_element_count != equal_element_count:
        failures.append(f"{total_element_count - equal_element_count} elements differ")
    if max_absolute_difference != 0.0:
        failures.append(f"max abs difference {max_absolute_difference!r} != 0")
    if (compared_byte_count != paddle_total_bytes
            or compared_byte_count != pytorch_total_bytes):
        failures.append("not all weight bytes were examined")

    weights_are_identical = not failures
    verdict = (
        "WEIGHTS BIT-WISE IDENTICAL" if weights_are_identical else "WEIGHTS DIFFER"
    )
    print(f"\n  >>> {verdict}")
    for failure in failures:
        print(f"      ! {failure}")
    return weights_are_identical, name_mapping


# ===================================================== part 2: architecture
def compare_architecture(
    pytorch_tensors, paddle_parameters, graph_ops, model_outputs, name_mapping
):
    """Structural comparison of the two computation graphs.

    Uses the verified bijection from part 1 to decide which PyTorch tensors
    belong to the live 2D path, rather than matching on key-name substrings.
    """
    print("\n" + "=" * 78)
    print("PART 2  —  ARCHITECTURE  (structural)")
    print("=" * 78)

    failures = []

    # --- which paddle parameters does the graph actually consume? -----------
    first_consuming_op_index = {}
    for op_index, op in enumerate(graph_ops):
        if op.name() == "builtin.parameter":
            continue
        for name in parameters_consumed_by(op, paddle_parameters):
            first_consuming_op_index.setdefault(name, op_index)

    unused_paddle_parameters = sorted(
        set(paddle_parameters) - set(first_consuming_op_index)
    )

    print(f"  paddle params declared  : {len(paddle_parameters)}")
    print(f"  paddle params consumed  : {len(first_consuming_op_index)}")
    print(f"  paddle params UNUSED    : {len(unused_paddle_parameters)}"
          f"   (shipped but never wired into the graph)")
    for name in unused_paddle_parameters:
        print(f"      {name:<22} {str(paddle_parameters[name].shape):<18} "
              f"<- {name_mapping.get(name, '?')}")

    # PyTorch tensors on the live path = all comparable ones, minus the
    # counterparts of the unused Paddle parameters. Derived from the proven
    # pairing rather than from key-name substrings.
    unused_pytorch_counterparts = {
        name_mapping[n] for n in unused_paddle_parameters if n in name_mapping
    }
    pytorch_live_path = {
        name: tensor
        for name, tensor in pytorch_tensors.items()
        if not name.endswith(BATCH_COUNTER_SUFFIX)
        and name not in unused_pytorch_counterparts
    }
    print(f"  pytorch tensors on live path: {len(pytorch_live_path)}  "
          f"({len(unused_pytorch_counterparts)} excluded via the verified pairing)")

    # --- build (layer_kind, shape) sequences for both sides -----------------
    paddle_layer_sequence = []
    for op in graph_ops:
        if op.name() not in LEARNABLE_OP_NAMES:
            continue
        consumed = parameters_consumed_by(op, paddle_parameters)
        if not consumed:
            continue
        layer_kind = op.name().replace("pd_op.", "").rstrip("_")
        # For conv the kernel is the highest-rank parameter; for BN every
        # parameter is (C,); for prelu it is (1,). Picking the highest-rank,
        # largest parameter yields the kernel for conv and the channel vector
        # otherwise -- in all three cases the shape that identifies the layer.
        representative = max(
            consumed,
            key=lambda n: (paddle_parameters[n].ndim, paddle_parameters[n].size),
        )
        paddle_layer_sequence.append(
            (layer_kind, tuple(paddle_parameters[representative].shape))
        )

    pytorch_layer_sequence = []
    for name, tensor in pytorch_live_path.items():
        if name.endswith(".weight") and tensor.ndim == 4:
            pytorch_layer_sequence.append(("conv2d", tuple(tensor.shape)))
        elif name.endswith("running_mean"):
            pytorch_layer_sequence.append(("batch_norm", tuple(tensor.shape)))
        elif tensor.shape == (1,):
            # PReLU has no distinguishing key name in this checkpoint, so shape
            # (1,) is the only available signal. num_batches_tracked is also
            # shape (1,) in some layers but is already excluded above. The count
            # is cross-checked against the Paddle side below.
            pytorch_layer_sequence.append(("prelu", tuple(tensor.shape)))

    print(f"\n  learnable layer census:")
    print(f"    paddle  : {len(paddle_layer_sequence):>3}   "
          f"{dict(Counter(kind for kind, _ in paddle_layer_sequence))}")
    print(f"    pytorch : {len(pytorch_layer_sequence):>3}   "
          f"{dict(Counter(kind for kind, _ in pytorch_layer_sequence))}")

    # --- per-type ordered comparison ----------------------------------------
    # Comparing the two sequences positionally would be invalid: a PyTorch
    # state_dict serialises in __init__ definition order while a Paddle graph
    # is in execution order, so the interleaving legitimately differs. The
    # original residual block defines (conv1, conv2, bn1, bn2) but executes
    # (conv1, bn1, conv2, bn2). The order-invariant test is per layer type:
    # the Nth conv on one side must match the Nth conv on the other.
    layer_kinds = sorted(
        {kind for kind, _ in paddle_layer_sequence}
        | {kind for kind, _ in pytorch_layer_sequence}
    )
    print(f"\n  per-type ordered sequence (order-invariant test):")
    for layer_kind in layer_kinds:
        paddle_shapes = [s for k, s in paddle_layer_sequence if k == layer_kind]
        pytorch_shapes = [s for k, s in pytorch_layer_sequence if k == layer_kind]
        sequences_match = paddle_shapes == pytorch_shapes
        print(f"    {layer_kind:<12} paddle={len(paddle_shapes):>3}  "
              f"pytorch={len(pytorch_shapes):>3}  ORDERED EQUAL: {sequences_match}")
        if not sequences_match:
            failures.append(f"{layer_kind} sequence differs")
            for index, (a, b) in enumerate(zip(paddle_shapes, pytorch_shapes)):
                if a != b:
                    print(f"        first difference at {layer_kind} "
                          f"#{index}: {a} vs {b}")
                    break
            if len(paddle_shapes) != len(pytorch_shapes):
                print(f"        count differs: {len(paddle_shapes)} "
                      f"vs {len(pytorch_shapes)}")

    interleaving_differences = sum(
        1 for a, b in zip(paddle_layer_sequence, pytorch_layer_sequence) if a != b
    )
    print(f"\n    interleaving differences : {interleaving_differences}  "
          f"(informational — definition order vs execution order)")

    # --- is anything learnable after the grid? ------------------------------
    # The grid-producing layer is the final conv2d PLUS its bias epilogue
    # (conv2d -> full_int_array -> reshape(bias) -> add). That reshape consumes
    # the conv's own bias, so it belongs to the conv, not to anything after it.
    conv_op_indices = [
        i for i, op in enumerate(graph_ops) if op.name() == "pd_op.conv2d"
    ]
    if not conv_op_indices:
        print("      ! no conv2d ops found")
        return False
    final_conv_index = conv_op_indices[-1]

    grid_output_op_index = final_conv_index
    for op_index in range(final_conv_index + 1, len(graph_ops)):
        op_name = graph_ops[op_index].name()
        if op_name == "pd_op.add":
            grid_output_op_index = op_index
            break
        if op_name in ("pd_op.grid_sample", "pd_op.bilinear_interp"):
            break  # bias-less conv: the grid ends at the conv itself

    grid_conv_kernels = [
        paddle_parameters[name]
        for name in parameters_consumed_by(
            graph_ops[final_conv_index], paddle_parameters
        )
        if paddle_parameters[name].ndim == 4
    ]
    ops_after_grid = [
        op for op in graph_ops[grid_output_op_index + 1:]
        if op.name() != "builtin.parameter"
    ]
    learnable_ops_after_grid = [
        op for op in ops_after_grid if parameters_consumed_by(op, paddle_parameters)
    ]

    kernel_shape = grid_conv_kernels[0].shape if grid_conv_kernels else None
    print(f"\n  final learnable op      : pd_op.conv2d, kernel {kernel_shape}"
          f"  -> {kernel_shape[0] if kernel_shape else '?'}-channel grid")
    print(f"  ops after the grid      : {[op.name() for op in ops_after_grid]}")
    print(f"  of which parameter-bearing : {len(learnable_ops_after_grid)}")
    if learnable_ops_after_grid:
        failures.append(f"{len(learnable_ops_after_grid)} learnable ops after the grid")

    if len(model_outputs) != 1:
        failures.append(f"expected 1 model output, found {len(model_outputs)}")
    print(f"  model outputs           : {len(model_outputs)}")
    print(f"  output producer         : {str(model_outputs[0])[:70]}")

    architecture_is_identical = not failures
    verdict = (
        "2D PATH STRUCTURALLY IDENTICAL; nothing learnable after the grid"
        if architecture_is_identical
        else "ARCHITECTURE DIFFERS"
    )
    print(f"\n  >>> {verdict}")
    for failure in failures:
        print(f"      ! {failure}")
    return architecture_is_identical


# ====================================================================== main
def main():
    """Run both comparisons and print a summary. Returns a process exit code."""
    pytorch_tensors = load_pytorch_checkpoint()
    # `program` must stay referenced for the lifetime of `graph_ops`.
    (
        paddle_parameters,
        graph_ops,
        program,  # noqa: F841 — binding keeps the PIR handles alive
        model_outputs,
    ) = load_paddle_inference_model()

    weights_are_identical, name_mapping = compare_weights(
        pytorch_tensors, paddle_parameters
    )
    architecture_is_identical = compare_architecture(
        pytorch_tensors, paddle_parameters, graph_ops, model_outputs, name_mapping
    )

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  WEIGHTS      : "
          f"{'BIT-WISE IDENTICAL' if weights_are_identical else 'DIFFER'}")
    print(f"  ARCHITECTURE : "
          f"{'STRUCTURALLY IDENTICAL' if architecture_is_identical else 'DIFFERS'}")
    print()
    print("  Bit-identical weights mean the Paddle release is a conversion of the")
    print("  original checkpoint, not a retrain: no training run, however short,")
    print("  leaves every stored value unchanged.")
    print("=" * 78)

    return 0 if (weights_are_identical and architecture_is_identical) else 1


if __name__ == "__main__":
    sys.exit(main())
