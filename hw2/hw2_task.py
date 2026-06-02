import torch
from utils import (
    PROFILE_STEPS,
    RESULTS_DIR,
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
)


@torch.inference_mode()
def optimized_loop(model, input_ids, n_steps):
    past_key_values = None
    cur_ids = input_ids
    token_buf = torch.empty(n_steps, dtype=torch.long, device=input_ids.device)

    for step in range(n_steps):
        outputs = model(
            input_ids=cur_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        token_buf[step] = next_token.squeeze(-1)
        cur_ids = next_token

    return token_buf.tolist()


def profile(loop_fn, model, input_ids, trace_name: str):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        loop_fn(model, input_ids, PROFILE_STEPS)

    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=25,
        )
    )
    trace_path = RESULTS_DIR / trace_name
    prof.export_chrome_trace(str(trace_path))
    print(f"Chrome trace saved to {trace_path}")


def generate_optimized(optimized_trace_name: str) -> float:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = build_model(torch.bfloat16)
    input_ids = get_input_ids()

    # Warm up (compilation / caches) before profiling and timing.
    optimized_loop(model, input_ids, 4)
    torch.cuda.synchronize()

    profile(optimized_loop, model, input_ids, optimized_trace_name)
    elapsed = time_generation(optimized_loop, model, input_ids, "Optimized")

    del model
    torch.cuda.empty_cache()
    return elapsed


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(optimized_trace_name="v1_optimized_trace.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Changes made and speedup per fix:
# 1. KV cache (use_cache=True, pass only the last token after the prompt):
#    avoids re-computing attention over the full growing sequence each step.
# 2. bfloat16 model weights/activations in generate_optimized().
# 3. Removed per-step .item() and torch.cat — keep tokens on GPU in a
#    preallocated buffer; one .tolist() at the end instead of 128 syncs.
# 4. torch.inference_mode() around the decode loop.
# 5. TF32 matmul enabled for Ampere/Hopper-class GPUs.
#
# Biggest impact and why:
# KV cache — V0 reruns full-sequence attention every step (cost grows with
# sequence length); with cache each step is roughly constant-time in sequence
# length for the new token only.
