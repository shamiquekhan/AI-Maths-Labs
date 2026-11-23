"""MiniGPT laboratory page."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st  # noqa: E402

from from_zero.app import common  # noqa: E402
from from_zero.app.services import TORCH_AVAILABLE, generate_gpt, run_gpt_demo  # noqa: E402

common.setup_page("MiniGPT", "◆")
common.sidebar()

common.hero(
    "A decoder-only transformer you can inspect.",
    "Character embeddings enter four pre-layer-norm blocks. Causal self-attention mixes "
    "only past positions, and every position predicts the next token. The result is a "
    "small Shakespeare model, not an assistant.",
    ("PyTorch", "Causal attention", "Top-k sampling", "Checkpoint reload"),
)

if not TORCH_AVAILABLE:
    common.callout(
        "This laboratory needs PyTorch.",
        "The public dashboard runs the strict from-scratch engine and the NumPy "
        "laboratories without PyTorch. Install the deep-learning extra to train or "
        "sample the MiniGPT model locally.",
    )
    st.code("pip install -e '.[deep-learning]'", language="bash")
    st.markdown(
        "The strict character-level GPT on the custom tensor engine is available on the "
        "**Strict Engine** page and needs no third-party numerical library."
    )
    st.stop()

learn, build, visualise, evidence = st.tabs(["Learn", "Build", "Visualise", "Evidence"])

with learn:
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("### Next-token prediction")
        st.markdown(
            "Given a sequence of tokens $x_{1:T}$, the model maximizes the log-likelihood "
            "of each next token. The training target is the input shifted one position to the left."
        )
        st.latex(r"\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_{t+1}\mid x_{1:t})")
        st.markdown("### Causal self-attention")
        st.latex(
            r"Q = XW_Q,\quad K = XW_K,\quad V = XW_V,\quad "
            r"\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V"
        )
        st.markdown(
            "The mask $M_{ij}=0$ for $j\\le i$ and $-\\infty$ for $j>i$. Without it, "
            "position $t$ could copy the answer from position $t+1$. The test suite changes "
            "a future token and checks that earlier logits do not move."
        )
        st.markdown("### The block")
        st.latex(r"x \leftarrow x + \mathrm{Attention}(\mathrm{LN}(x))")
        st.latex(r"x \leftarrow x + \mathrm{FFN}(\mathrm{LN}(x))")
        st.markdown(
            "The feed-forward network expands by 4×, applies GELU, then projects back. "
            "Residual connections and layer normalization keep the optimization stable."
        )
    with right:
        st.markdown("### Sampling")
        st.markdown(
            "Generation is autoregressive. At each step the model produces logits for the "
            "next token; temperature divides them, top-k keeps the $k$ largest, and softmax "
            "turns the rest into a categorical distribution."
        )
        st.latex(r"p_i = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}")
        common.callout(
            "A low validation loss is not a personality.",
            "This model is trained on 1.1 MB of public-domain plays with a character-level "
            "vocabulary. It can imitate surface statistics of Shakespeare, but it has no "
            "factual grounding and cannot follow instructions.",
            tone="warn",
        )
        st.markdown("### Architecture in the saved run")
        metrics = common.read_json("artifacts/experiments/mini_gpt_shakespeare/metrics.json")
        model = metrics.get("model", {})
        if model:
            st.dataframe(
                {
                    "setting": [
                        "parameters",
                        "context length",
                        "embedding dim",
                        "heads",
                        "layers",
                        "vocabulary",
                        "tokenizer",
                    ],
                    "value": [
                        str(model.get("parameters")),
                        str(model.get("context_length")),
                        str(model.get("embedding_dim")),
                        str(model.get("num_heads")),
                        str(model.get("num_layers")),
                        str(model.get("vocab_size")),
                        "character",
                    ],
                },
                hide_index=True,
                width="stretch",
            )

with build:
    st.markdown("### Generate from a checkpoint")
    checkpoints = common.checkpoint_options("pytorch-mini-gpt")
    if not checkpoints:
        st.info(
            "No MiniGPT checkpoint was found. Train the saved model with "
            "`python scripts/train_gpt.py`, or run a short dashboard demo below."
        )
    else:
        with st.form("gpt_generate_form"):
            checkpoint = st.selectbox("Checkpoint", checkpoints, index=0)
            prompt = st.text_area(
                "Prompt",
                value="ROMEO:\n",
                height=90,
                help="Characters outside the tokenizer vocabulary are filtered out.",
            )
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                max_new_tokens = st.slider("New tokens", 20, 500, 200, step=10)
            with col2:
                temperature = st.slider("Temperature", 0.1, 1.6, 0.9, step=0.05)
            with col3:
                top_k = st.slider("Top-k", 1, 65, 40)
            with col4:
                seed = st.number_input("Seed", min_value=0, max_value=10_000, value=42)
            greedy = st.checkbox("Greedy decoding (ignore temperature and top-k)", value=False)
            generate = st.form_submit_button("Generate text", type="primary")

        if generate:
            with st.spinner("Loading the checkpoint and sampling…"):
                try:
                    st.session_state["gpt_text"] = generate_gpt(
                        checkpoint,
                        prompt=prompt,
                        max_new_tokens=int(max_new_tokens),
                        temperature=float(temperature),
                        top_k=int(top_k),
                        seed=int(seed),
                        greedy=bool(greedy),
                    )
                except Exception as error:
                    st.error(f"Generation failed: {error}")

        if st.session_state.get("gpt_text"):
            st.markdown("#### Generated text")
            st.text(st.session_state["gpt_text"])

    st.divider()
    st.markdown("### Short dashboard training run")
    st.caption(
        "This is intentionally tiny: 64-token context, 64-dimensional embeddings, two layers. "
        "It demonstrates the pipeline; it will not produce coherent text."
    )
    with st.form("gpt_train_form"):
        col1, col2 = st.columns(2)
        with col1:
            max_steps = st.slider("Optimizer steps", 10, 300, 40, step=10)
        with col2:
            seed = st.number_input("Training seed", min_value=0, max_value=10_000, value=42)
        train = st.form_submit_button("Run demo training", type="primary")

    if train:
        with st.spinner("Training a tiny model on Tiny Shakespeare…"):
            try:
                st.session_state["gpt_run"] = run_gpt_demo(max_steps=int(max_steps), seed=int(seed))
            except Exception as error:
                st.error(f"Demo training failed: {error}")

    run = st.session_state.get("gpt_run")
    if run is not None:
        st.success(f"Finished in {run.result.directory}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Best validation loss", f"{run.best_val_loss:.3f}")
        col2.metric("Parameters", f"{run.metrics['model']['parameters']:,}")
        col3.metric("Steps", str(run.config.max_steps))
        st.text(run.metrics.get("final_sample", "")[:1200])

with visualise:
    st.markdown("### Evidence from the saved 3,000-step run")
    common.figure(
        "artifacts/experiments/mini_gpt_shakespeare/plots/loss_curve.png",
        "Training and validation loss at evaluation points",
    )
    col1, col2 = st.columns(2)
    with col1:
        common.figure(
            "artifacts/experiments/mini_gpt_shakespeare/plots/attention.png",
            "Layer 1, head 1 attention weights",
        )
    with col2:
        st.markdown("#### Final sample")
        sample = common.load_text(
            "artifacts/experiments/mini_gpt_shakespeare/samples/final_sample.txt"
        )
        st.text(sample or "Run `python scripts/train_gpt.py` to produce a sample.")
    st.markdown("### Training history")
    history = common.read_json("artifacts/experiments/mini_gpt_shakespeare/metrics.json")
    samples = history.get("samples", [])
    if samples:
        st.dataframe(
            [
                {
                    "step": row.get("step"),
                    "validation loss": round(float(row.get("val_loss", float("nan"))), 4),
                    "sample preview": row.get("text", "")[:120].replace("\n", " "),
                }
                for row in samples
            ],
            hide_index=True,
            width="stretch",
        )

with evidence:
    st.markdown("### Saved MiniGPT run")
    st.write(
        "The checkpoint stores model weights, optimizer state, the architecture config, "
        "and the character vocabulary. Loading uses `torch.load(..., weights_only=True)`."
    )
    metrics = common.read_json("artifacts/experiments/mini_gpt_shakespeare/metrics.json")
    if metrics:
        st.json(
            {
                "dataset": metrics.get("dataset"),
                "model": metrics.get("model"),
                "training": metrics.get("training"),
                "checkpoints": metrics.get("checkpoints"),
            },
            expanded=False,
        )
    else:
        st.info("Run `python scripts/train_gpt.py` to create the saved model.")
    notes = common.load_text("artifacts/experiments/mini_gpt_shakespeare/notes.md")
    if notes:
        st.markdown("#### Notes")
        st.markdown(notes)

common.footer()
