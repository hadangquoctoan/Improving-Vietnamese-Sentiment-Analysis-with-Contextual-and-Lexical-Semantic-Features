"""Inspection output for preprocessing, lexicon profile, and PhoBERT hidden states."""

from vnsentiment.preprocessing.collapse import collapse_elongations, normalize_basic
from vnsentiment.preprocessing.pipeline import PreprocessingPipeline, RawSample
from vnsentiment.preprocessing.variants import VariantMatcher


def show_sample(
    text: str,
    matcher: VariantMatcher,
    pipeline: PreprocessingPipeline,
    phobert_branch=None,
    lexicon_cli=None,
) -> None:
    basic = normalize_basic(text)
    collapsed = collapse_elongations(basic)
    mapped = matcher.replace(collapsed)
    result = pipeline.process(RawSample("cli", text))
    print(f"  Raw       : {text}")
    print(f"  NFC/lower : {basic}")
    print(f"  Collapse  : {collapsed}")
    print(f"  Biến thể  : {mapped}")
    print(f"  Segmented : {result.normalized_text}")
    if mapped == collapsed:
        print("  (không có biến thể nào được thay)")
    if lexicon_cli is not None:
        lexicon_cli.print_report(result.normalized_text)
    if phobert_branch is not None:
        import torch

        with torch.no_grad():
            batch, hidden = phobert_branch.encode([result.normalized_text])
        _print_hidden(phobert_branch, batch, hidden)


def _print_hidden(phobert_branch, batch, hidden) -> None:
    content = batch.content_mask[0].bool()
    tokens = phobert_branch.tokenizer.convert_ids_to_tokens(batch.input_ids[0].tolist())
    norms = hidden[0].norm(dim=-1)
    print(f"  H          : shape={tuple(hidden.shape)} | device={hidden.device}")
    print("  token norms:")
    for token, is_content, norm in zip(tokens, content.tolist(), norms.tolist()):
        if is_content:
            print(f"    {token:<16} {norm:.4f}")
    if not bool(content.any()):
        print("  pooled     : (không có content token)")
        return
    pooled = hidden[0][content.to(hidden.device)].mean(dim=0)
    first8 = ", ".join(f"{value:.4f}" for value in pooled[:8].tolist())
    print(
        f"  pooled     : norm={pooled.norm().item():.4f} "
        f"mean={pooled.mean().item():+.4f} first8=[{first8}]"
    )
    print("  (hidden states PhoBERT; classifier tạo logits [B,3] hoặc [B,2] tùy checkpoint)")