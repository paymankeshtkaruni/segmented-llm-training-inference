"""Temp: decompose validation token accuracy to explain why it's high.

For the trained checkpoint, over the FULL validation set (teacher forced):
  * overall token accuracy (should reproduce metrics.json ~0.9988)
  * accuracy on TEMPLATE positions  (fixed scaffold: ' Issue', ':', ',', ' level', eos)
  * accuracy on CONTENT positions   (the actual issue/level words)
  * teacher-forced EXACT-MATCH (all target positions of a label correct)
  * what fraction of supervised tokens are template vs content
"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "final_paper_scripts_results/full_model/scripts")
import torch
from torch.utils.data import DataLoader
import _common as C

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = C.build_tokenizer()
model = C.build_model(tok.vocab_size).to(device)
ck = torch.load("final_paper_scripts_results/full_model/outputs/train_gpu/checkpoints/best.pt",
                map_location=device, weights_only=False)
model.load_state_dict(ck["model_state_dict"]); model.eval()

# token ids that make up the FIXED template scaffold
TEMPLATE_STRS = [" Issue", ":", ",", " level", "<|endoftext|>"]
template_ids = set()
for s in TEMPLATE_STRS:
    ids = tok.encode(s, add_special_tokens=False)
    if len(ids) == 1:
        template_ids.add(ids[0])
template_ids.add(tok.eos_token_id)
print("template token ids:", sorted(template_ids))

ds = C.build_dataset(C.VAL_CSV)
loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=C.build_collator(tok))

tot_c = tot_v = tmpl_c = tmpl_v = cont_c = cont_v = 0
exact_ok = exact_n = 0
with torch.no_grad():
    for batch in loader:
        ii = batch["input_ids"].to(device); lab = batch["labels"].to(device)
        logits, _ = model(ii, pad_token_id=tok.pad_token_id)
        sl = logits[:, :-1, :]; stab = lab[:, 1:]
        preds = sl.argmax(-1)
        mask = stab != -100
        correct = (preds == stab) & mask
        # overall
        tot_c += int(correct.sum()); tot_v += int(mask.sum())
        # template vs content
        is_tmpl = torch.zeros_like(stab, dtype=torch.bool)
        for tid in template_ids:
            is_tmpl |= (stab == tid)
        tmpl_mask = mask & is_tmpl
        cont_mask = mask & (~is_tmpl)
        tmpl_c += int((correct & tmpl_mask).sum()); tmpl_v += int(tmpl_mask.sum())
        cont_c += int((correct & cont_mask).sum()); cont_v += int(cont_mask.sum())
        # exact match per row (all supervised positions correct)
        per_row_valid = mask.sum(1)
        per_row_correct = correct.sum(1)
        exact_ok += int((per_row_correct == per_row_valid).sum())
        exact_n += ii.size(0)

print(f"\nOVERALL token acc : {tot_c}/{tot_v} = {tot_c/tot_v:.4f}")
print(f"TEMPLATE token acc: {tmpl_c}/{tmpl_v} = {tmpl_c/tmpl_v:.4f}  "
      f"({tmpl_v/tot_v:.1%} of supervised tokens)")
print(f"CONTENT  token acc: {cont_c}/{cont_v} = {cont_c/cont_v:.4f}  "
      f"({cont_v/tot_v:.1%} of supervised tokens)")
print(f"EXACT-MATCH (teacher-forced, whole label): {exact_ok}/{exact_n} = {exact_ok/exact_n:.4f}")
