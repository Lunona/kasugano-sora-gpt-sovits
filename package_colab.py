from pathlib import Path
import json, zipfile
import torch

root = Path(__file__).resolve().parent
data = root / "colab_data"
bundle = root / "colab_bundle"
prep = root / "prep"


def cell(kind, text):
    value = dict(cell_type=kind, metadata={}, source=text.splitlines(keepends=True))
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


CELL_1 = """from google.colab import drive
from pathlib import Path
import hashlib, os, subprocess, sys, urllib.request, zipfile

drive.mount('/content/drive')
drive_root = Path('/content/drive/MyDrive')
sora_dir = drive_root / '春日野穹_GPTSoVITS_训练'
bundle = sora_dir / 'sora_condition_colab.zip'
assert bundle.is_file(), f'找不到 {bundle}'
job = Path('/content/sora_condition')
job.mkdir(exist_ok=True)
with zipfile.ZipFile(bundle) as archive:
    archive.extractall(job)

repo = job / 'GPT-SoVITS'
if not repo.exists():
    subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout',
                    'https://github.com/RVC-Boss/GPT-SoVITS.git', str(repo)], check=True)
subprocess.run(['git', '-C', str(repo), 'checkout', '--detach',
                '1b47ca07ef43a284e0b8bd7f29d3e1e3ff76082a'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', 'torchmetrics==1.5.0', 'tqdm', 'x-transformers'], check=True)
subprocess.run(['nvidia-smi'], check=True)

HF_BASE = 'https://huggingface.co/lj1995/GPT-SoVITS/resolve/main'
WEIGHTS = {
    's1v3.ckpt': (
        f'{HF_BASE}/s1v3.ckpt',
        '87133414860ea14ff6620c483a3db5ed07b44be42e2c3fcdad65523a729a745a',
    ),
    's2Gv2ProPlus.pth': (f'{HF_BASE}/v2Pro/s2Gv2ProPlus.pth', None),
    's2Dv2ProPlus.pth': (f'{HF_BASE}/v2Pro/s2Dv2ProPlus.pth', None),
}

def sha256_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

for fname, (url, expected_sha) in WEIGHTS.items():
    target = job / fname
    if not target.is_file() or (expected_sha and sha256_file(target) != expected_sha):
        print(f'正在下载预训练底座 {fname} ...', flush=True)
        pending = target.with_suffix('.pending')
        urllib.request.urlretrieve(url, pending)
        if expected_sha:
            assert sha256_file(pending) == expected_sha, f'{fname} SHA256 校验失败'
        pending.replace(target)
print('环境与全部 2,575 条语音训练数据准备完毕。', flush=True)
"""

CELL_2 = """env = dict(os.environ, SORAVOICE_APP=str(repo), PYTHONPATH=f"{repo / 'GPT_SoVITS'}:{repo}")
out_dir = sora_dir / 'sora_condition_results_v2proplus'

# 1. 训练 GPT（s1v3 底座 + 句内情绪条件层 Adapter + GPT 语义预测主干联合微调）
gpt_cmd = [sys.executable, '-u', str(job / 'train_condition.py'),
           '--data', str(job / 'training-data.pt'),
           '--base', str(job / 's1v3.ckpt'),
           '--batch-size', '8',
           '--out', str(out_dir)]
subprocess.run(gpt_cmd + ['--smoke'], env=env, cwd=job, check=True)
subprocess.run(gpt_cmd + ['--epochs', '15'], env=env, cwd=job, check=True)

# 2. 训练 SoVITS v2ProPlus（2,575 条春日野穹真实语音 + HuBERT + SV声纹 微调）
sovits_cmd = [sys.executable, '-u', str(job / 'train_sovits.py'),
              '--data', str(job / 'training-data.pt'),
              '--prep', str(job / 'prep'),
              '--base-g', str(job / 's2Gv2ProPlus.pth'),
              '--base-d', str(job / 's2Dv2ProPlus.pth'),
              '--batch-size', '8',
              '--out', str(out_dir)]
subprocess.run(sovits_cmd + ['--smoke'], env=env, cwd=job, check=True)
subprocess.run(sovits_cmd + ['--epochs', '8'], env=env, cwd=job, check=True)

# 3. 校验产物并同步释放 Colab GPU
import io, sys, torch
from google.colab import runtime

if str(repo / 'GPT_SoVITS') not in sys.path:
    sys.path.insert(0, str(repo / 'GPT_SoVITS'))

weights = torch.load(out_dir / 'phrase-adapter-epoch15.pt', map_location='cpu', weights_only=True)
gpt_ckpt = torch.load(out_dir / 'sora-gpt-v2-epoch15.ckpt', map_location='cpu', weights_only=True)
sovits_bytes = (out_dir / 'sora-sovits-latest.pth').read_bytes()
if sovits_bytes[:2] == b'06':
    sovits_bytes = b'PK' + sovits_bytes[2:]
sovits_ckpt = torch.load(io.BytesIO(sovits_bytes), map_location='cpu', weights_only=False)
assert weights.get('adapter') and weights.get('gpt_weight'), 'GPT+Adapter 权重缺失'
assert gpt_ckpt.get('weight') and sovits_ckpt.get('weight'), 'GPT 或 SoVITS 导出权重缺失'
print('v2ProPlus 全套权重（s1v3+Adapter 与 SoVITS v2ProPlus）已全部验证通过，正在同步 Drive。', flush=True)
drive.flush_and_unmount()
print('Drive 已同步，正在释放 GPU 运行时。', flush=True)
runtime.unassign()
"""

cells = [
    cell(
        "markdown",
        "# 春日野穹 GPT‑SoVITS 全量微调（GPT句内情绪条件层 + GPT预测主干 + SoVITS 2,575条语音音色）\n\n"
        "把 `sora_condition_colab.zip` 放到 Google Drive 的 `MyDrive` 根目录。选 GPU，依次运行下面两个单元。支持断点续训。",
    ),
    cell("code", CELL_1),
    cell("code", CELL_2),
]

nb = dict(
    cells=cells,
    metadata={
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    },
    nbformat=4,
    nbformat_minor=4,
)

for nb_name in ("穹_条件层训练.ipynb", "穹_条件层训练_云盘版.ipynb"):
    (root / nb_name).write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")

raw = torch.load(data / "training-data.pt", map_location="cpu", weights_only=True)
train_ids = [uid for uid, split, _, _, _ in raw if split == "train"]
assert len(train_ids) == 2575

files = [
    (data / "training-data.pt", "training-data.pt"),
    (bundle / "train_condition.py", "train_condition.py"),
    (bundle / "train_sovits.py", "train_sovits.py"),
    (bundle / "sovits_emotion.py", "sovits_emotion.py"),
    (bundle / "vocal_tokens.py", "vocal_tokens.py"),
    (bundle / "performance.py", "performance.py"),
    (bundle / "phrase-overrides.json", "phrase-overrides.json"),
]
for uid in train_ids:
    files.append((prep / "4-cnhubert" / f"{uid}.wav.pt", f"prep/4-cnhubert/{uid}.wav.pt"))
    files.append((prep / "5-wav32k" / f"{uid}.wav", f"prep/5-wav32k/{uid}.wav"))
    files.append((prep / "7-sv_cn" / f"{uid}.wav.pt", f"prep/7-sv_cn/{uid}.wav.pt"))

zip_path = root / "sora_condition_colab.zip"
print(f"Packing {len(files)} files into {zip_path} ...", flush=True)
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
    for file_p, arcname in files:
        archive.write(file_p, arcname)

for c in cells:
    if c["cell_type"] == "code":
        compile("".join(c["source"]), "<notebook>", "exec")

print(f"Packed {zip_path.name}: {zip_path.stat().st_size / (1024*1024):.1f} MB ({len(files)} files)", flush=True)
