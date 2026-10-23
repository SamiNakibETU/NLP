# %% [markdown]
# # Test de bout en bout de l'appli Kalam sur un vrai GPU (T4)
# Lance speech_app/verifier_reel.py : direct, serveur (WebSocket), rapport (pyannote + Qwen2.5-7B nf4), lecture vocale,
# dépôt de fichier (S3, Seamless, S0), LoRA. Le jeton HF vient du secret Kaggle HF_TOKEN (jamais affiché ni écrit).

# %%
import subprocess, sys, os, json, time, shutil, re
ARGS = ["--etapes", "donnees,ecapa,depot,rapport,perf,tts", "--moteurs", "s0"]   # modifié à chaque essai
def pip(*a):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *a], capture_output=True, text=True)
    print("pip", " ".join(a)[:110], "->", r.returncode)
    if r.returncode: print(r.stderr[-1800:])
    return r.returncode
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
if not shutil.which("ffmpeg"):
    subprocess.run("apt-get -qq install -y ffmpeg", shell=True)
print("ffmpeg :", shutil.which("ffmpeg"))
pip("-U", "transformers>=5.4", "peft", "accelerate", "bitsandbytes", "sentencepiece", "protobuf", "librosa", "soundfile",
    "pyyaml", "sacremoses", "brotli>=1.2", "numpy<2.3")
pip("fastapi", "uvicorn[standard]", "python-multipart", "gradio>=6.0", "yt-dlp", "requests", "websockets", "trafilatura",
    "qrcode[pil]", "python-docx")
pip("speechbrain>=1.0", "piper-tts>=1.2", "edge-tts>=6.1")
pip("pyannote.audio>=4.0", "numpy<2.3")
r = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
print("pip check :", (r.stdout + r.stderr).strip()[-2500:] or "aucun conflit")
PIP_CHECK = (r.stdout + r.stderr).strip()
r = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
GEL = [l for l in r.stdout.splitlines() if re.match(r"(torch|torchaudio|torchvision|torchcodec|transformers|peft|accelerate|bitsandbytes|pyannote|speechbrain|piper|edge|fastapi|uvicorn|websockets|gradio|huggingface|numpy|librosa|soundfile|tokenizers|safetensors|lightning|onnx)", l, re.I)]
print("\n".join(GEL))
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

# %%
# Jeton Hugging Face : secret Kaggle uniquement, jamais affiché
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN présent")
except Exception as exc:
    os.environ.pop("HF_TOKEN", None)
    print("pas de secret HF_TOKEN :", type(exc).__name__)
if os.environ.get("HF_TOKEN"):
    from huggingface_hub import HfApi
    for depot in ("CohereLabs/cohere-transcribe-arabic-07-2026", "pyannote/speaker-diarization-community-1", "pyannote/segmentation-3.0"):
        try:
            HfApi().model_info(depot, token=os.environ["HF_TOKEN"]); print("accès ok :", depot)
        except Exception as exc:
            print("accès refusé :", depot, type(exc).__name__)

# %%
from pathlib import Path
DEPOT = Path("/kaggle/working/depot")
if DEPOT.exists(): shutil.rmtree(DEPOT)
subprocess.run(["git", "clone", "-q", "-b", "integration", "https://github.com/SamiNakibETU/NLP", str(DEPOT)], check=True)
import base64, io, zipfile
zipfile.ZipFile(io.BytesIO(base64.b64decode(PATCH_B64))).extractall(DEPOT)      # surcouche posée sur le clone
patch = "surcouche embarquée"
print("code : clone integration + surcouche", patch, "; projecteur S3 :", (DEPOT / "modeles/s3_temps1_50k/projecteur_meilleur.pt").is_file())
SORTIE = Path("/kaggle/working")
cmd = [sys.executable, "-u", "verifier_reel.py", "--sortie", str(SORTIE / "verification_reelle.json"),
       "--travail", "/kaggle/working/verif", *ARGS]
env = dict(os.environ, PYTHONIOENCODING="utf-8")
t0 = time.time()
proc = subprocess.Popen(cmd, cwd=str(DEPOT / "speech_app"), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for ligne in proc.stdout:
    print(ligne.rstrip())
proc.wait()
print("code de sortie", proc.returncode, "en", round(time.time() - t0), "s")

# %%
# Artefacts à relire : journal du serveur, fichiers de rapport, gel des versions
out = SORTIE / "artefacts"; out.mkdir(exist_ok=True)
(out / "pip_check.txt").write_text(PIP_CHECK, encoding="utf-8")
(out / "versions.txt").write_text("\n".join(GEL), encoding="utf-8")
tr = Path("/kaggle/working/verif")
if (tr / "serveur.log").is_file():
    tok = os.environ.get("HF_TOKEN") or "\0"
    (out / "serveur.log").write_text((tr / "serveur.log").read_text(errors="replace").replace(tok, "***"), encoding="utf-8")
for d in (tr / "sessions").glob("*/rapport"):
    shutil.copytree(d, out / ("rapport-" + d.parent.name), dirs_exist_ok=True)
shutil.rmtree(tr / "audio", ignore_errors=True)
for f in ("test.tar.gz", "ar_eg.tsv"):
    (tr / f).unlink(missing_ok=True)
shutil.rmtree(DEPOT, ignore_errors=True)
shutil.rmtree(tr / "sessions", ignore_errors=True)
print(sorted(p.name for p in out.iterdir()))
