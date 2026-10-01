# Installing CLARA with Qwen3 (no Llama needed)

A step-by-step guide to installing the **Qwen3 version of CLARA**
([aipapadim/multimodal-chatbot](https://github.com/aipapadim/multimodal-chatbot)).
It's written for people who are new to coding.

## First, the most important point

The Qwen3 version **does not use Llama 3.2 at all**. Llama was only needed
by the *old* CLARA repo. The Meta restriction for the EU doesn't affect you here.

| Repository | What it is | Do you need it? |
|---|---|---|
| [aipapadim/CLARA](https://github.com/aipapadim/CLARA) | Old version (Llama 3.2 + Grounding DINO + Stable Diffusion) | No install needed. Its clarification prompt is reused in Step 8 |
| [aipapadim/llama3.2-api](https://github.com/aipapadim/llama3.2-api) | The Llama dialogue server for the old version | **No** |
| [aipapadim/multimodal-chatbot](https://github.com/aipapadim/multimodal-chatbot) | **New version with Qwen3-VL** | **Yes, only this one** |

The new version uses only models that are open in the EU:

| Model | Job | License |
|---|---|---|
| **Qwen3-VL 8B** (via Ollama) | The "brain": understands your message and the image | Apache 2.0 |
| **Florence-2-base** | Finds where objects are in the image (boxes) | MIT |
| **SAM 2.1** | Cuts the object out precisely | Apache 2.0 |
| **LaMa (big-lama)** | Fills the empty space after removing/moving an object | Apache 2.0 |

### How the pieces talk to each other

```
 You (browser) ──► Gradio UI  :7861   (conda env: qwen3)
                      │
                      ├──► Qwen3 API :5555  (conda env: qwen3)
                      │        ├── Ollama :11434 ── qwen3-vl:8b
                      │        └── Florence-2 (finds object boxes)
                      │
                      └──► Inpaint API :4444 (conda env: inpaint-anything)
                               ├── SAM 2.1 (cuts the object)
                               └── LaMa    (fills the hole)
```

> **About the conda environments you already made:** the old repo used the names
> `clara-llama`, `clara-grounding`, `clara-inpaint`, `clara-diffusion`.
> The new start script looks for **different names**: `qwen3` and
> `inpaint-anything`. Step 3 creates these two for you. You can leave the old
> ones alone, or delete them later to save disk space
> (`conda env remove -n clara-llama`, etc.).

---

## What you need

- **Linux** (Ubuntu works well). On **Windows**, install
  [WSL2 with Ubuntu](https://learn.microsoft.com/windows/wsl/install) and
  do everything inside the Ubuntu terminal.
- An **NVIDIA GPU** with at least ~12 GB of memory. The code is written for
  `cuda:0` and won't run on CPU only.
- **About 25 GB** of free disk space.
- **Miniconda/Anaconda**, which you already have.

Run these commands in a terminal, one block at a time. Lines starting
with `#` are comments; you don't need to type them.

---

## Step 1 — Basic tools

```bash
sudo apt update
sudo apt install -y git wget unzip lsof
nvidia-smi        # should show your GPU. If it errors, install the NVIDIA driver first.
```

## Step 2 — Download the code

```bash
cd ~
git clone https://github.com/aipapadim/multimodal-chatbot.git
git clone https://github.com/dianacerdasv/clara_installation.git
```

You now have two folders in your home directory:
`~/multimodal-chatbot` (CLARA itself) and `~/clara_installation` (this guide and its helper scripts).

## Step 3 — Create the two conda environments (≈ 15–30 min)

```bash
bash ~/clara_installation/scripts/01_create_envs.sh ~/multimodal-chatbot
```

This creates:
- `qwen3`: the Qwen3 API and the chat web page
- `inpaint-anything`: SAM 2.1 + LaMa

> If you have a new **RTX 50xx** card, run this instead:
> ```bash
> TORCH_SPEC="torch==2.7.1 torchvision==0.22.1" TORCH_INDEX=https://download.pytorch.org/whl/cu128 \
>   bash ~/clara_installation/scripts/01_create_envs.sh ~/multimodal-chatbot
> ```

## Step 4 — Install Ollama (this replaces Llama)

Ollama is a small program that runs the Qwen3 model on your GPU.

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version     # must be 0.12.7 or newer (needed for qwen3-vl)
```

On normal Ubuntu, Ollama starts automatically in the background. If a later
step says it can't connect to Ollama, open a **new terminal**, run
`ollama serve`, and leave that terminal open.

### Option: Ollama on another server instead of this PC

If you already have Ollama on a lab or local server, you can use it instead.
Qwen will then run on the server's GPU.

1. On the **server**: download the model. It must be the vision model
   `qwen3-vl:8b`; the plain `qwen3` can't see images.
   ```bash
   ollama pull qwen3-vl:8b
   ```
2. On the **server**: let Ollama accept connections from other machines (by default it only
   listens to itself). Run `sudo systemctl edit ollama`, add these two lines, save, then
   run `sudo systemctl restart ollama`:
   ```
   [Service]
   Environment="OLLAMA_HOST=0.0.0.0:11434"
   ```
   Port **11434** must also be open in the server's firewall.
3. On **your PC**: skip installing Ollama, and point CLARA to the server (replace the IP):
   ```bash
   bash ~/clara_installation/scripts/use_remote_ollama.sh ~/multimodal-chatbot 192.168.1.50
   ```
   It checks the server answers and has `qwen3-vl:8b`, then changes the one line in
   `qwen3_api/ai_engine.py` that holds the Ollama address.
   To go back to a local Ollama: `... use_remote_ollama.sh ~/multimodal-chatbot local`.

Keep in mind:
- Your PC **still needs its NVIDIA GPU**. Florence-2, SAM 2.1 and LaMa always run
  locally, only Qwen moves to the server (that frees about 6 GB on your GPU).
- Your images are sent to the server, so use a server you trust (your lab / local network).
- Step 5 will say "skipping qwen3-vl:8b". That's expected, and the other models still download.

## Step 5 — Download the models (≈ 10 GB)

```bash
bash ~/clara_installation/scripts/02_download_models.sh ~/multimodal-chatbot
```

This downloads:
1. `qwen3-vl:8b` with Ollama (no account or approval needed)
2. `Florence-2-base` from Hugging Face (no login needed)
3. `sam2.1_hiera_base_plus.pt` and `sam2.1_hiera_large.pt`
4. `big-lama` and places it where the code expects it:
   ```
   multimodal-chatbot/Inpaint-Anything/pretrained_models/
     sam2.1_hiera_base_plus.pt
     sam2.1_hiera_large.pt
     big-lama/config.yaml
     big-lama/best.ckpt
     big-lama/models/best.ckpt
   ```

> If the big-lama download fails, download the `big-lama` folder manually from
> [this Google Drive link](https://drive.google.com/drive/folders/1B2x7eQDgecTL0oh3LSIBDGj0fTxs6Ips).
> Put `config.yaml` in `big-lama/` and `best.ckpt` in **both** `big-lama/` and `big-lama/models/`.

## Step 6 — Check everything

```bash
bash ~/clara_installation/scripts/03_check_install.sh ~/multimodal-chatbot
```

Every line should say `[OK]`. If any line says `[MISSING]`, the message
tells you which step to repeat.

## Step 7 — Start CLARA

```bash
cd ~/multimodal-chatbot
bash start_pillar_chatbot.sh
```

Wait about 1–2 minutes the first time (the models load into the GPU). Then open
**http://127.0.0.1:7861** in your browser.

How to use it:
- Upload an image of a scene.
- **Remove** an object: *"remove the red cup"*
- **Move** an object: *"move the apple to the right of the bowl"*
- Anything else is normal chat about the image: *"what objects are on the table?"*

To stop CLARA, press `Ctrl + C` in the terminal.

> Note: the UI starts with `share=True`, which also creates a temporary public
> `gradio.live` link. If you don't want that, open
> `multimodal-chatbot/core/src/multimodal_chatbot_ui.py` and change
> `share=True` to `share=False`.

---

## Step 8 (optional) — The second version: CLARA with clarification questions

The authors' Qwen3 interface (Step 7) doesn't ask clarification questions.
It looks for keywords ("remove", "move", "put"...) and does one edit at a time.
Other messages go to Qwen without CLARA's prompt and without the earlier
conversation.

This repo can build a **second copy** that restores the CLARA behaviour, using
the purpose-clarification prompt from the original
[CLARA repo](https://github.com/aipapadim/CLARA):

```bash
bash ~/clara_installation/scripts/04_make_clara_version.sh ~/multimodal-chatbot
```

This creates `~/multimodal-chatbot-clara`. Your first folder is not changed,
and both versions share the same conda environments and model files, so the
copy takes only a few MB.

| | `~/multimodal-chatbot` | `~/multimodal-chatbot-clara` |
|---|---|---|
| How it reacts | Keywords → one remove / move | CLARA dialogue: asks questions, one at a time |
| System prompt | Not sent to Qwen | CLARA prompt (editable under **Additional Inputs**) |
| Remembers the conversation | No | Yes: all messages and images |
| Final result | One edited image per command | When Qwen is sure, it outputs `image-gen, result = {"container": ["object", ...]}` and each object is moved into its container |
| Temperature / Top-P / Tokens sliders | Ignored | Used |

Start it the same way:

```bash
cd ~/multimodal-chatbot-clara
bash start_pillar_chatbot.sh
```

Then open **http://127.0.0.1:7861**, upload a scene and write the goal,
for example *"organize the objects"*. Answer the questions until it
generates the new image.

Good to know:
- **Only one version can run at a time** (they use the same ports). Starting
  one automatically stops the other.
- The final step reuses the authors' "move" function: each object is cut out
  and pasted on the **center** of its container, then the next one, on the same
  image. Florence-2 finds **one** box per name, so with three identical cups only
  one is moved. Objects it can't find are listed as "skipped" in the reply.
- The changed files are in `clara_version/` in this repo
  (`qwen3_api/ai_engine.py` and `core/src/multimodal_chatbot_ui.py`), if you want
  to read or tweak them.
- If you set a remote Ollama server, the copy uses the same server automatically.
- To rebuild the copy (e.g. after changing those files):
  `rm -rf ~/multimodal-chatbot-clara` and run the script again.

---

## Common problems

| Message | What to do |
|---|---|
| `cannot import name 'Sentinel' from 'typing_extensions' (/usr/lib/python3/dist-packages/...)` | Your `PYTHONPATH` points conda at the system's Python packages. Run `unset PYTHONPATH`, then delete the `export PYTHONPATH=...` line from `~/.bashrc` and open a new terminal. Check with `echo $PYTHONPATH` (it should print an empty line). |
| `CondaToSNonInteractiveError: Terms of Service have not been accepted` | Accept them once: `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main` and the same with `.../pkgs/r`. Then re-run the script. |
| `Weights only load failed ... pytorch_lightning ... ModelCheckpoint` (when moving/removing objects) | PyTorch 2.6+ blocks the LaMa checkpoint. Run `sed -i 's/torch.load(path, map_location=map_location)$/torch.load(path, map_location=map_location, weights_only=False)/' ~/multimodal-chatbot/Inpaint-Anything/lama/saicinpainting/training/trainers/__init__.py` (and the same in `~/multimodal-chatbot-clara` if it is a separate copy), then restart CLARA. Step 1 now does this for you. |
| `conda: command not found` | Close and reopen the terminal, or run `source ~/miniconda3/etc/profile.d/conda.sh`. |
| `Could not connect to ollama` / `Connection refused ... 11434` | Local Ollama: run `ollama serve` in a separate terminal and leave it open. Remote server: check it's on, listens on `0.0.0.0`, and port 11434 is open. |
| `model "qwen3-vl:8b" not found` | `ollama pull qwen3-vl:8b` |
| Ollama says the model needs a newer version | Re-run the Ollama install command from Step 4 to update it. |
| `CUDA out of memory` | Close other programs that use the GPU (games, other notebooks). Check with `nvidia-smi`. |
| `torch.cuda.is_available()` is `False` | Your NVIDIA driver is too old for the PyTorch build. Update the driver, or see the RTX 50xx note in Step 3. |
| `No such file or directory: ... pretrained_models/...` | A model file is missing. Re-run Step 5, then Step 6. |
| `You're likely running Python from the parent directory of the sam2 repository` | SAM2 was installed with `pip install -e`. Fix it with: `conda activate inpaint-anything && pip uninstall -y SAM-2 && SAM2_BUILD_CUDA=0 pip install --no-build-isolation ~/multimodal-chatbot/Inpaint-Anything/sam2` |
| `No module named 'pkg_resources'` | `conda activate inpaint-anything && pip install "setuptools<81"` |
| Port already in use (4444 / 5555 / 7861) | The start script normally clears these. If not, reboot or run `lsof -i :5555` to find the process. |

### Running each part by hand (for debugging)

If something fails, start each server in its own terminal to see the error clearly:

```bash
# Terminal 1 – Qwen3 API
conda activate qwen3 && cd ~/multimodal-chatbot/qwen3_api && python app.py

# Terminal 2 – Inpaint API
conda activate inpaint-anything && cd ~/multimodal-chatbot/Inpaint-Anything && python app.py

# Terminal 3 – Web UI
conda activate qwen3 && cd ~/multimodal-chatbot/core/src && python multimodal_chatbot_ui.py
```

---

### What was tested

Both environments were installed from these exact commands and checked on a
CPU-only Linux machine with Python 3.12. All of CLARA's modules imported
correctly, and each server stopped only where it needed the model files or
the GPU. The CLARA version (Step 8) was also tested in a browser with fake
model servers standing in for Qwen, Florence-2 and the inpainting service:
a 3-turn dialogue with history, the image-gen → moves step, and the next turn
using the new image all worked. Neither version has been run end-to-end on a
real GPU with the real models yet, so if a step fails on
your PC, the error message plus the "Common problems" table above is the
place to start.
