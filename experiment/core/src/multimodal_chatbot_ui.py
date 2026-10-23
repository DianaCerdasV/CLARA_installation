import os
import re
import ast
import json
import requests
import gradio as gr
import threading 
import time
import shutil
import subprocess



# --- CONFIGURATION ---
QWEN_API_URL = "http://127.0.0.1:5555/api/relations"
INPAINT_API_URL = "http://127.0.0.1:4444/api/image-gen"

# CLARA purpose-clarification prompt (from the original CLARA, aipapadim/CLARA).
# {phrase} is replaced with the user's first request of the conversation.
CLARA_PROMPT = """You are an AI assistant and you must follow the instructions with discipline.
You are provided with an image depicting the scene and a phrase, `{phrase}`, describing the
desired state in abstraction. So, you must decompose this phrase and your primary task is to
clarify the user's intent regarding `{phrase}` through structured dialogue. Each of your question
is based on the previous answer.
    When decomposing `{phrase}`, focus on understanding the following that is related with the desired state:
        1. You must clarify the new position of the carriable objects to the desired state
        2. You must clarify which container should contain which objects in the desired state
        3. You do not care about how the objects to placed inside some container and do not mention that in the dialogue
        4. Your language should be sharp and humble
        5. After gathering all necessary details and confirming there is nothing more to add, provide a clear and concise description of the new image, but not by exposing the changes from the previous image
        6. The questions should be provided one at a time
        7. Generate a detailed task description in text format.
        8.You should not care about the room, only the objects you see. Also, you are connected with a image generation module and you can call the module when you think you have collected
    enough information for the desired state, just produce the phrase 'image-gen' to give the signal and in the same response provide to the user a json string for the relations of the containers and objects such as:
        image-gen, result = {{"name of container A": ["name of object type A"], "name of container B": ["name of object type B", "name of object type C"]}}.
        Always, you should focus on the last image of the dialogue. When you produce the 'image-gen' token you should produce also a text response based on the last image.
        Always give the final answer such as:
        Detailed task description in text format based on the last image.
        image-gen, result = {{"name of container A": ["name of object type A"], "name of container B": ["name of object type B", "name of object type C"]}}.
        DO NOT ADD ANY MORE TEXT OR INFORMATION ONLY THE description text and image-gen, result."""

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

# ----------------------------------------------------------------------------
# API CALLS
# ----------------------------------------------------------------------------

def qwen_call(payload):
    res = requests.post(QWEN_API_URL, json=payload, timeout=1200)
    res.raise_for_status()
    return res.json()

def inpaint_call(payload):
    res = requests.post(INPAINT_API_URL, json=payload, timeout=1200)
    res.raise_for_status()
    return res.json()

# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------

def image_paths(content):
    """Image file paths inside one Gradio history entry (tuple, list, dict or str)."""
    if isinstance(content, str):
        return [content] if content.lower().endswith(IMAGE_EXTS) and os.path.isfile(content) else []
    if isinstance(content, (tuple, list)):
        return [p for item in content for p in image_paths(item)]
    if hasattr(content, "value") and not isinstance(content, dict):  # Gradio component
        return image_paths(content.value)
    if isinstance(content, dict):
        for key in ("path", "file", "value"):
            if key in content:
                return image_paths(content[key])
    return []

def clean_history(history):
    """Keep only role + text/image paths, so the history can be sent as JSON."""
    cleaned = []
    for entry in history:
        content = entry.get("content")
        paths = image_paths(content)
        if paths:
            cleaned.append({"role": entry.get("role"), "content": paths})
        elif isinstance(content, str) and content:
            cleaned.append({"role": entry.get("role"), "content": content})
    return cleaned

def parse_image_gen(text):
    """Detect the 'image-gen, result = {...}' signal (same format as the original CLARA).

    Returns (found, relations, text_without_signal)."""
    found = "image-gen" in text
    relations = None
    match = re.search(r",?\s*[rR]esult\s*[=:]\s*(\{[^{}]+\})", text)
    if match:
        try:
            parsed = ast.literal_eval(match.group(1))
            relations = {str(k).strip(): [str(v).strip() for v in (vals if isinstance(vals, list) else [vals])]
                         for k, vals in parsed.items()}
            text = text.replace(match.group(0), "")
        except (SyntaxError, ValueError):
            relations = None
    text = text.replace("image-gen", "").replace("```json", "").replace("```", "").strip(" ,\n")
    return found, relations, text

def save_history(history, filename):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)

def arrange(image_path, relations, log):
    """Move every object into its container, one move at a time, on the same image."""
    current = image_path
    for container, objects in relations.items():
        for obj in objects:
            boxes = qwen_call({"intent": "locate", "image_path": current, "names": [obj, container]}).get("bboxes", {})
            if not boxes.get(obj):
                log.append(f"- Could not find **{obj}**, skipped.")
                continue
            if not boxes.get(container):
                log.append(f"- Could not find **{container}**, skipped {obj}.")
                continue
            res = inpaint_call({"intent": "move", "image_path": current, "source_bbox": boxes[obj],
                                "dest_bbox": boxes[container], "position": "center"})
            if res.get("status") != "success":
                log.append(f"- Moving {obj} failed: {res.get('message')}")
                continue
            current = res["response"]
            log.append(f"- {obj} → {container}")
    return current if current != image_path else None

# ----------------------------------------------------------------------------
# MAIN LOGIC
# ----------------------------------------------------------------------------
class ClaraSession:
    """Una conversación de CLARA. Al generar la imagen guarda el resultado y avisa de que terminó."""

    def __init__(self):
        self.final_purpose = None   # descripción final de la tarea
        self.image_path = None      # imagen generada
        self.done = threading.Event()

    def answer(self, message, history, history_file, max_new_tokens, temperature, top_p, sys_prompt):
        images = [p for e in history for p in image_paths(e.get("content"))] + list(message.get("files", []))
        if not images:
            yield "Please upload an image of the scene!"
            return
        image_path = images[-1]  # always work on the last image of the dialogue

        yield "Thinking..."
        try:
            reply = qwen_call({
                "intent": "clarify",
                "message": message,
                "history": clean_history(history),
                "sys_prompt": sys_prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }).get("response", "")
        except Exception as e:
            yield f"Error API: {str(e)}"
            return

        found, relations, text = parse_image_gen(reply)
        text = text or "(Qwen returned an empty reply — check the qwen3_api terminal.)"
        outputs = [{"role": "assistant", "content": text}]
        result_path = None

        if found and relations:
            yield text + "\n\nArranging the scene..."
            log = []
            try:
                result_path = arrange(image_path, relations, log)
            except Exception as e:
                log.append(f"- Error: {str(e)}")
            print("\n".join(log))  # details go to the terminal, not the chat
            if result_path:
                # Same as the original CLARA: image first, then the description
                outputs.insert(0, {"role": "assistant", "content": (result_path,)})
            else:
                outputs[0]["content"] = text + "\n\n" + "\n".join(log)  # show the log only when it failed
        elif found:
            outputs[0]["content"] = text + "\n\n(The model asked for image generation but gave no valid result dictionary.)"

        yield outputs

        try:
            saved = clean_history(history)
            if message.get("files"):
                saved.append({"role": "user", "content": list(message["files"])})
            saved.append({"role": "user", "content": message.get("text", "")})
            saved.append({"role": "assistant", "content": reply})
            save_history(saved, history_file)
        except Exception as e:
            print(f"Could not save history: {e}")

        # Fin de CLARA: guardar el resultado y avisar para cerrar la interfaz
        if result_path:
            self.final_purpose = text
            self.image_path = result_path
            self.done.set()

# ----------------------------------------------------------------------------
# INTERFACE
# ----------------------------------------------------------------------------

def build_ui(session):
    return gr.ChatInterface(
        fn=session.answer,
        title="CLARA (Qwen3-VL)",
        type="messages",
        additional_inputs=[
            gr.Textbox(label="History file", value="history/chat_log.json"),
            gr.Slider(16, 8192, 4096, step=16, label="Tokens"),
            gr.Slider(0.1, 2.0, 0.7, label="Temperature"),
            gr.Slider(0.1, 1.0, 0.9, label="Top-P"),
            gr.Textbox(label="System Prompt", value=CLARA_PROMPT, lines=8),
        ],
        multimodal=True,
        textbox=gr.MultimodalTextbox(file_types=["image"]),
        fill_height=True,
    )

# Υπολογίζει δυναμικά την απόλυτη διαδρομή για τον φάκελο outputs
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "Inpaint-Anything", "outputs")

def open_app_window(url):
    """Abre la interfaz en su propia ventana de navegador, que luego Python puede cerrar."""
    # Perfil propio: así la ventana es un proceso independiente de tu navegador normal.
    # Va dentro del proyecto (no en /tmp) porque Chromium/Firefox instalados con snap no ven /tmp.
    profile = os.path.join(PROJECT_ROOT, "browser_profile")
    os.makedirs(profile, exist_ok=True)

    for name in ("google-chrome", "chromium", "chromium-browser"):
        exe = shutil.which(name)
        if exe:
            return subprocess.Popen([exe, f"--app={url}", f"--user-data-dir={profile}",
                                     "--no-first-run", "--no-default-browser-check",
                                     "--window-size=1200,900"])
    exe = shutil.which("firefox")
    if exe:
        return subprocess.Popen([exe, "--new-instance", "--profile", profile, url])

    print(f"No se encontró Chrome/Chromium/Firefox. Abre {url} a mano.")
    return None


def run_clara(show_seconds=5):
    """Abre CLARA en su propia ventana, espera a que genere la imagen final,
    cierra la ventana y la interfaz, y devuelve la sesión con final_purpose e image_path."""
    session = ClaraSession()
    demo = build_ui(session)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        allowed_paths=[OUTPUTS_DIR],
        prevent_thread_lock=True,
    )
    browser = open_app_window("http://127.0.0.1:7861")
    try:
        while not session.done.wait(0.5):
            if browser and browser.poll() is not None:   # el usuario cerró la ventana a mano
                print("Ventana cerrada antes de terminar CLARA.")
                break
        else:
            time.sleep(show_seconds)                     # deja ver el resultado antes de cerrar
    finally:
        if browser and browser.poll() is None:
            browser.terminate()                          # cierra la ventana
        demo.close()                                     # cierra el servidor
    return session


if __name__ == "__main__":
    session = run_clara()
    print("final_purpose:", session.final_purpose)
    print("image_path:", session.image_path)
