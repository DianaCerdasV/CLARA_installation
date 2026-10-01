import base64
import json
import os
import re
import torch
from ollama import Client
from PIL import Image
import io
from transformers import AutoProcessor, AutoModelForCausalLM 
import gc

#   Grounding 
FLORENCE_ID = "microsoft/Florence-2-base"
device = "cuda:0" # Qwen and Florence can share GPU 0
fl_model = AutoModelForCausalLM.from_pretrained(
    FLORENCE_ID, 
    trust_remote_code=True,
    attn_implementation="eager",
    torch_dtype=torch.float16
).eval().to(device)
fl_processor = AutoProcessor.from_pretrained(FLORENCE_ID, trust_remote_code=True)

client = Client(host="http://127.0.0.1:11434")
MODEL_NAME = "qwen3-vl:8b" 

def get_florence_bbox(image, text_input):
    """Florence-2 converts text to pixel-perfect boxes with Memory Optimization"""
    task_prompt = "<OPEN_VOCABULARY_DETECTION>"
    prompt = task_prompt + text_input
    
    # 1. THE VRAM SAVER
    eval_img = image.copy()
    eval_img.thumbnail((768, 768)) 
    
    inputs = fl_processor(text=prompt, images=eval_img, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    
    # 2. Garbage Collector + CUDA cache
    gc.collect()
    torch.cuda.empty_cache()
    
    with torch.inference_mode():
        generated_ids = fl_model.generate(
          input_ids=inputs["input_ids"],
          pixel_values=inputs["pixel_values"],
          max_new_tokens=1024,
          num_beams=1,          
          use_cache=False)

    generated_text = fl_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    parsed_answer = fl_processor.post_process_generation(
        generated_text, 
        task=task_prompt,  
        image_size=(eval_img.width, eval_img.height)
    )
    
    results = parsed_answer[task_prompt]  
    if not results.get('bboxes'): return None
    
    box = results['bboxes'][0] # [x1, y1, x2, y2]
    
    # 3. Conversion to 0-1000 scale. 
    ymin = (box[1] / eval_img.height) * 1000
    xmin = (box[0] / eval_img.width) * 1000
    ymax = (box[3] / eval_img.height) * 1000
    xmax = (box[2] / eval_img.width) * 1000
    return [ymin, xmin, ymax, xmax]

# ==========================================
# CLARA: purpose clarification dialogue
# ==========================================
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

def image_paths(content):
    """Image file paths inside one Gradio history entry (tuple, list, dict or str)."""
    if isinstance(content, str):
        return [content] if content.lower().endswith(IMAGE_EXTS) and os.path.isfile(content) else []
    if isinstance(content, (tuple, list)):
        return [p for item in content for p in image_paths(item)]
    if isinstance(content, dict):
        for key in ("path", "file", "value"):
            if key in content:
                return image_paths(content[key])
    return []

def is_text(content):
    return isinstance(content, str) and content and not image_paths(content)

def clarify(payload):
    """One turn of the CLARA dialogue: system prompt + full history -> Qwen3-VL."""
    message = payload["message"]
    history = payload.get("history", [])
    sys_prompt = payload.get("sys_prompt", "")

    # {phrase} = the user's first request in the conversation (as in the original CLARA)
    phrase = next((e["content"] for e in history if e.get("role") == "user" and is_text(e.get("content"))),
                  message.get("text", ""))
    sys_prompt = sys_prompt.replace("{phrase}", phrase).replace("{{", "{").replace("}}", "}")

    messages = [{"role": "system", "content": sys_prompt}]
    pending_images = []  # images waiting to be attached to the next user turn

    def add(role, text, images):
        if role == "user":
            images = pending_images + images
            pending_images.clear()
        elif images:  # Ollama only accepts images on user turns
            pending_images.extend(images)
            images = []
        if not text and not images:
            return
        last = messages[-1]
        if last["role"] == role and role != "system":
            last["content"] = (last["content"] + "\n" + text).strip()
            last.setdefault("images", []).extend(images)
        else:
            entry = {"role": role, "content": text}
            if images:
                entry["images"] = images
            messages.append(entry)

    for entry in history:
        content = entry.get("content")
        role = entry.get("role", "user")
        add(role, content if is_text(content) else "", [encode_image_to_base64(p) for p in image_paths(content)])
    add("user", message.get("text", ""), [encode_image_to_base64(p) for p in message.get("files", [])])

    response = client.chat(
        model=MODEL_NAME,
        messages=messages,
        options={
            "num_ctx": 8192,
            "temperature": float(payload.get("temperature", 0.7)),
            "top_p": float(payload.get("top_p", 0.9)),
            "num_predict": int(payload.get("max_new_tokens", 1024)),
        },
    )
    reply = response.message.content or ""
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
    print(f"CLARA reply: {reply}")
    return {"response": reply}

def locate(payload):
    """Florence-2 boxes for a list of object names in one image."""
    raw_img = Image.open(payload["image_path"]).convert("RGB")
    return {"bboxes": {name: get_florence_bbox(raw_img, name) for name in payload.get("names", [])}}

def generate(payload):
    try:

        intent = payload.get("intent", "chat")

        if intent == "clarify":
            return clarify(payload)
        if intent == "locate":
            return locate(payload)

        text = payload["message"].get("text", "")
        files = payload["message"].get("files", [])

        # ==========================================
        # MODE 1: EDIT (Remove Object)
        # ==========================================
        if intent == "edit":
            if not files: return {"objects": [], "error": "No image provided for editing."}

            raw_img = Image.open(files[0]).convert("RGB")
            img_b64 = encode_image_to_base64(files[0])

            qwen_prompt = f"The user wants to: '{text}'. What specific single object should be removed? Answer with 1-3 words only."

            response = client.chat(
              model=MODEL_NAME, 
              messages=[{"role": "user", "content": qwen_prompt, "images": [img_b64]}], 
              options={"num_ctx": 4096}
             )

            target_object = response.message.content.strip()
            print(f"Qwen identified target: {target_object}")

            precise_bbox = get_florence_bbox(raw_img, target_object)

            if precise_bbox:
                return {"objects": [{"name": target_object, "bbox": precise_bbox}]}
            else:
                return {"objects": [], "error": "Could not locate object in the image."}

        # ==========================================
        # MODE 2: MOVE (Move Objects)
        # ==========================================
        elif intent == "move":
            if not files: return {"error": "No image provided for moving."}

            raw_img = Image.open(files[0]).convert("RGB")
            img_b64 = encode_image_to_base64(files[0])

            
            qwen_prompt = f"""The user wants to: '{text}'. 
            Identify the object to be moved (source), the reference object (destination), and the spatial relationship (position).
            Position MUST be exactly one of these words: "top", "bottom", "left", "right", "center".
            Reply ONLY with a valid JSON format like this: {{"source": "object to move", "destination": "reference object", "position": "right"}}"""

            response = client.chat(
                model=MODEL_NAME, 
                messages=[{"role": "user", "content": qwen_prompt, "images": [img_b64]}], 
                options={"num_ctx": 4096}
            )

            reply_text = response.message.content.strip()
            
            
            reply_text = reply_text.replace("```json", "").replace("```", "").strip()
            
            try:
                move_data = json.loads(reply_text)
                src_obj = move_data.get("source", "")
                dst_obj = move_data.get("destination", "")
                position = move_data.get("position", "center") 
            except json.JSONDecodeError:
                print(f"Qwen didn't return valid JSON. Raw reply: {reply_text}")
                return {"error": "Failed to parse source, destination and position from Qwen."}

            print(f"Qwen identified MOVE -> Source: '{src_obj}', Destination: '{dst_obj}', Position: '{position}'")

            
            src_bbox = get_florence_bbox(raw_img, src_obj)
            dst_bbox = get_florence_bbox(raw_img, dst_obj)

            if not src_bbox:
                return {"error": f"Could not locate the source object: {src_obj}"}
            if not dst_bbox:
                return {"error": f"Could not locate the destination reference: {dst_obj}"}

            return {
                "source": {"name": src_obj, "bbox": src_bbox},
                "destination": {"name": dst_obj, "bbox": dst_bbox},
                "position": position 
            }

        # ==========================================
        # MODE 3: CHAT
        # ==========================================
        else:
            msg_content = {"role": "user", "content": text}
            
            if files:
                msg_content["images"] = [encode_image_to_base64(files[0])]
                print("Chatting with image...")
            else:
                print("Chatting (text only)...")

            response = client.chat(
             model=MODEL_NAME, 
             messages=[msg_content], 
             options={"num_ctx": 4096}
             )
            
            return {"response": response.message.content}

    except Exception as e:
        import traceback
        traceback.print_exc()
        
        if payload.get("intent") == "edit":
            return {"objects": [], "error": str(e)}
        return {"error": str(e)}

def encode_image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
