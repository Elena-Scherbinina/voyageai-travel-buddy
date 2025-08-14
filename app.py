from huggingface_hub import InferenceClient
import os

# ---- Setup ----
HF_TOKEN = os.environ.get("HF_TOKEN") # set in Space
if not HF_TOKEN:
    print("⚠️ HF_TOKEN is not set. Text/images may fail until you add it in Space → Settings → Secrets.")


client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3", token=HF_TOKEN, timeout=45)
img_client = InferenceClient("black-forest-labs/FLUX.1-schnell", provider="hf-inference", token=HF_TOKEN)

# ---- Settings ----
dev_allow_imgs = False  # change to True to test image generation

allow_text_check = False # dont call additional time




# =========================
# Cell 3 — Gradio App (clean)
# =========================

import re
import gradio as gr

# ---- Safety checks: make sure clients exist (from Cell 2) ----
try:
    client
except NameError:
    raise RuntimeError("`client` is not defined. Run Cell 2 to create the text model client first.")

try:
    img_client
except NameError:
    # If you prefer to run without images, you can set img_client=None safely.
    img_client = None


# =========================
# 1) Constants / Prompt
# =========================
SYSTEM = """
You are a meticulous travel planner.
Return the itinerary as **valid Markdown** with this exact structure for each day:

### Day {n}
- **☀️ Morning:** 1–2 specific sights (🎟️ note if tickets)
- **🍽️ Lunch:** 1–2 places (cuisine/type)
- **🏛️ Afternoon:** 1–2 sights
- **🌅 Evening:** sunset spot / show / neighborhood
- **🚶 Logistics:** transit/walk notes, rough time blocks
- **💡 Backup:** indoor or low-energy option

Rules:
- Group by neighborhood to minimize backtracking.
- Mix indoor/outdoor and free/paid.
- Use short, specific bullet points.
- Include ticket note emoji 🎟️ where relevant.
- Separate each day with a horizontal rule `---`.

Linking rules:
- For each sight/restaurant, include a Markdown link: either the official site if obvious, or a Google Maps link.
- Example: [Eiffel Tower](https://www.google.com/maps?q=Eiffel+Tower+Paris).
- Never paste raw URLs; always use [Name](link) format.
- Put each section on its own bullet line; add a blank line between sections and between days.
"""



# =========================
# 2) Core logic
# =========================

def tidy_markdown(md: str) -> str:
    """Fixes common formatting issues for nicer Markdown rendering."""
    md = md.replace("### Day", "\n\n### Day")
    md = re.sub(r"\s*-\s+\*\*", r"\n- **", md)  # ensure section bullets on new lines
    return md.strip()


def _ensure_days(text, place, days):
    have = {int(m) for m in re.findall(r'^### Day\s+(\d+)', text, flags=re.M)}
    for d in range(1, days+1):
        if d not in have:
            text += (
                f"\n\n### Day {d}\n"
                f"- **☀️ Morning:** Key sights in {place} (🎟️ if needed)\n"
                f"- **🍽️ Lunch:** Local spot (cuisine)\n"
                f"- **🏛️ Afternoon:** 1–2 sights\n"
                f"- **🌅 Evening:** Viewpoint / promenade\n"
                f"- **🚶 Logistics:** Transit or walk\n"
                f"- **💡 Backup:** Indoor option\n"
            )
    return text



def _truncate_to_days(text, days):
    chunks = re.split(r'(?m)^(### Day \d+)\s*', text)
    out, count = [], 0
    for i in range(1, len(chunks), 2):
        count += 1
        if count > days:
            break
        out.append(chunks[i] + chunks[i+1])
    return "".join(out).strip() if out else text


def make_itinerary(place: str, days: int):
    days = int(days)
    user_msg = (
        f"Destination: {place}\n"
        f"Days: {days}\n"
        f"Return exactly {days} day sections — do NOT include extra days.\n"
        "Include Morning, Lunch, Afternoon, Evening, and one Backup per day.\n"
        "Return ONLY the itinerary in Markdown — no extra explanation."
    )

    try:
        resp = client.chat_completion(
            messages=[{"role":"system","content":SYSTEM},
                      {"role":"user","content":user_msg}],
            max_tokens=900, temperature=0.7,
        )
        md_text = (resp.choices[0].message["content"] or "").strip()
        status_text = "✅ Done. You can now generate images (if enabled)."
    except Exception as e:
        sections = []
        for d in range(1, days + 1):
            sections.append(
                f"### Day {d}\n"
                f"- **☀️ Morning:** Key sights in {place} (🎟️ if needed)\n"
                f"- **🍽️ Lunch:** Local spot (cuisine)\n"
                f"- **🏛️ Afternoon:** 1–2 museums/markets\n"
                f"- **🌅 Evening:** Viewpoint / promenade\n"
                f"- **🚶 Logistics:** Group nearby areas; transit or walk\n"
                f"- **💡 Backup:** Indoor/low-energy option\n"
            )
        md_text = "---\n\n".join(sections)
        status_text = "⚠️ Credits exceeded — showing fallback itinerary."

    # Now post-process the text
    start = md_text.find("### Day")
    if start != -1:
        md_text = md_text[start:]
    md_text = _truncate_to_days(md_text, days)
    md_text = _ensure_days(md_text, place, days)

    return tidy_markdown(md_text), status_text




# ---- Landmark extraction helpers for images ----
def _clean_landmark(text: str) -> str:
    text = re.sub(r"[•*–-]\s*", "", text)
    text = re.sub(r"[☀️🏛️🌅🚶💡🎟️]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = text.replace("**", "").strip(" -–:·").strip()
    text = re.sub(r"\b(viewpoint|view|walk|stroll|area|neighborhood|district)\b$", "", text, flags=re.I).strip()
    return text

_GENERIC = {
    "museum","park","square","cathedral","temple","bridge","tower","market","palace",
    "plaza","beach","harbor","fort","garden","monument","church","old town","downtown"
}

def _looks_specific(name: str) -> bool:
    low = name.lower()
    if not name or len(name) < 3:
        return False
    has_cap = any(w and w[0].isupper() for w in re.split(r"\s+", name))
    if low in _GENERIC:
        return False
    return has_cap or (" " in name)

def extract_landmarks(itinerary_md: str, max_count: int = 3):
    """Pick varied, specific sights from Morning/Afternoon/Evening sections."""
    lines = [ln.rstrip() for ln in itinerary_md.splitlines()]
    by_section = {"morning": [], "afternoon": [], "evening": []}
    section_keys = {
        "morning": re.compile(r"\*\*.*morning.*\*\*", re.I),
        "afternoon": re.compile(r"\*\*.*afternoon.*\*\*", re.I),
        "evening": re.compile(r"\*\*.*evening.*\*\*", re.I),
    }

    i = 0
    while i < len(lines):
        line = lines[i]
        hit = next((k for k, pat in section_keys.items() if pat.search(line)), None)
        if hit:
            j = i + 1
            collected = 0
            while j < len(lines) and collected < 4:
                nxt = lines[j].strip()
                if not nxt or nxt.startswith("###") or any(p.search(nxt) for p in section_keys.values()):
                    break
                if nxt.lstrip().startswith("-"):
                    item = _clean_landmark(nxt)
                    if item and _looks_specific(item):
                        by_section[hit].append(item)
                        collected += 1
                j += 1
            i = j
        else:
            i += 1

    # round-robin variety
    picks, idx, order = [], {"morning":0,"afternoon":0,"evening":0}, ["morning","afternoon","evening"]
    while len(picks) < max_count:
        progressed = False
        for sec in order:
            k = idx[sec]
            if k < len(by_section[sec]):
                cand = by_section[sec][k]
                if cand not in picks:
                    picks.append(cand)
                idx[sec] += 1
                progressed = True
                if len(picks) >= max_count:
                    break
        if not progressed:
            break
    return picks


def generate_images_from_itinerary(itinerary_md: str, n_images: int, allow: bool):
    """
    Returns (images[], status_text). If `allow` is False, no API calls are made.
    """
    if not allow:
        return [], "⚠️ Image generation is disabled."

    if img_client is None:
        return [], "⚠️ Image client is not configured."

    landmarks = extract_landmarks(itinerary_md, max_count=int(n_images))
    if not landmarks:
        return [], "No landmarks found in itinerary."

    images, notes = [], []
    for i, place in enumerate(landmarks, 1):
        prompt = f"{place}, iconic view, photorealistic, golden hour, high detail"
        try:
            img = img_client.text_to_image(prompt)
            images.append(img)
            notes.append(f"✅ Img {i}: {place}")
        except Exception as e:
            notes.append(f"❌ Img {i}: {place} — {e}")

    return images, "\n".join(notes)



# --- Small helpers used by events (place ABOVE the Blocks) ---
def show_loading(_place, _days):
    return "⏳ Generating itinerary, please wait..."

def compute_img_section_visibility(itinerary_text, allow):
    visible = bool((itinerary_text or "").strip()) and bool(allow)
    return gr.update(visible=visible), gr.update(interactive=visible)  # img_section, btn_img_top

def on_allow_change(allow, itinerary_text):
    visible = bool((itinerary_text or "").strip()) and bool(allow)
    return gr.update(visible=visible), gr.update(interactive=visible)



# =========================
# 3) Gradio UI
# =========================
with gr.Blocks(title="Travel Buddy AI") as demo:
    # Top status
    status_msg = gr.Label(label="Status")
    gr.HTML("""
    <style>
    .controls-row { display:flex; justify-content:space-between; align-items:center; width:100%; }
    .rightbar { display:flex; align-items:center; gap:10px; }
    .rightbar .label { color:#6b7280 !important; font-weight:600; }
    </style>
    """)


    # Inputs
    with gr.Row():
        place = gr.Textbox(label="Destination")
        days = gr.Slider(1, 21, step=1, value=3, label="Days")

    # Image controls (checkbox left, "# of images" + radio right)
    with gr.Row(elem_classes=["controls-row"]):
        allow_imgs = gr.Checkbox(label="Allow image generation (may use credits)", value=False)
        with gr.Row(elem_classes=["rightbar"]):
            gr.HTML("<div class='label'>Images to generate</div>")
            img_count = gr.Radio(choices=[1, 2, 3], value=2, show_label=False)

    # Generate itinerary
    btn_plan = gr.Button("Generate Itinerary", variant="primary")
    out_md = gr.Markdown(label="Itinerary")

    # Image section (hidden until itinerary exists AND checkbox is on)
    with gr.Column(visible=False) as img_section:
        btn_img_top = gr.Button("Generate Images from Itinerary", interactive=False)
        gallery_top = gr.Gallery(label="Preview Gallery", columns=3, height=400)
        status_top = gr.Textbox(label="Image Status", interactive=False, lines=3)

    # ---- Wire events ----
    btn_plan.click(
        fn=show_loading, inputs=[place, days], outputs=status_msg
    ).then(
        fn=make_itinerary, inputs=[place, days], outputs=[out_md, status_msg]
    ).then(
        fn=compute_img_section_visibility, inputs=[out_md, allow_imgs],
        outputs=[img_section, btn_img_top]
    )

    allow_imgs.change(
        fn=on_allow_change, inputs=[allow_imgs, out_md],
        outputs=[img_section, btn_img_top]
    )

    btn_img_top.click(
        fn=generate_images_from_itinerary,
        inputs=[out_md, img_count, allow_imgs],
        outputs=[gallery_top, status_top]
    ).then(
        fn=compute_img_section_visibility, inputs=[out_md, allow_imgs],
        outputs=[img_section, btn_img_top]
    )

demo.launch()
