import urllib.parse, gradio as gr

def plan_trip(destination: str, days: int):
    q = urllib.parse.quote(f"{destination} sights")
    maps = f"https://www.google.com/maps/search/?api=1&query={q}"
    md = (
        f"## {destination}: {days}-day plan\n"
        f"- Morning: Landmark A → Cafe B\n"
        f"- Afternoon: Museum C → Park D\n"
        f"- Evening: Viewpoint E → Dinner F\n\n"
        f"[Open on Google Maps]({maps})"
    )
    return md, f"Photo ideas for {destination}: landmarks, parks, sunset."

with gr.Blocks() as demo:
    gr.Markdown("# Travel Companion — daily plans + Google Maps links/photos")
    with gr.Row():
        dest = gr.Textbox(label="Destination", value="San Francisco")
        days = gr.Slider(1, 14, value=3, step=1, label="Days")
    go = gr.Button("Build plan")
    out_md = gr.Markdown()
    out_photos = gr.Textbox(label="Photo prompts", lines=3)
    go.click(plan_trip, [dest, days], [out_md, out_photos])

if __name__ == "__main__":
    demo.launch()
