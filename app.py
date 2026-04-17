# app.py — MorphGen

import torch
import numpy as np
import streamlit as st
from PIL import Image
import os
import random
from io import BytesIO

from config import LATENT_DIM, NUM_ATTRIBUTES, DEVICE, CHECKPOINT_DIR
from models.generator import Generator
from data.dataset import denormalize


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MorphGen — Conditional Face Generation",
    page_icon="",
    layout="wide"
)

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
            background-color: #0d0d0d;
            color: #e0e0e0;
        }
        h1, h2, h3 {
            font-family: 'Space Mono', monospace;
            color: #00ffe0;
        }
        .stButton > button {
            background-color: #00ffe0;
            color: #0d0d0d;
            font-family: 'Space Mono', monospace;
            font-weight: 700;
            border: none;
            border-radius: 4px;
            padding: 0.5rem 1.5rem;
            width: 100%;
        }
        .stButton > button:hover {
            background-color: #ffd700;
            color: #0d0d0d;
        }
        .stCheckbox > label {
            font-size: 0.85rem;
        }
    </style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# CelebA attribute names
# ---------------------------------------------------------------------------

ATTR_NAMES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
]

DISPLAY_NAMES = [a.replace("_", " ").title() for a in ATTR_NAMES]

ATTR_GROUPS = {
    "Face Shape": ["Attractive", "Chubby", "Oval Face", "High Cheekbones", "Pale Skin", "Young"],
    "Hair":       ["Bald", "Bangs", "Black Hair", "Blond Hair", "Brown Hair", "Gray Hair",
                   "Receding Hairline", "Straight Hair", "Wavy Hair"],
    "Facial Hair":["5 O Clock Shadow", "Goatee", "Mustache", "No Beard", "Sideburns"],
    "Features":   ["Arched Eyebrows", "Bags Under Eyes", "Big Lips", "Big Nose",
                   "Bushy Eyebrows", "Double Chin", "Eyeglasses", "Narrow Eyes", "Pointy Nose",
                   "Rosy Cheeks"],
    "Expression": ["Mouth Slightly Open", "Smiling"],
    "Gender":     ["Male", "Heavy Makeup"],
    "Accessories":["Wearing Earrings", "Wearing Hat", "Wearing Lipstick",
                   "Wearing Necklace", "Wearing Necktie"],
}


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

@st.cache_resource
def load_generator():
    G = Generator().to(DEVICE)
    weights_path = os.path.join(CHECKPOINT_DIR, "generator_final.pt")
    if not os.path.exists(weights_path):
        st.error(f"Trained weights not found at `{weights_path}`.")
        st.stop()
    G.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    G.eval()
    return G


@st.cache_resource
def load_upsampler():
    from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor
    processor = Swin2SRImageProcessor()
    model     = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x4-64")
    return processor, model


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def upscale(fake_np):
    pil_img = Image.fromarray(fake_np)
    processor, sr_model = load_upsampler()
    inputs  = processor(pil_img, return_tensors="pt")
    with torch.no_grad():
        outputs = sr_model(**inputs)
    sr_img = outputs.reconstruction.squeeze().clamp(0, 1)
    sr_img = sr_img.permute(1, 2, 0).numpy()
    return (sr_img * 255).astype(np.uint8)


def generate_image(G, attr_vector, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        z     = torch.randn(1, LATENT_DIM, device=DEVICE)
        attrs = torch.tensor([attr_vector], dtype=torch.float32, device=DEVICE)
        fake  = G(z, attrs)
        fake  = denormalize(fake).squeeze(0)
        fake  = fake.permute(1, 2, 0).cpu().numpy()
        fake  = (fake * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(upscale(fake))


def generate_from_z(G, z, attr_vector):
    attrs = torch.tensor([attr_vector], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        fake = G(z, attrs)
        fake = denormalize(fake).squeeze(0)
        fake = fake.permute(1, 2, 0).cpu().numpy()
        fake = (fake * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(upscale(fake))


def build_attr_vector(selected_attrs):
    return [1.0 if selected_attrs.get(name, False) else 0.0 for name in DISPLAY_NAMES]


def random_attr_vector():
    return [float(random.random() > 0.7) for _ in DISPLAY_NAMES]


def image_to_bytes(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

for key, default in [
    ("history", []),
    ("last_image", None),
    ("last_image_a", None),
    ("last_image_b", None),
    ("fixed_z", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("MorphGen")
st.markdown("**Conditional GAN trained on CelebA** — select attributes to control face generation.")
st.markdown("---")

G = load_generator()

tab1, tab2, tab3, tab4 = st.tabs(["Generate", "Compare", "Interpolate", "Batch"])


# ============================================================
# TAB 1 — Generate
# ============================================================
with tab1:
    col_controls, col_output = st.columns([1, 1], gap="large")

    with col_controls:
        st.markdown("### Attributes")
        selected_attrs = {}
        for group, attrs in ATTR_GROUPS.items():
            with st.expander(group, expanded=(group in ["Hair", "Expression", "Gender"])):
                for attr in attrs:
                    key = f"t1_{attr.replace(' ', '_').upper()}"
                    selected_attrs[attr] = st.checkbox(attr, key=key, value=False)

        st.markdown("---")
        st.markdown("### Randomness")
        use_fixed_seed = st.toggle("Fix random seed", value=False)
        seed_val = st.number_input("Seed", min_value=0, max_value=99999,
                                   value=42, step=1) if use_fixed_seed else None
        st.markdown("---")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            generate_btn = st.button("Generate Face")
        with col_btn2:
            random_btn = st.button("Random Face")

    with col_output:
        st.markdown("### Generated Face")

        if random_btn:
            attr_vector = random_attr_vector()
            img = generate_image(G, attr_vector, seed=seed_val)
            st.session_state["last_image"] = img
            st.session_state["history"].insert(0, img)
            st.session_state["history"] = st.session_state["history"][:8]

        elif generate_btn:
            attr_vector = build_attr_vector(selected_attrs)
            img = generate_image(G, attr_vector, seed=seed_val)
            st.session_state["last_image"] = img
            st.session_state["history"].insert(0, img)
            st.session_state["history"] = st.session_state["history"][:8]

        elif st.session_state["last_image"] is None:
            attr_vector = build_attr_vector(selected_attrs)
            img = generate_image(G, attr_vector, seed=seed_val)
            st.session_state["last_image"] = img
            st.session_state["history"].insert(0, img)
            st.session_state["history"] = st.session_state["history"][:8]

        if st.session_state["last_image"]:
            st.image(st.session_state["last_image"],
                     caption="Generated face",
                     use_container_width=False,
                     width=256)
            st.download_button(
                label="Download Face",
                data=image_to_bytes(st.session_state["last_image"]),
                file_name="morphgen_face.png",
                mime="image/png"
            )

        active = [n for n, v in selected_attrs.items() if v]
        if active:
            st.markdown("**Active attributes:** " + ", ".join(active))
        else:
            st.markdown("*No attributes selected — pure random generation.*")

        if st.session_state["history"]:
            st.markdown("---")
            st.markdown("### History")
            cols = st.columns(len(st.session_state["history"]))
            for col, hist_img in zip(cols, st.session_state["history"]):
                with col:
                    st.image(hist_img, use_container_width=True)


# ============================================================
# TAB 2 — Compare
# ============================================================
with tab2:
    st.markdown("### Side-by-Side Comparison")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Face A")
        attrs_a = {}
        for group, attrs in ATTR_GROUPS.items():
            with st.expander(group, expanded=False):
                for attr in attrs:
                    attrs_a[attr] = st.checkbox(attr, key=f"a_{attr.replace(' ', '_').upper()}", value=False)
        gen_a = st.button("Generate A")

    with col_b:
        st.markdown("#### Face B")
        attrs_b = {}
        for group, attrs in ATTR_GROUPS.items():
            with st.expander(group, expanded=False):
                for attr in attrs:
                    attrs_b[attr] = st.checkbox(attr, key=f"b_{attr.replace(' ', '_').upper()}", value=False)
        gen_b = st.button("Generate B")

    if gen_a:
        st.session_state["last_image_a"] = generate_image(G, build_attr_vector(attrs_a))
    if gen_b:
        st.session_state["last_image_b"] = generate_image(G, build_attr_vector(attrs_b))

    st.markdown("---")
    res_a, res_b = st.columns(2)
    with res_a:
        if st.session_state["last_image_a"]:
            st.image(st.session_state["last_image_a"], caption="Face A", width=256)
            st.download_button("Download A",
                               data=image_to_bytes(st.session_state["last_image_a"]),
                               file_name="morphgen_face_a.png", mime="image/png")
    with res_b:
        if st.session_state["last_image_b"]:
            st.image(st.session_state["last_image_b"], caption="Face B", width=256)
            st.download_button("Download B",
                               data=image_to_bytes(st.session_state["last_image_b"]),
                               file_name="morphgen_face_b.png", mime="image/png")


# ============================================================
# TAB 3 — Interpolate
# ============================================================
with tab3:
    st.markdown("### Attribute Interpolation")
    st.markdown("Smoothly morph between two attribute sets using the same base face.")

    col_ia, col_ib = st.columns(2)

    with col_ia:
        st.markdown("#### Start Attributes")
        attrs_start = {}
        for group, attrs in ATTR_GROUPS.items():
            with st.expander(group, expanded=False):
                for attr in attrs:
                    attrs_start[attr] = st.checkbox(attr, key=f"s_{attr.replace(' ', '_').upper()}", value=False)

    with col_ib:
        st.markdown("#### End Attributes")
        attrs_end = {}
        for group, attrs in ATTR_GROUPS.items():
            with st.expander(group, expanded=False):
                for attr in attrs:
                    attrs_end[attr] = st.checkbox(attr, key=f"e_{attr.replace(' ', '_').upper()}", value=False)

    steps = st.slider("Interpolation steps", min_value=3, max_value=7, value=5)
    interp_btn = st.button("Interpolate")

    if interp_btn:
        vec_start = build_attr_vector(attrs_start)
        vec_end   = build_attr_vector(attrs_end)

        # Fix a single z for consistent base face
        torch.manual_seed(42)
        fixed_z = torch.randn(1, LATENT_DIM, device=DEVICE)

        st.markdown("---")
        cols = st.columns(steps)
        for i, col in enumerate(cols):
            alpha = i / (steps - 1)
            interp_vec = [
                (1 - alpha) * s + alpha * e
                for s, e in zip(vec_start, vec_end)
            ]
            img = generate_from_z(G, fixed_z, interp_vec)
            with col:
                st.image(img, caption=f"α={alpha:.1f}", use_container_width=True)


# ============================================================
# TAB 4 — Batch
# ============================================================
with tab4:
    st.markdown("### Batch Generation")
    st.markdown("Generate multiple faces with the same attributes but different seeds.")

    col_ctrl, col_grid = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.markdown("#### Attributes")
        batch_attrs = {}
        for group, attrs in ATTR_GROUPS.items():
            with st.expander(group, expanded=False):
                for attr in attrs:
                    batch_attrs[attr] = st.checkbox(attr, key=f"batch_{attr.replace(' ', '_').upper()}", value=False)
        n_images  = st.slider("Number of faces", min_value=2, max_value=8, value=4)
        batch_btn = st.button("Generate Batch")

    with col_grid:
        st.markdown("#### Results")
        if batch_btn:
            attr_vector = build_attr_vector(batch_attrs)
            cols = st.columns(min(n_images, 4))
            batch_imgs = []
            for i in range(n_images):
                img = generate_image(G, attr_vector, seed=i * 42)
                batch_imgs.append(img)
                with cols[i % 4]:
                    st.image(img, use_container_width=True, caption=f"Face {i+1}")

            # Download all as zip
            import zipfile
            zip_buf = BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                for i, img in enumerate(batch_imgs):
                    img_buf = BytesIO()
                    img.save(img_buf, format="PNG")
                    zf.writestr(f"morphgen_face_{i+1}.png", img_buf.getvalue())
            st.download_button(
                label="Download All",
                data=zip_buf.getvalue(),
                file_name="morphgen_batch.zip",
                mime="application/zip"
            )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    "<p style='font-family: Space Mono, monospace; font-size: 0.75rem; color: #555;'>"
    "MorphGen · Conditional DCGAN · WGAN-GP · CelebA · PyTorch"
    "</p>",
    unsafe_allow_html=True
)
