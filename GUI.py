"""
TrashNet Classifier — Professional GUI (Fixed)
================================================
- Scrollable left controls panel so Run button is always accessible
- Sticky Run button pinned to bottom of left panel
- Better error reporting for analysis failures
- LIME fallback to Grad-CAM if lime not installed

Requirements:
    pip install customtkinter tensorflow pillow numpy lime scikit-image scikit-learn

Run:
    python trashnet_app.py

Place your model at: model/garbage_classifier.keras
"""

import os
import json
import threading
import traceback
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image, ImageDraw
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox

# ─────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

C = {
    "bg":          "#F5F3EE",
    "panel":       "#FFFFFF",
    "card":        "#FAFAF8",
    "dark":        "#1A1A1A",
    "ink":         "#2C2C2C",
    "muted":       "#8A8A8A",
    "rule":        "#E0DDD7",
    "accent":      "#2D6A4F",
    "accent_lt":   "#52B788",
    "accent_pale": "#D8F3DC",
    "red":         "#C1121F",
    "amber":       "#E76F51",
}

CLASS_META = {
    "cardboard": {"color": "#B5838D", "pale": "#F2E9EA", "desc": "Corrugated boxes, packaging"},
    "glass":     {"color": "#4895EF", "pale": "#EBF4FF", "desc": "Bottles, jars, containers"},
    "metal":     {"color": "#7B7B8D", "pale": "#EFEFF3", "desc": "Cans, foil, metal scraps"},
    "paper":     {"color": "#D4A373", "pale": "#FDF4E7", "desc": "Newspapers, office paper"},
    "plastic":   {"color": "#E63946", "pale": "#FEECEE", "desc": "Bags, bottles, containers"},
    "trash":     {"color": "#6B6B6B", "pale": "#F0F0F0", "desc": "Non-recyclable waste"},
}

DEFAULT_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
DEFAULT_MODEL   = "model/garbage_classifier.keras"


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
class TrashNetApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("TrashNet — Waste Classification System")
        self.geometry("1280x860")
        self.minsize(1100, 760)
        self.configure(fg_color=C["bg"])

        self.model            = None
        self.class_names      = list(DEFAULT_CLASSES)
        self.current_img_path = None

        self._build_shell()
        self._show_intro()
        self._try_autoload()

    # ─────────────────────────────────────────────
    # SHELL
    # ─────────────────────────────────────────────
    def _build_shell(self):
        nav = ctk.CTkFrame(self, fg_color=C["dark"], corner_radius=0, height=56)
        nav.pack(fill="x")
        nav.pack_propagate(False)

        logo = ctk.CTkFrame(nav, fg_color="transparent")
        logo.pack(side="left", padx=24, fill="y")
        ctk.CTkLabel(logo, text="♻", font=ctk.CTkFont(size=22),
                     text_color=C["accent_lt"]).pack(side="left", padx=(0, 8), pady=16)
        ctk.CTkLabel(logo, text="TrashNet",
                     font=ctk.CTkFont(family="Georgia", size=18, weight="bold"),
                     text_color="#FFFFFF").pack(side="left")
        ctk.CTkLabel(logo, text="  Classifier",
                     font=ctk.CTkFont(size=13), text_color=C["muted"]).pack(side="left", pady=3)

        nav_right = ctk.CTkFrame(nav, fg_color="transparent")
        nav_right.pack(side="right", padx=20, fill="y")

        self._nav_btns = {}
        for label, cmd in [("Overview", self._show_intro), ("Classify", self._show_predict)]:
            btn = ctk.CTkButton(nav_right, text=label, width=110, height=32,
                                fg_color="transparent", hover_color="#333333",
                                text_color="#CCCCCC", corner_radius=6,
                                font=ctk.CTkFont(size=13), command=cmd)
            btn.pack(side="left", padx=4, pady=12)
            self._nav_btns[label] = btn

        self.model_pill = ctk.CTkLabel(nav_right, text="  No Model",
                                        fg_color="#2A2A2A", corner_radius=20,
                                        text_color=C["muted"],
                                        font=ctk.CTkFont(size=11), padx=12, pady=4)
        self.model_pill.pack(side="left", padx=(16, 0), pady=14)

        self.page_container = ctk.CTkFrame(self, fg_color="transparent")
        self.page_container.pack(fill="both", expand=True)

    def _set_nav_active(self, label):
        for k, btn in self._nav_btns.items():
            btn.configure(fg_color="#333333" if k == label else "transparent",
                          text_color="#FFFFFF" if k == label else "#CCCCCC")

    def _clear_page(self):
        for w in self.page_container.winfo_children():
            w.destroy()

    # ─────────────────────────────────────────────
    # OVERVIEW PAGE
    # ─────────────────────────────────────────────
    def _show_intro(self):
        self._set_nav_active("Overview")
        self._clear_page()

        scroll = ctk.CTkScrollableFrame(self.page_container, fg_color=C["bg"],
                                         scrollbar_button_color=C["rule"],
                                         scrollbar_button_hover_color=C["muted"])
        scroll.pack(fill="both", expand=True)

        # Hero
        hero = ctk.CTkFrame(scroll, fg_color=C["dark"], corner_radius=0)
        hero.pack(fill="x")
        hero_in = ctk.CTkFrame(hero, fg_color="transparent")
        hero_in.pack(padx=80, pady=56)
        ctk.CTkLabel(hero_in, text="Intelligent Waste Classification",
                     font=ctk.CTkFont(family="Georgia", size=34, weight="bold"),
                     text_color="#FFFFFF").pack(anchor="w")
        ctk.CTkLabel(hero_in,
                     text="Powered by EfficientNet deep learning · Explained with LIME",
                     font=ctk.CTkFont(size=14), text_color="#888888").pack(anchor="w", pady=(6, 24))
        stats = ctk.CTkFrame(hero_in, fg_color="transparent")
        stats.pack(anchor="w")
        for val, lbl in [("2,527","Total Images"),("6","Categories"),("~92%","Val Accuracy"),("LIME","Explainability")]:
            sc = ctk.CTkFrame(stats, fg_color="#2A2A2A", corner_radius=10)
            sc.pack(side="left", padx=(0, 12))
            ctk.CTkLabel(sc, text=val,
                         font=ctk.CTkFont(family="Courier New", size=22, weight="bold"),
                         text_color=C["accent_lt"]).pack(padx=20, pady=(12, 2))
            ctk.CTkLabel(sc, text=lbl, font=ctk.CTkFont(size=11),
                         text_color="#777777").pack(padx=20, pady=(0, 12))

        def section(txt):
            f = ctk.CTkFrame(scroll, fg_color="transparent")
            f.pack(fill="x", padx=80, pady=(40, 0))
            ctk.CTkLabel(f, text=txt,
                         font=ctk.CTkFont(family="Georgia", size=20, weight="bold"),
                         text_color=C["ink"]).pack(side="left")
            ctk.CTkFrame(f, fg_color=C["rule"], height=1).pack(side="left", fill="x",
                                                                 expand=True, padx=16, pady=4)

        section("About the Dataset")
        about = ctk.CTkFrame(scroll, fg_color="transparent")
        about.pack(fill="x", padx=80, pady=(12, 0))
        about.columnconfigure((0, 1), weight=1)
        for col, txt in enumerate([
            "TrashNet is a benchmark dataset developed at Stanford University containing 2,527 "
            "photographs of individual waste items. Each image is labeled into one of six recycling "
            "categories: cardboard, glass, metal, paper, plastic, and general trash.\n\n"
            "The dataset supports machine learning research in automated recycling sorting — "
            "a critical challenge for sustainable waste management worldwide.",
            "Images were photographed against a white background using a Nexus 5x smartphone, "
            "ensuring controlled lighting and minimal background noise.\n\n"
            "Class distribution is imbalanced — paper (594 images) dominates while trash (137) "
            "is underrepresented — mirroring real-world sorting challenges that deep learning "
            "models must learn to handle robustly."
        ]):
            ctk.CTkLabel(about, text=txt, wraplength=470, justify="left",
                         font=ctk.CTkFont(size=13), text_color=C["ink"]
                         ).grid(row=0, column=col, sticky="nw", padx=(0, 28))

        section("Waste Categories")
        ctk.CTkLabel(scroll, text="Six recyclable and non-recyclable categories",
                     font=ctk.CTkFont(size=12), text_color=C["muted"]).pack(anchor="w", padx=80, pady=(4, 12))

        grid = ctk.CTkFrame(scroll, fg_color="transparent")
        grid.pack(fill="x", padx=80, pady=(0, 36))
        class_data = [
            ("cardboard","403 images","Flattened boxes, tubes, cardboard sleeves. Recycled into new paper products."),
            ("glass",    "501 images","Bottles and jars of all colors. Sorted by color at recycling facilities."),
            ("metal",    "410 images","Aluminum cans, steel tins, foil. One of the most infinitely recyclable materials."),
            ("paper",    "594 images","Newspapers, magazines, office paper. Must be clean and dry for recycling."),
            ("plastic",  "482 images","PET bottles, HDPE containers. Recycling code (1–7) determines recyclability."),
            ("trash",    "137 images","Non-recyclable waste: food-contaminated, mixed or broken materials."),
        ]
        for i, (cls, count, desc) in enumerate(class_data):
            r, c = divmod(i, 3)
            meta = CLASS_META[cls]
            card = ctk.CTkFrame(grid, fg_color=C["panel"], corner_radius=12,
                                 border_width=1, border_color=C["rule"])
            card.grid(row=r, column=c, padx=7, pady=7, sticky="nsew")
            grid.columnconfigure(c, weight=1)
            ctk.CTkFrame(card, fg_color=meta["color"], height=4, corner_radius=0).pack(fill="x")
            inn = ctk.CTkFrame(card, fg_color="transparent")
            inn.pack(fill="both", expand=True, padx=16, pady=12)
            top = ctk.CTkFrame(inn, fg_color="transparent")
            top.pack(fill="x")
            ctk.CTkLabel(top, text=cls.upper(),
                         font=ctk.CTkFont(family="Georgia", size=14, weight="bold"),
                         text_color=C["ink"]).pack(side="left")
            ctk.CTkLabel(top, text=count, font=ctk.CTkFont(family="Courier New", size=10),
                         text_color=C["muted"], fg_color=meta["pale"],
                         corner_radius=8, padx=7, pady=2).pack(side="right")
            ctk.CTkLabel(inn, text=meta["desc"], font=ctk.CTkFont(size=11),
                         text_color=C["muted"], anchor="w").pack(anchor="w", pady=(4, 4))
            ctk.CTkLabel(inn, text=desc, wraplength=270, justify="left",
                         font=ctk.CTkFont(size=12), text_color=C["ink"]).pack(anchor="w")

        section("Model Architecture")
        mtable = ctk.CTkFrame(scroll, fg_color=C["panel"], corner_radius=12,
                               border_width=1, border_color=C["rule"])
        mtable.pack(fill="x", padx=80, pady=(12, 0))
        for i, (k, v) in enumerate([
            ("Backbone",       "EfficientNet-B0  (pretrained ImageNet-1K)"),
            ("Input Shape",    "224 × 224 × 3  (RGB)"),
            ("Head",           "GlobalAveragePool → BN → Dense(256, ReLU) → Dropout(0.5) → Dense(6, Softmax)"),
            ("Augmentation",   "Random flip, rotation ±25°, zoom ±20%, contrast, translation"),
            ("Fine-tuning",    "Unfreeze last 20 layers · LR = 1e-5"),
            ("Explainability", "LIME — Local Interpretable Model-agnostic Explanations"),
        ]):
            row = ctk.CTkFrame(mtable, fg_color=C["card"] if i % 2 == 0 else C["panel"], corner_radius=0)
            row.pack(fill="x", padx=1)
            ctk.CTkLabel(row, text=k, width=160, anchor="w",
                         font=ctk.CTkFont(family="Courier New", size=12, weight="bold"),
                         text_color=C["accent"]).pack(side="left", padx=(20, 0), pady=10)
            ctk.CTkLabel(row, text=v, anchor="w",
                         font=ctk.CTkFont(size=12), text_color=C["ink"]).pack(side="left", padx=14)

        cta = ctk.CTkFrame(scroll, fg_color=C["accent"], corner_radius=0)
        cta.pack(fill="x", pady=(36, 0))
        ctk.CTkLabel(cta, text="Ready to classify your waste image?",
                     font=ctk.CTkFont(family="Georgia", size=20, weight="bold"),
                     text_color="#FFFFFF").pack(pady=(28, 6))
        ctk.CTkLabel(cta, text="Upload any image · get instant prediction · explore LIME explanation",
                     font=ctk.CTkFont(size=13), text_color="#B7E4C7").pack()
        ctk.CTkButton(cta, text="Open Classifier  →", height=42, width=190,
                      fg_color="#FFFFFF", hover_color="#D8F3DC",
                      text_color=C["accent"], corner_radius=21,
                      font=ctk.CTkFont(size=14, weight="bold"),
                      command=self._show_predict).pack(pady=24)

    # ─────────────────────────────────────────────
    # CLASSIFY PAGE
    # ─────────────────────────────────────────────
    def _show_predict(self):
        self._set_nav_active("Classify")
        self._clear_page()

        outer = ctk.CTkFrame(self.page_container, fg_color="transparent")
        outer.pack(fill="both", expand=True, padx=20, pady=16)
        outer.columnconfigure(0, weight=0, minsize=285)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        self._build_left(outer)

        self.right_panel = ctk.CTkFrame(outer, fg_color="transparent")
        self.right_panel.grid(row=0, column=1, sticky="nsew")
        self._show_right_placeholder()

    # ── LEFT: header + scrollable body + sticky run button ──
    def _build_left(self, parent):
        container = ctk.CTkFrame(parent, fg_color=C["panel"], corner_radius=14,
                                  border_width=1, border_color=C["rule"], width=285)
        container.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        container.pack_propagate(False)
        container.grid_rowconfigure(1, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Header (fixed)
        hdr = ctk.CTkFrame(container, fg_color=C["dark"], corner_radius=0, height=50)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text="Controls",
                     font=ctk.CTkFont(family="Georgia", size=15, weight="bold"),
                     text_color="#FFFFFF").pack(side="left", padx=18, pady=13)

        # Scrollable body (fills remaining space)
        body = ctk.CTkScrollableFrame(container, fg_color="transparent",
                                       scrollbar_button_color=C["rule"],
                                       scrollbar_button_hover_color=C["muted"])
        body.grid(row=1, column=0, sticky="nsew")

        self._fill_left_body(body)

        # Sticky bottom area (fixed)
        bottom = ctk.CTkFrame(container, fg_color=C["panel"], corner_radius=0)
        bottom.grid(row=2, column=0, sticky="ew")

        ctk.CTkFrame(bottom, fg_color=C["rule"], height=1).pack(fill="x")

        self.run_btn = ctk.CTkButton(
            bottom, text="▶  Run Analysis", height=48,
            fg_color=C["dark"], hover_color="#333",
            text_color="#FFFFFF", corner_radius=0,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._run_analysis)
        self.run_btn.pack(fill="x")

        self.run_status = ctk.CTkLabel(
            bottom, text="Load a model · choose an image · run",
            font=ctk.CTkFont(size=11), text_color=C["muted"])
        self.run_status.pack(pady=(4, 10))

    def _fill_left_body(self, body):
        def section_hdr(lbl):
            ctk.CTkLabel(body, text=lbl,
                         font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
                         text_color=C["accent"]).pack(anchor="w", padx=16, pady=(14, 3))
            ctk.CTkFrame(body, fg_color=C["rule"], height=1).pack(fill="x", padx=16)

        # ── MODEL ──
        section_hdr("MODEL")
        self.model_path_var = tk.StringVar(value=DEFAULT_MODEL)
        ctk.CTkEntry(body, textvariable=self.model_path_var, height=30,
                     fg_color=C["bg"], border_color=C["rule"],
                     text_color=C["ink"], font=ctk.CTkFont(size=11)
                     ).pack(fill="x", padx=16, pady=(6, 4))
        br = ctk.CTkFrame(body, fg_color="transparent")
        br.pack(fill="x", padx=16, pady=(0, 4))
        ctk.CTkButton(br, text="Browse", width=100, height=30,
                      fg_color=C["bg"], hover_color=C["rule"],
                      border_width=1, border_color=C["rule"],
                      text_color=C["ink"], corner_radius=6,
                      command=self._browse_model).pack(side="left")
        ctk.CTkButton(br, text="Load  ↗", width=100, height=30,
                      fg_color=C["accent"], hover_color="#236B40",
                      text_color="#FFFFFF", corner_radius=6,
                      command=self._load_model).pack(side="right")
        self.model_status = ctk.CTkLabel(body, text="No model loaded",
                                          font=ctk.CTkFont(size=11), text_color=C["muted"])
        self.model_status.pack(anchor="w", padx=16, pady=(2, 4))

        # ── IMAGE ──
        section_hdr("IMAGE")
        self.drop_zone = ctk.CTkFrame(body, fg_color=C["bg"], corner_radius=10,
                                       border_width=2, border_color=C["rule"], height=155)
        self.drop_zone.pack(fill="x", padx=16, pady=(8, 6))
        self.drop_zone.pack_propagate(False)
        self.drop_hint = ctk.CTkLabel(self.drop_zone, text="♻\n\nClick to upload",
                                       font=ctk.CTkFont(size=13), text_color=C["muted"],
                                       justify="center")
        self.drop_hint.pack(expand=True)
        self.preview_lbl = ctk.CTkLabel(self.drop_zone, text="")
        self.drop_zone.bind("<Button-1>", lambda e: self._choose_image())
        self.drop_hint.bind("<Button-1>", lambda e: self._choose_image())

        ctk.CTkButton(body, text="📂  Choose Image", height=36,
                      fg_color=C["accent_pale"], hover_color="#B7E4C7",
                      text_color=C["accent"], border_width=1, border_color=C["accent_lt"],
                      corner_radius=8, font=ctk.CTkFont(size=13, weight="bold"),
                      command=self._choose_image).pack(fill="x", padx=16, pady=(0, 4))
        self.img_name_lbl = ctk.CTkLabel(body, text="No image selected",
                                          font=ctk.CTkFont(size=10), text_color=C["muted"])
        self.img_name_lbl.pack(anchor="w", padx=16)

        # ── LIME ──
        section_hdr("LIME SETTINGS")

        def srow(lbl, var, lo, hi, default):
            ctk.CTkLabel(body, text=lbl, font=ctk.CTkFont(size=11),
                         text_color=C["muted"]).pack(anchor="w", padx=16, pady=(8, 0))
            r = ctk.CTkFrame(body, fg_color="transparent")
            r.pack(fill="x", padx=16, pady=(2, 0))
            vl = ctk.CTkLabel(r, text=str(default), width=44,
                               font=ctk.CTkFont(family="Courier New", size=12, weight="bold"),
                               text_color=C["ink"])
            vl.pack(side="right")
            def upd(v): vl.configure(text=str(int(float(v))))
            ctk.CTkSlider(r, from_=lo, to=hi, variable=var, command=upd,
                          button_color=C["accent"], progress_color=C["accent_lt"],
                          fg_color=C["rule"]).pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.lime_samples_var  = ctk.IntVar(value=500)
        self.lime_features_var = ctk.IntVar(value=6)
        srow("Num Samples",  self.lime_samples_var,  100, 1500, 500)
        srow("Top Features", self.lime_features_var,  2,   15,   6)

        ctk.CTkFrame(body, fg_color="transparent", height=12).pack()

    # ─────────────────────────────────────────────
    # RIGHT PANEL
    # ─────────────────────────────────────────────
    def _show_right_placeholder(self):
        for w in self.right_panel.winfo_children():
            w.destroy()
        ph = ctk.CTkFrame(self.right_panel, fg_color=C["panel"], corner_radius=14,
                           border_width=1, border_color=C["rule"])
        ph.pack(fill="both", expand=True)
        ctk.CTkLabel(ph, text="♻", font=ctk.CTkFont(size=64), text_color=C["rule"]).pack(pady=(100, 0))
        ctk.CTkLabel(ph,
                     text="Load model  →  Choose image  →  Run Analysis",
                     font=ctk.CTkFont(family="Georgia", size=15),
                     text_color=C["muted"], justify="center").pack(pady=(14, 100))

    def _show_results(self, class_name, probs, lime_pil):
        for w in self.right_panel.winfo_children():
            w.destroy()

        meta     = CLASS_META.get(class_name, {"color": C["accent"], "pale": C["accent_pale"], "desc": ""})
        top_conf = probs[self.class_names.index(class_name)]
        pill_col = C["accent"] if top_conf > 0.75 else C["amber"] if top_conf > 0.5 else C["red"]

        scroll = ctk.CTkScrollableFrame(self.right_panel, fg_color="transparent",
                                         scrollbar_button_color=C["rule"],
                                         scrollbar_button_hover_color=C["muted"])
        scroll.pack(fill="both", expand=True)

        # ── Prediction header ──
        hc = ctk.CTkFrame(scroll, fg_color=C["panel"], corner_radius=12,
                           border_width=1, border_color=C["rule"])
        hc.pack(fill="x", pady=(0, 10))
        ctk.CTkFrame(hc, fg_color=meta["color"], height=5, corner_radius=0).pack(fill="x")

        hrow = ctk.CTkFrame(hc, fg_color="transparent")
        hrow.pack(fill="x", padx=22, pady=16)

        badge = ctk.CTkFrame(hrow, fg_color=meta["pale"], corner_radius=10, width=115)
        badge.pack(side="left", padx=(0, 18))
        badge.pack_propagate(False)
        ctk.CTkLabel(badge, text=class_name.upper(),
                     font=ctk.CTkFont(family="Georgia", size=16, weight="bold"),
                     text_color=meta["color"]).pack(expand=True, padx=8, pady=14)

        info = ctk.CTkFrame(hrow, fg_color="transparent")
        info.pack(side="left", fill="both", expand=True)
        ctk.CTkLabel(info, text="Predicted Class", font=ctk.CTkFont(size=11),
                     text_color=C["muted"]).pack(anchor="w")
        ctk.CTkLabel(info, text=class_name.title(),
                     font=ctk.CTkFont(family="Georgia", size=26, weight="bold"),
                     text_color=C["ink"]).pack(anchor="w")
        ctk.CTkLabel(info, text=meta["desc"], font=ctk.CTkFont(size=12),
                     text_color=C["muted"]).pack(anchor="w")

        ctk.CTkLabel(hrow, text=f"{top_conf*100:.1f}%",
                     font=ctk.CTkFont(family="Courier New", size=30, weight="bold"),
                     text_color=pill_col, fg_color=C["bg"],
                     corner_radius=12, padx=18, pady=10).pack(side="right")

        # ── Images side by side ──
        vis = ctk.CTkFrame(scroll, fg_color="transparent")
        vis.pack(fill="x", pady=(0, 10))
        vis.columnconfigure((0, 1), weight=1)

        for col, (title, sub, use_lime) in enumerate([
            ("Original Image",   "",                               False),
            ("LIME Explanation", "Green = supports · Red = opposes", True),
        ]):
            card = ctk.CTkFrame(vis, fg_color=C["panel"], corner_radius=12,
                                 border_width=1, border_color=C["rule"])
            card.grid(row=0, column=col, sticky="nsew",
                      padx=(0, 5) if col == 0 else (5, 0))

            th = ctk.CTkFrame(card, fg_color="transparent")
            th.pack(fill="x", padx=16, pady=(12, 6))
            ctk.CTkLabel(th, text=title,
                         font=ctk.CTkFont(family="Georgia", size=13, weight="bold"),
                         text_color=C["ink"]).pack(side="left")
            if sub:
                ctk.CTkLabel(th, text=sub, font=ctk.CTkFont(size=9),
                             text_color=C["muted"], fg_color=C["bg"],
                             corner_radius=6, padx=6, pady=2).pack(side="right")
            ctk.CTkFrame(card, fg_color=C["rule"], height=1).pack(fill="x", padx=16)

            pil_src = lime_pil if use_lime else Image.open(self.current_img_path).convert("RGB")
            pil_src.thumbnail((380, 280), Image.LANCZOS)
            ctk_img = ctk.CTkImage(pil_src, size=pil_src.size)
            lbl = ctk.CTkLabel(card, text="", image=ctk_img)
            lbl._ctk_img = ctk_img
            lbl.pack(expand=True, pady=14)

        # ── Confidence bars ──
        cc = ctk.CTkFrame(scroll, fg_color=C["panel"], corner_radius=12,
                           border_width=1, border_color=C["rule"])
        cc.pack(fill="x")
        ctk.CTkLabel(cc, text="Class Confidence Distribution",
                     font=ctk.CTkFont(family="Georgia", size=14, weight="bold"),
                     text_color=C["ink"]).pack(anchor="w", padx=20, pady=(14, 4))
        ctk.CTkFrame(cc, fg_color=C["rule"], height=1).pack(fill="x", padx=20)

        bars = ctk.CTkFrame(cc, fg_color="transparent")
        bars.pack(fill="x", padx=20, pady=10)

        for idx in np.argsort(probs)[::-1]:
            cls   = self.class_names[idx]
            conf  = probs[idx]
            m     = CLASS_META.get(cls, {"color": C["muted"], "pale": C["bg"]})
            top   = (cls == class_name)

            row = ctk.CTkFrame(bars, fg_color=m["pale"] if top else "transparent",
                                corner_radius=6)
            row.pack(fill="x", pady=2, padx=2)

            ctk.CTkLabel(row, text=("★  " if top else "     ") + cls.title(),
                         width=120, anchor="w",
                         font=ctk.CTkFont(size=12, weight="bold" if top else "normal"),
                         text_color=m["color"] if top else C["ink"]
                         ).pack(side="left", padx=(10, 0), pady=8)

            bg = ctk.CTkFrame(row, fg_color=C["rule"], height=14, corner_radius=6)
            bg.pack(side="left", fill="x", expand=True, padx=8, pady=8)
            ctk.CTkFrame(bg, fg_color=m["color"], height=14, corner_radius=6,
                          width=max(4, int(conf * 340))).place(x=0, y=0)

            ctk.CTkLabel(row, text=f"{conf*100:5.1f}%", width=58, anchor="e",
                         font=ctk.CTkFont(family="Courier New", size=12,
                                           weight="bold" if top else "normal"),
                         text_color=m["color"] if top else C["muted"]
                         ).pack(side="right", padx=10)

        ctk.CTkFrame(scroll, fg_color="transparent", height=16).pack()

    # ─────────────────────────────────────────────
    # ACTIONS
    # ─────────────────────────────────────────────
    def _browse_model(self):
        p = filedialog.askopenfilename(
            title="Select Trained Model",
            filetypes=[("Keras/H5", "*.keras *.h5"), ("All", "*.*")])
        if p:
            self.model_path_var.set(p)

    def _load_model(self):
        path = self.model_path_var.get().strip()
        if not os.path.exists(path):
            messagebox.showerror("Not Found", f"Model not found:\n{path}")
            return
        self.model_status.configure(text="Loading…", text_color=C["amber"])
        self.update()

        def _do():
            try:
                import tensorflow as tf
                m = tf.keras.models.load_model(path)
                self.model = m
                cn = os.path.join(os.path.dirname(path), "class_names.json")
                if os.path.exists(cn):
                    with open(cn) as f:
                        self.class_names = json.load(f)
                else:
                    self.class_names = DEFAULT_CLASSES
                name = os.path.basename(path)
                self.after(0, lambda: (
                    self.model_status.configure(text=f"✓  {name}", text_color=C["accent"]),
                    self.model_pill.configure(text=f"  ✓ {name}", text_color=C["accent_lt"]),
                    self.run_status.configure(text="Model ready · choose an image", text_color=C["muted"])
                ))
            except Exception as e:
                err = str(e)
                self.after(0, lambda: (
                    self.model_status.configure(text=f"Error loading model", text_color=C["red"]),
                    messagebox.showerror("Load Failed", f"Could not load model:\n{err}")
                ))
        threading.Thread(target=_do, daemon=True).start()

    def _choose_image(self):
        p = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"), ("All", "*.*")])
        if not p:
            return
        self.current_img_path = p
        self.img_name_lbl.configure(text=os.path.basename(p), text_color=C["ink"])
        try:
            img = Image.open(p).convert("RGB")
            img.thumbnail((230, 145), Image.LANCZOS)
            ctk_img = ctk.CTkImage(img, size=img.size)
            self.drop_hint.pack_forget()
            self.preview_lbl.configure(image=ctk_img, text="")
            self.preview_lbl._ctk_img = ctk_img
            self.preview_lbl.pack(expand=True, pady=6)
            self.drop_zone.configure(border_color=C["accent"])
            self.run_status.configure(text="Image ready · click  ▶ Run Analysis",
                                       text_color=C["accent"])
        except Exception as e:
            messagebox.showerror("Image Error", str(e))

    def _run_analysis(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Please load a model first.")
            return
        if self.current_img_path is None:
            messagebox.showwarning("No Image", "Please choose an image first.")
            return

        self.run_btn.configure(state="disabled", text="Analyzing…")
        self.run_status.configure(text="Running inference + LIME  (10–30 s)…",
                                   text_color=C["amber"])
        self.update()

        def _worker():
            try:
                IMG_SIZE = (224, 224)
                img = Image.open(self.current_img_path).convert("RGB")
                arr = np.array(img.resize(IMG_SIZE), dtype=np.float32)
                preds = self.model.predict(np.expand_dims(arr, 0), verbose=0)[0]
                pred_cls = self.class_names[int(np.argmax(preds))]
                lime_pil = self._compute_lime(arr)
                self.after(0, lambda: self._on_done(pred_cls, preds, lime_pil))
            except Exception:
                err = traceback.format_exc()
                print(err)
                self.after(0, lambda: self._on_error(err))

        threading.Thread(target=_worker, daemon=True).start()

    def _compute_lime(self, img_arr):
        try:
            from lime import lime_image
            from skimage.segmentation import mark_boundaries

            def predict_fn(images):
                imgs = np.array(images, dtype=np.float32)
                return self.model.predict(imgs, verbose=0)

            explainer = lime_image.LimeImageExplainer()
            exp = explainer.explain_instance(
                img_arr.astype(np.uint8), predict_fn,
                top_labels=1, hide_color=0,
                num_samples=int(self.lime_samples_var.get()),
                random_seed=42,
            )
            top_label = exp.top_labels[0]
            temp, mask = exp.get_image_and_mask(
                top_label, positive_only=False,
                num_features=int(self.lime_features_var.get()),
                hide_rest=False,
            )
            base    = np.array(Image.fromarray(img_arr.astype(np.uint8)).resize((224, 224)))
            overlay = np.zeros_like(base, dtype=np.float32)
            overlay[mask == 1]  = [0, 210, 100]
            overlay[mask == -1] = [220, 40,  40]
            blended = (base * 0.60 + overlay * 0.40).clip(0, 255).astype(np.uint8)
            bordered = mark_boundaries(blended / 255.0, mask,
                                        color=(1, 1, 1), outline_color=(0.1, 0.1, 0.1))
            return Image.fromarray((bordered * 255).astype(np.uint8))

        except ImportError:
            print("LIME not installed — using Grad-CAM fallback. pip install lime scikit-image")
            return self._grad_cam_fallback(img_arr)
        except Exception as e:
            print(f"LIME error: {e} — using Grad-CAM fallback")
            return self._grad_cam_fallback(img_arr)

    def _grad_cam_fallback(self, img_arr):
        try:
            import tensorflow as tf
            last_conv = None
            for layer in reversed(self.model.layers):
                if 'conv' in layer.name.lower() and hasattr(layer, 'filters'):
                    last_conv = layer
                    break
            if last_conv is None:
                raise ValueError("No conv layer")
            grad_model = tf.keras.Model(
                inputs=self.model.inputs,
                outputs=[last_conv.output, self.model.output])
            inp = tf.cast(np.expand_dims(img_arr, 0), tf.float32)
            with tf.GradientTape() as tape:
                conv_out, preds = grad_model(inp)
                loss = preds[:, tf.argmax(preds[0])]
            grads   = tape.gradient(loss, conv_out)[0]
            weights = tf.reduce_mean(grads, axis=(0, 1))
            cam     = tf.reduce_sum(tf.multiply(weights, conv_out[0]), axis=-1).numpy()
            cam     = np.maximum(cam, 0)
            cam    /= (cam.max() + 1e-8)
            import cv2
            cam_r   = cv2.resize(cam, (224, 224))
            heat    = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heat    = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
            base    = img_arr.astype(np.uint8)
            blended = (base * 0.55 + heat * 0.45).clip(0, 255).astype(np.uint8)
            return Image.fromarray(blended)
        except Exception as e:
            print(f"Grad-CAM error: {e}")
            return Image.open(self.current_img_path).convert("RGB")

    def _on_done(self, cls, probs, lime_pil):
        self.run_btn.configure(state="normal", text="▶  Run Analysis")
        self.run_status.configure(text=f"✓  Classified as  '{cls}'", text_color=C["accent"])
        self._show_results(cls, probs, lime_pil)

    def _on_error(self, err):
        self.run_btn.configure(state="normal", text="▶  Run Analysis")
        self.run_status.configure(text="Analysis failed — see details", text_color=C["red"])
        messagebox.showerror("Analysis Failed",
                              f"Common causes:\n"
                              f"• TensorFlow not installed\n"
                              f"• Model input shape mismatch (expects 224×224)\n\n"
                              f"Error:\n{err[:500]}")

    def _try_autoload(self):
        if os.path.exists(DEFAULT_MODEL):
            self.after(700, self._load_model)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("TrashNet Classifier GUI")
    print("pip install customtkinter tensorflow pillow numpy lime scikit-image scikit-learn")
    TrashNetApp().mainloop()