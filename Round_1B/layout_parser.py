import os
import pickle

import fitz                         
import numpy as np               
from sklearn.cluster import KMeans 
import pandas as pd
import re


def merge_section_numbers(spans):
    merged = []
    i = 0
    while i < len(spans):
        text = spans[i]["text"]
       
        if re.fullmatch(r'\d+(\.\d+)*', text) and i + 1 < len(spans):
            spans[i + 1]["text"] = text + " " + spans[i + 1]["text"]
           
            i += 1  
        else:
            merged.append(spans[i])
        i += 1
    return merged

def is_probably_body(text, size, median_size, gap):
   
    if size < median_size * 1.05 and gap < 8:
        return True

    if text.count(' ') > 6 or text.endswith('.'):
        return True
    return False

def is_heading_noise(text):
    text = text.strip()
    
    if re.fullmatch(r'\d+\.?', text):
        return True
   
    if re.fullmatch(r'(\d{1,2}[./-]){2}\d{2,4}', text):  
        return True
    if re.fullmatch(r'\d{4}[./-]\d{1,2}[./-]\d{1,2}', text):  
        return True
    if re.fullmatch(r'\d{1,2}/\d{1,2}/\d{2,4}', text): 
        return True
    if re.fullmatch(r'\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}', text): 
        return True
    if re.fullmatch(r'[A-Za-z]{3,9}\s+\d{4}', text): 
        return True
    
    if re.fullmatch(r'[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}', text): 
        return True

    if len(text) < 3:
        return True
    return False

def is_noise_span(text):
    text = text.strip()
   
    if re.match(r'^(\d{1,2}([.-]\d{1,2}){0,2})$', text):
        return True
    if re.match(r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}$', text, re.IGNORECASE):
        return True
    return False

def normalize(text):
    import re
    text = text.lower().strip()
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r'[^\w\s]', '', text)  
    return text

def load_fallback(path="model/gbt_model.pkl"):

    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    return None

def detect_headings(pdf_path, fallback=None):



    doc = fitz.open(pdf_path)
    spans = []

    for pnum, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    txt = span["text"].strip()
                    if not txt:
                        continue
                    x0, y0, x1, y1 = span["bbox"]
                    page_width = page.rect.width
                    centered = abs((x0 + x1)/2 - page_width/2) < (page_width * 0.15)  
                    spans.append({
                        "page": pnum,
                        "text": txt,
                        "size": round(span["size"], 1),
                        "font": span["font"],
                        "x0": x0, "x1": x1,
                        "y0": y0, "y1": y1,
                        "gap": 0,
                        "is_bold": int("Bold" in span["font"]),
                        "centered": int(centered)
                    })

    if not spans:
        return []

    spans = merge_section_numbers(spans)

    spans.sort(key=lambda s: (s["page"], s["y0"]))
    for i, s in enumerate(spans):
        if i == 0 or spans[i-1]["page"] != s["page"]:
            s["gap"] = s["y0"]
        else:
            s["gap"] = max(0, s["y0"] - spans[i-1]["y1"])

    sizes = sorted({s["size"] for s in spans})
    k = min(4, len(sizes))
    arr = np.array(sizes).reshape(-1, 1)
    km = KMeans(n_clusters=k, random_state=0).fit(arr)
    centers = km.cluster_centers_.flatten()
    order = np.argsort(centers)[::-1]
    tier_map = {size: int(order.tolist().index(label))
                for size, label in zip(sizes, km.labels_)}


    median_size = np.median([s["size"] for s in spans]) if spans else 1.0

    
    for s in spans:
        s["norm_text"] = normalize(s["text"])

    spans = [s for s in spans if not is_noise_span(s["text"])]

    candidates = []
    for s in spans:
        tier = tier_map[s["size"]]
        lvl = None
        
        if tier == 0:
            lvl = "H1"
        elif tier == 1:
            lvl = "H2"
        elif tier == 2:
            lvl = "H3"
        elif tier == 3:
            lvl = "H4"

     
        if lvl is None and fallback:
            feature_order = [
                "size", "gap", "width", "text_len",
                "is_bold", "ends_with_colon", "has_digits", "is_title_case"
            ]
            feature_dict = {
                "size": s["size"],
                "gap": s["gap"],
                "width": s["x1"] - s["x0"],
                "text_len": len(s["norm_text"]),
                "is_bold": int("Bold" in s["font"]),
                "ends_with_colon": int(s["text"].endswith(":")),
                "has_digits": int(any(c.isdigit() for c in s["norm_text"])),
                "is_title_case": int(s["text"].istitle()),
          
            }
            feat_df = pd.DataFrame([feature_dict], columns=feature_order)
            pred = fallback.predict(feat_df)[0]
            lvl = pred if pred != "O" else None

        if lvl:
            candidates.append({**s, "level": lvl})

    if not candidates:
        return []

    merged = []
    cur = candidates[0]
    for nxt in candidates[1:]:
        same_pg = nxt["page"] == cur["page"]
        close_y = abs(nxt["y0"] - cur["y0"]) < 6
        close_x = abs(nxt["x0"] - cur["x0"]) < 20
        if same_pg and close_y and close_x and nxt["level"] == cur["level"]:
           
            cur["text"] += " " + nxt["text"]
            cur["y1"], cur["x1"] = nxt["y1"], nxt["x1"]
        else:
            merged.append(cur)
            cur = nxt
    merged.append(cur)

    final_headings = [
    h for h in merged
    if not is_heading_noise(h["text"])
    and not is_probably_body(h["text"], h["size"], median_size, h["gap"])
    
    ]

    seen = set()
    unique_headings = []
    for h in final_headings:
        key = (h["level"], h["text"].strip().lower())
        if key not in seen:
            unique_headings.append(h)
            seen.add(key)

    return [
        {"level": m["level"], "text": m["text"].strip(), "page": m["page"], "y0": m["y0"]}
        for m in unique_headings
    ]