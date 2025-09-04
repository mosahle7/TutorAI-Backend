# # import pdfplumber
# # import os

# # def pdf_to_text(pdf_path):
# #     save_dir = "pdf_data"
# #     text = ""

# #     os.makedirs(save_dir, exist_ok=True)

# #     with pdfplumber.open(pdf_path) as pdf:
# #         for page in pdf.pages:
# #             text+=page.extract_text() or ""
    
# #     base_name = os.path.splitext(os.path.basename(pdf_path))[0]
# #     txt_path = os.path.join(save_dir, f"{base_name}.text")

# #     with open(txt_path, "w", encoding="utf-8") as f:
# #         f.write(text)
    
# #     print("Saved text file!")
# #     return txt_path

# # pdf_to_text("/root/TutorAI/backend/app/data/ComputerApplicationCommerce1Year_removed.pdf")


# import pdfplumber
# import os, re
# def extract_text_from_pdf(path):
#     save_dir = "pdf_data3"
#     os.makedirs(save_dir, exist_ok=True)
#     all_text = []
#     with pdfplumber.open(path) as pdf:
#         for page in pdf.pages:
#             words = page.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False)
            
#             # Split into left and right columns based on x coordinate
#             midpoint = page.width / 2
#             left_col = [w for w in words if w["x0"] < midpoint]
#             right_col = [w for w in words if w["x0"] >= midpoint]
            
#             # Sort each column by y (top to bottom), then x (left to right)
#             left_col_sorted = sorted(left_col, key=lambda w: (w["top"], w["x0"]))
#             right_col_sorted = sorted(right_col, key=lambda w: (w["top"], w["x0"]))
            
#             # Convert back into text
#             left_text = " ".join(w["text"] for w in left_col_sorted)
#             right_text = " ".join(w["text"] for w in right_col_sorted)
            
#             page_text = left_text + "\n\n" + right_text
#             all_text.append(page_text)
#     text =  "\n\n".join(all_text)

#     # Remove single newlines within paragraphs
#     text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
#     # Normalize multiple spaces
#     text = re.sub(r'\s+', ' ', text)
#     text = text.strip()

#     base_name = os.path.splitext(os.path.basename(path))[0]
#     txt_path = os.path.join(save_dir, f"{base_name}.text")

#     with open(txt_path, "w", encoding="utf-8") as f:
#         f.write(text)
    

# extract_text_from_pdf("/root/TutorAI/backend/app/data/ComputerApplicationCommerce1Year_removed.pdf")

import fitz
import numpy as np
from sklearn.cluster import KMeans
import os, re

def extract_text_preserve_layout(pdf_path, output_txt_path):
    doc = fitz.open(pdf_path)
    all_text = []

    for page in doc:
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        blocks = [b for b in blocks if b[6] == 0]  # keep only text blocks

        # Get x positions (left edge of block)
        x_positions = np.array([[b[0]] for b in blocks])

        # Default: assume single column
        column_labels = np.zeros(len(blocks), dtype=int)

        # Try detecting 2 columns via KMeans
        if len(blocks) > 6:  # only try if enough blocks
            try:
                kmeans = KMeans(n_clusters=2, n_init="auto").fit(x_positions)
                column_labels = kmeans.labels_

                # Check distance between clusters
                centers = sorted(kmeans.cluster_centers_.flatten())
                if abs(centers[1] - centers[0]) < 50:  
                    # very close, treat as one column
                    column_labels = np.zeros(len(blocks), dtype=int)
            except:
                pass  # fallback to single column

        # Group blocks by column
        col_blocks = {c: [] for c in set(column_labels)}
        for label, block in zip(column_labels, blocks):
            col_blocks[label].append(block)

        # Sort: left column first (min x), then right
        sorted_cols = sorted(col_blocks.items(), key=lambda kv: min(b[0] for b in kv[1]))

        for _, col in sorted_cols:
            # Sort top-to-bottom inside each column
            for b in sorted(col, key=lambda b: (b[1], b[0])):
                text = b[4].strip()
                # 🔑 Clean up line breaks inside block (paragraph stays intact)
                text = re.sub(r"\n+", " ", text).strip()
                if text:
                    all_text.append(text)

    # Join blocks with double newline to separate paragraphs
    clean_text = "\n\n".join(all_text)

    # Save output
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(clean_text)

    return clean_text


extract_text_preserve_layout("/root/TutorAI/backend/app/data/ComputerApplicationCommerce1Year_removed.pdf", "/root/TutorAI/backend/app/pdf_data/ComputerApplicationCommerce1Year_removed.txt")