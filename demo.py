import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from deep_translator import GoogleTranslator
from tkinter import filedialog, Label, Button, Entry, Frame, Canvas, Scrollbar, Checkbutton, BooleanVar, Scale, filedialog
from PIL import Image, ImageTk
import base64
import openai
import os
import kdbai_client as kdbai
import time
from IPython.display import display
from openai import OpenAI
from scipy.spatial.distance import cosine

# Set up KDB.AI connection
KDBAI_ENDPOINT = "https://cloud.kdb.ai/instance/6wk1w0ql39"
KDBAI_API_KEY = "ba14da401f-LcWyhQFqvY38vZ/Smmqc/xdoLYxeDIVgPkgKWuz1X/frlwdJXkmxUpv9YZ+TW/nisTrUobjZ+tZCbgDa"
session = kdbai.Session(api_key=KDBAI_API_KEY, endpoint=KDBAI_ENDPOINT)
db = session.database('default')

os.environ["OPENAI_API_KEY"] = ""
openai.api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI()

selected_image_path = None

# Encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Summarize image
def image_summarize(img_base64,prompt):
    ''' Image summary '''
    for _ in range(3):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                    },
                    },
                ],
                }
            ],
            max_tokens=150,
        )
        content = response.choices[0].message.content

        if "I'm sorry" not in content:
            return content
        time.sleep(3)
    return content

# Convert description to embedding
def text_to_embedding(text):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model="text-embedding-3-small").data[0].embedding

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings."""
    return 1 - cosine(embedding1, embedding2)

# Function to translate text to English
def translate_to_english(vietnamese_text):
    translation = GoogleTranslator(source='auto', target='en').translate(vietnamese_text)
    return translation

# Function to translate text to Vietnamese
def translate_to_vietnamese(english_text):
    translation = GoogleTranslator(source='auto', target='vi').translate(english_text)
    return translation

# Main function to generate query vector from image
def image_to_query_vector(image_path, prompt):
    img_base64 = encode_image(image_path)
    description = image_summarize(img_base64, prompt)
    query_vector = text_to_embedding(description)
    return query_vector

# Reset function
def reset_app():
    global selected_image_path
    selected_image_path = None
    entry_description.delete(0, tk.END)
    lbl_image.config(image='')
    lbl_image.image = None
    lbl_error_message.config(text="")
    lbl_upload_message.config(text="")

# Handle browsing image
def browse_image():
    global selected_image_path
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if filepath:
        selected_image_path = filepath
        img = Image.open(filepath)
        img.thumbnail((250, 250))
        img = ImageTk.PhotoImage(img)
        lbl_image.config(image=img)
        lbl_image.image = img

# Optimize Image Size Before Uploading
def compress_image(image_path, quality=70):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.save("temp_compressed.jpg", format="JPEG", quality=quality)
    return "temp_compressed.jpg"

# Define the function to calculate sizes of each component
def calculate_component_sizes(image_base64, embedding, metadata=""):
    image_size = len(image_base64.encode('utf-8'))
    embedding_size = len(str(embedding).encode('utf-8'))
    metadata_size = len(metadata.encode('utf-8'))
    total_size = image_size + embedding_size + metadata_size
    return image_size, embedding_size, metadata_size, total_size

# Function to upload images to the database
def upload_images_to_database(image_paths):
    max_size = 10485760  # 10 MB limit
    batch = []
    batch_size = 0

    for image_path in image_paths:
        # Compress and encode image
        compressed_image_path = compress_image(image_path)
        img_base64 = encode_image(compressed_image_path)
        prompt = "Summarize the content of the image."
        query_vector = image_to_query_vector(image_path, prompt)

        # Create entry with image data and embedding
        entry = {
            "path": image_path,        # File path
            "media_type": "image",     # Type of media
            "text": "",                # Optional metadata or description
            "embeddings": query_vector # Embedding vector
        }

        # Calculate the serialized size of each component
        image_size, embedding_size, metadata_size, entry_size = calculate_component_sizes(
            img_base64, query_vector, ""
        )

        # Check if adding this entry exceeds max batch size
        if batch_size + entry_size > max_size:
            db.table("multi_modal_demo").insert(batch)  # Insert current batch
            batch = []  # Reset batch
            batch_size = 0

        # Append entry to batch and update batch size
        batch.append(entry)
        batch_size += entry_size

    # Insert the last batch if it's not empty
    if batch:
        db.table("multi_modal_demo").insert(batch)

    lbl_upload_message.config(text="Thêm ảnh vào database thành công!")


def open_file_dialog():
    image_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.png")])
    if image_paths:
        upload_images_to_database(image_paths)

# Function to handle option selection and toggle visibility
def handle_option_selection():
    use_image = use_image_var.get()
    use_description = use_description_var.get()
    if use_image:
        btn_browse.pack()
    else:
        btn_browse.pack_forget()
        similarity_frame.pack_forget()
    if use_description:
        description_frame.pack()
    else:
        description_frame.pack_forget()
        similarity_frame.pack_forget()
    if use_image and use_description:
        btn_browse.pack()
        description_frame.pack()
        similarity_frame.pack()
    else:
        hidden_frame.pack_forget()

# Query function
def handle_query():
    retrieved_data_for_RAG = []
    display_items = []
    similarity_scores = []
    use_image = use_image_var.get()
    use_description = use_description_var.get()

    if not use_image and not use_description:
        lbl_error_message.config(text="Xin hãy chọn ít nhất một phương thức để thực thi")
        return
    else:
        lbl_error_message.config(text="")

    start_time = time.time()

    if use_image and use_description:
        if not selected_image_path:
            lbl_error_message.config(text="Xin hãy chọn một ảnh để thực thi")
            return
        if not entry_description.get():
            lbl_error_message.config(text="Xin hãy nhập mô tả để thực thi")
            return

        img_base64 = encode_image(selected_image_path)
        description_vietnamese = entry_description.get()
        description_english = translate_to_english(description_vietnamese)
        
        image_embedding = image_to_query_vector(selected_image_path, description_english)
        text_embedding = text_to_embedding(description_english)

        similarity_threshold = similarity_slider.get() / 100.0
        similarity_score = calculate_similarity(image_embedding, text_embedding)

        if similarity_score < similarity_threshold:
            lbl_error_message.config(text="Ảnh và mô tả không tương đồng, xin vui lòng kiểm tra lại.")
            return

        prompt = description_english
        query_vector = image_embedding

    elif use_image:
        if not selected_image_path:
            lbl_error_message.config(text="Xin hãy chọn một ảnh để thực thi")
            return

        img_base64 = encode_image(selected_image_path)
        prompt = "Summarize the content of the image."
        query_vector = image_to_query_vector(selected_image_path, prompt)

    elif use_description:
        if not entry_description.get():
            lbl_error_message.config(text="Xin hãy nhập mô tả để thực thi")
            return

        description_vietnamese = entry_description.get()
        description_english = translate_to_english(description_vietnamese)
        prompt = description_english
        query_vector = text_to_embedding(description_english)

    table = db.table("multi_modal_demo")
    results = table.search({'vectorIndex': [query_vector]}, n=5)

    for index, row in results[0].iterrows():
        if row[2] == 'image':
            retrieved_data_for_RAG.append(row[3])
            image = Image.open(row[1])
            display_items.append(image)
            # Calculate similarity for each image
            image_emb = image_to_query_vector(row[1], prompt)
            score = calculate_similarity(query_vector, image_emb)
            similarity_scores.append(score)
        elif row[2] == 'text':
            retrieved_data_for_RAG.append(row[3])

    rag_prompt = prompt if use_description else "Answer based on the provided image content."
    response = RAG(retrieved_data_for_RAG, rag_prompt)
    vietnamese_RAG = translate_to_vietnamese(response)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Pass similarity scores to the display function
    display_results(display_items, vietnamese_RAG, elapsed_time, similarity_scores)

# Display result
# def display_results(images, rag_text, processing_time, similarity_scores):
#     # Create a new Toplevel window
#     result_window = tk.Toplevel(root)
#     result_window.title("Kết quả thực thi")
#     result_window.geometry("700x800")

#     # Create a Canvas and Scrollbar for scrollable content
#     canvas = tk.Canvas(result_window)
#     scrollbar = tk.Scrollbar(result_window, orient="vertical", command=canvas.yview)
#     scrollable_frame = tk.Frame(canvas)

#     # Configure scrollbar and canvas
#     scrollable_frame.bind(
#         "<Configure>",
#         lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
#     )
#     canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
#     canvas.configure(yscrollcommand=scrollbar.set)

#     # Pack the canvas and scrollbar
#     canvas.pack(side="left", fill="both", expand=True)
#     scrollbar.pack(side="right", fill="y")

#     # Create two columns within the scrollable frame
#     image_frame = tk.Frame(scrollable_frame)
#     image_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
#     text_frame = tk.Frame(scrollable_frame)
#     text_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
#     scrollable_frame.grid_columnconfigure(0, weight=1)
#     scrollable_frame.grid_columnconfigure(1, weight=1)

#     # Loop through images and similarity scores
#     # Display images with similarity scores in the left frame
#     has_results = False

#     for index, img in enumerate(images):
#         if similarity_scores[index] > 0.5:
#             has_results = True
#             img_resized = img.resize((250, int(250 * img.height / img.width)))  # Resize for consistent display
#             tk_img = ImageTk.PhotoImage(img_resized)

#             lbl_img = tk.Label(image_frame, image=tk_img)
#             lbl_img.image = tk_img  # Keep a reference to avoid garbage collection
#             lbl_img.pack(pady=5)

#             # Display similarity score below the image
#             score_label = tk.Label(
#                 image_frame,
#                 text=f"Độ tương đồng: {similarity_scores[index]:.2f}",
#                 fg="white"
#             )
#             score_label.pack()

#     # Check if any results were found
#     if has_results:
#         # Display the RAG text in the right frame
#         lbl_rag_text = tk.Label(
#             text_frame,
#             text=rag_text,
#             wraplength=350,
#             anchor="nw"
#         )
#         lbl_rag_text.pack(pady=10, padx=5)
#     else:
#         # Display "No results" message in the main frame
#         no_results_label = tk.Label(
#             image_frame,
#             text="Không có kết quả phù hợp",
#             fg="red",
#             font=("Helvetica", 14, "bold")
#         )
#         no_results_label.pack()
#     # Display the processing time below both frames
#     time_label = tk.Label(
#         scrollable_frame,
#         text=f"Thời gian xử lý: {processing_time:.2f} giây",
#         fg="white",
#         anchor="w"
#     )
#     time_label.grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")
#     time_label.grid(row=1, column=1, columnspan=2, pady=10, sticky="ew")

#     # Adjust layout to fill space
#     image_frame.grid(row=0, column=0, sticky="nsew")
#     text_frame.grid(row=0, column=1, sticky="nsew")
#     scrollable_frame.grid_rowconfigure(0, weight=1)



def display_results(images, rag_text, processing_time, similarity_scores):
    # Create a new Toplevel window
    result_window = tk.Toplevel(root)
    result_window.title("Kết quả thực thi")
    result_window.geometry("700x800")

    # Create a Canvas and Scrollbar for scrollable content
    canvas = tk.Canvas(result_window)
    scrollbar = tk.Scrollbar(result_window, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    # Configure scrollbar and canvas
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Pack the canvas and scrollbar
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Create two columns within the scrollable frame
    image_frame = tk.Frame(scrollable_frame)
    image_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

    text_frame = tk.Frame(scrollable_frame)
    text_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

    # Add weight to columns for resizing
    scrollable_frame.grid_columnconfigure(0, weight=1)
    scrollable_frame.grid_columnconfigure(1, weight=1)

    for index, img in enumerate(images):
        img_resized = img.resize((250, int(250 * img.height / img.width)))  # Resize for consistent display
        tk_img = ImageTk.PhotoImage(img_resized)

        lbl_img = tk.Label(image_frame, image=tk_img)
        lbl_img.image = tk_img  # Keep a reference to avoid garbage collection
        lbl_img.pack(pady=5)

        # Display similarity score below the image
        score_label = tk.Label(
            image_frame,
            text=f"Độ tương đồng: {similarity_scores[index]:.2f}",
            fg="white"
        )
        score_label.pack()

    # Display the RAG text in the right frame
    lbl_rag_text = tk.Label(
        text_frame,
        text=rag_text,
        wraplength=350,
        # justify="left",
        anchor="nw"
    )
    lbl_rag_text.pack(pady=10, padx=5)

    # Display the processing time below both frames
    time_label = tk.Label(
        scrollable_frame,
        text=f"Thời gian xử lý: {processing_time:.2f} giây",
        fg="white",
        anchor="w"
    )
    time_label.grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")
    time_label.grid(row=1, column=1, columnspan=2, pady=10, sticky="ew")

    # Adjust layout to fill space
    image_frame.grid(row=0, column=0, sticky="nsew")
    text_frame.grid(row=0, column=1, sticky="nsew")
    scrollable_frame.grid_rowconfigure(0, weight=1)

# RAG function
def RAG(retrieved_data, prompt):
    messages = ""
    messages += prompt + "\n"
    if retrieved_data:
        for data in retrieved_data:
            messages += data + "\n"
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": messages},
            ],
            }
        ],
        max_tokens=300,
    )
    content = response.choices[0].message.content
    return content

#---------------------------------------------------------------------------------------------------------------------#
# Application
root = tk.Tk()
root.title("A Fewshot Multi-Modal RAG")
root.geometry("800x600")

# Welcoming Title
logo_image = Image.open("logo.jpg")
logo_image = logo_image.resize((50, 50), Image.LANCZOS)  # Resize the logo
tk_logo = ImageTk.PhotoImage(logo_image)

title_frame = Frame(root)
title_frame.pack(side="top", pady=20)

inner_frame = Frame(title_frame)
inner_frame.pack()

logo_label = Label(inner_frame, image=tk_logo)
logo_label.pack(side="left", padx=(0, 5))

lbl_title = Label(
    inner_frame,
    text="WELCOME TO\nIntelliCulture: A Fewshot Multi-Modal RAG Approach for\nIntangible Cultural Heritage Analysis System",
    font=("Helvetica", 14, "italic"),
    justify="center",
    wraplength=650
)
lbl_title.pack(side="left")

# Main frame to hold both image selection and description input
main_frame = Frame(root)
hidden_frame = Frame(root)

# Option checkboxes
checkbox_frame = Frame(main_frame)
use_image_var = tk.BooleanVar()
use_description_var = tk.BooleanVar()
chk_use_image = Checkbutton(checkbox_frame, text="Dùng ảnh", variable=use_image_var, command=handle_option_selection)
chk_use_image.pack(anchor="center", side="left", padx=5)
chk_use_description = Checkbutton(checkbox_frame, text="Dùng mô tả", variable=use_description_var, command=handle_option_selection)
chk_use_description.pack(anchor="center", side="right", padx=5)
checkbox_frame.pack()

# Image selection button
btn_browse = Button(main_frame, text="Chọn ảnh", command=browse_image)
btn_browse.pack(pady=10)  # Center-align the image selection button
btn_browse.pack_forget()

lbl_image = Label(main_frame)
lbl_image.pack(pady=10)

# Description input frame inside main_frame
description_frame = Frame(main_frame)
lbl_description_prompt = Label(description_frame, text="Hãy nhập mô tả")
lbl_description_prompt.pack(pady=10)
entry_description = Entry(description_frame, width=50)
entry_description.pack(pady=10)
description_frame.pack_forget()

# Add a slider for adjusting similarity threshold
similarity_frame = Frame(main_frame)
similarity_label = Label(similarity_frame, text="Hãy điều chỉnh giới hạn của độ tương đồng cho ảnh và mô tả")
similarity_label.pack()
similarity_slider = Scale(similarity_frame, from_=0, to=100, orient="horizontal")
similarity_slider.set(50)  # Default to 50%
similarity_slider.pack()
similarity_frame.pack_forget()
hidden_frame.pack(side='bottom')

# Frame for Query and Reset buttons
query_reset_frame = Frame(main_frame)
query_reset_frame.pack(pady=10)

# Query button
btn_query = Button(query_reset_frame, text="Truy vấn", command=handle_query)
btn_query.pack(side="left", padx=5)

# Reset button
btn_reset = Button(query_reset_frame, text="Tạo lại", command=reset_app)
btn_reset.pack(side="right", padx=5)

# Upload
btn_bulk_upload = Button(query_reset_frame, text="Thêm văn hóa", command=open_file_dialog)
btn_bulk_upload.pack(pady=10)

# Upload message and error message
lbl_upload_message = Label(root, text="", fg="green")
lbl_upload_message.pack(pady=10)

lbl_error_message = Label(root, text="", fg="red")
lbl_error_message.pack(pady=10)

# Footer
footer = tk.Frame(root, bg="lightgrey")
footer.pack(side="bottom", fill="x")

footer_label = tk.Label(
    footer,
    text="Made by Tran Tien - B2017010",
    bg="lightgrey",
    fg="black",
    font=("Arial", 10, "italic")
)
footer_label.pack(pady=5)
main_frame.pack()

root.mainloop()