import os
import uuid
import streamlit as st
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import json
from config import URL, API_KEY


# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Tunisia Smartphone Search", layout="wide",page_icon="🤖")

# PASTE YOUR DETAILS HERE FROM THE QDRANT DASHBOARD
image_path = r"c:\Users\user\Pictures\Screenshots\Capture d'écran 2026-01-23 142433.png"

if os.path.exists(image_path):
    # Create 3 columns: Left (1 part), Middle (2 parts), Right (1 part)
    left_co, cent_co, last_co = st.columns([1, 2, 1])
    
    with cent_co:
        st.image(image_path)
else:
    st.error(f"L'image '{image_path}' est introuvable !")




@st.cache_resource
def load_resources():
    # Connect to Cloud
    client = QdrantClient(url=URL, api_key=API_KEY, timeout=60)
    # Load AI Model
    model = SentenceTransformer('clip-ViT-B-32')
    return client, model

client, model = load_resources()
collection_name = "my_products"

# --- 2. UPLOAD LOGIC (Sends data to Cloud) ---
def upload_to_cloud():
    # 1. Create Collection
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=512,distance=models.Distance.COSINE)
    )

    # Create indexes
    client.create_payload_index(
        collection_name=collection_name,
        field_name="price",
        field_schema=models.PayloadSchemaType.FLOAT,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="category",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="color",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    
    # 2. Load your local json
    try:
        
        with open('phone_list.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        st.write(f"🔍 Type of data: {type(data)}")
        st.write(f"📊 Total items loaded: {len(data)}")
        
        # Check if it's a list of lists or nested structure
        if isinstance(data, list) and len(data) > 0:
            st.write(f"✅ First item: {data[0].get('Product Name', 'N/A')[:30]}")
            st.write(f"✅ Last item: {data[-1].get('Product Name', 'N/A')[:30]}")
        
        df = pd.DataFrame(data)
        
    except Exception as e:
        st.error(f"Error loading JSON: {e}")
        return 0
    df.columns = df.columns.str.strip()
    
    st.write(f"📊 Total rows in JSON file: {len(df)}")
    # PRE-ENCODE ALL VECTORS AT ONCE (faster!)
    st.write("🔄 Encoding product names...")
    all_names = [str(row['Product Name']) for _, row in df.iterrows()]
    all_vectors = model.encode(all_names, show_progress_bar=False).tolist()
    
    points = []
    total_uploaded = 0  # ← MAKE SURE THIS IS HERE
    progress_bar = st.progress(0)
    
    for idx, row in df.iterrows():
        # Clean price
        p_raw = str(row.get('Price') or '0')
        p_clean = "".join(filter(lambda x: x.isdigit() or x == '.', p_raw.replace(',', '.')))
        
        # Extract category
        product_name = str(row['Product Name']).lower()
        category = "Smartphone"
        if any(word in product_name for word in ['pc', 'laptop', 'ordinateur', 'macbook']):
            category = "Ordinateur"
        elif any(word in product_name for word in ['case', 'charger', 'cable', 'écouteur', 'casque']):
            category = "Accessoires"
        elif any(word in product_name for word in ['lave', 'vaisselle', 'dishwasher']):
            category = "Électroménager"
        
        # Extract color
        color = "Non spécifié"
        color_keywords = {
            'Noir': ['black', 'noir'],
            'Bleu': ['blue', 'bleu'],
            'Vert': ['green', 'vert'],
            'Rouge': ['red', 'rouge'],
            'Blanc': ['white', 'blanc'],
            'Gold': ['gold', 'doré'],
            'Silver': ['silver', 'argent', 'gris'],
            'Violet': ['violet', 'purple', 'mauve']
        }
        for color_name, keywords in color_keywords.items():
            if any(keyword in product_name for keyword in keywords):
                color = color_name
                break
        
        points.append(models.PointStruct(
            id=str(uuid.uuid4()),
            vector=all_vectors[idx],  # Use pre-encoded vector
            payload={
                "name": str(row['Product Name']),
                "price": float(p_clean) if p_clean else 0.0,
                "url": str(row.get('Product URL', '')),
                "image": str(row.get('Main Image', '')),
                "display_price": p_raw,
                "category": category,
                "availability": str(row.get('Availability', 'En Stock')),
                "brand": str(row.get('Brand Name', 'Divers')),
                "color": color
            }
        ))
        
        # Upload in batches of 20
        if len(points) >= 20:
            try:
                client.upsert(collection_name, points, wait=True)
                total_uploaded += len(points)
                points = []
            except Exception as e:
                st.error(f"Upload error: {e}")
                break
            
        progress_bar.progress((idx + 1) / len(df))
    
    # Upload remaining points
    if points:
        client.upsert(collection_name, points)
        total_uploaded += len(points)
    
    st.write(f"📦 Total products uploaded: {total_uploaded}")
    return total_uploaded  # ← RETURN THIS, NOT len(df)

# --- 3. UI ---
st.title("WELCOME TO TUNISIA SMARTPHONE SEARCH 📱")

with st.sidebar:
    if st.button("🚀 PUSH DATA TO CLOUD"):
        with st.spinner("Uploading items to Qdrant Cloud..."):
            count = upload_to_cloud()
            st.success(f"Successfully uploaded {count} products!")

    st.markdown("---")
    

# --- 4. SEARCH UI & EXECUTION ---
st.subheader("Find your phone by Name or Photo")
query = st.text_input("Rechercher...", placeholder="ex: Xiaomi")
budget = st.number_input("Budget maximum (DT)", 0, 5000, 5000)
category_filter = st.selectbox("Catégorie", ["Tous", "Smartphone", "Ordinateur", "Accessoires"])
color_filter = st.selectbox("Couleur 🎨", ["Toutes", "Noir", "Bleu", "Vert", "Rouge", "Blanc", "Gold", "Silver", "Violet"])
uploaded_file = st.file_uploader("Upload a phone image to search", type=['jpg', 'jpeg', 'png'])

st.markdown("<h2 style='color: #E91E63;'> </h2>", unsafe_allow_html=True)
search_vector = None

if uploaded_file is not None:
    from PIL import Image
    img = Image.open(uploaded_file)
    st.image(img, caption="Searching for similar phones to this...", width=150)
    search_vector = model.encode(img).tolist()
elif query:
        search_vector = model.encode(query).tolist()
        # This is the most basic, original way to search in Qdrant
        # It works even if your library is very old.
if search_vector:
    try:
        # Build filter conditions
        filter_conditions = [
            models.FieldCondition(
                key="price", 
                range=models.Range(lte=float(budget))
            )
        ]
        # Add category filter if not "Tous"
        if category_filter != "Tous":
            filter_conditions.append(
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value=category_filter)
                )
            )
        
        # Add color filter if not "Toutes"
        if color_filter != "Toutes":
            filter_conditions.append(
                models.FieldCondition(
                    key="color",
                    match=models.MatchValue(value=color_filter)
                )
            )
        
        results = client.search(
            collection_name=collection_name,
            query_vector=search_vector, 
            limit=100,
            query_filter=models.Filter(must=filter_conditions)
        )
        
        # --- Display Results ---
        
        
        if len(results) == 0:
            st.info("Aucun produit trouvé avec ces filtres.")
        else:
            st.write(f"### Results:")
            # 1. Setup Pagination State
            if 'page_offset' not in st.session_state:
                st.session_state.page_offset = 0
            if 'page_history' not in st.session_state:
                st.session_state.page_history = [0]

            # 2. Slice results for the current page (e.g., 9 items at a time)
            items_per_page = 9
            current_idx = st.session_state.get('page_offset', 0)
            # Since you fetched 100 results, we slice them here locally
            page_results = results[current_idx : current_idx + items_per_page]

            # 3. Create the Grid
            for i in range(0, len(page_results), 3):
                cols = st.columns(3)
                for j, hit in enumerate(page_results[i:i+3]):
                    with cols[j]:
                        with st.container(border=True):
                            p = hit.payload
                            
                            # Image Logic
                            img_url = str(p.get('image', ''))
                            if img_url and img_url.lower() != 'nan' and img_url.startswith('http'):
                                st.image(img_url, use_container_width=True)
                            else:
                                st.warning("📸 No Image")

                            # Availability Badge
                            status = str(p.get('availability', 'En Stock'))
                            stock_label = "✅ En Stock" if "Stock" in status else f"⏳ {status}"
                            
                            # Product Info
                            st.markdown(f"""
                                 <div style="min-height: 120px;">
                                    <div style="font-size: 0.7rem; color: #40E0D0; font-weight: bold;">{stock_label}</div>
                                    <div style="font-weight: bold; font-size: 0.9rem;">{p.get('name', 'N/A')[:45]}...</div>
                                    <div style="color: #E91E63; font-size: 1.2rem; font-weight: bold;">{p.get('display_price', 'N/A')}</div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Brand & Color Small Badges
                            c1, c2 = st.columns(2)
                            c1.caption(f"📦 {p.get('brand', 'Divers')}")
                            c2.caption(f"🎨 {p.get('color', 'N/A')}")
                            
                            st.link_button("View Offer", p.get('url', '#'), use_container_width=True)

            # --- 6. PAGINATION BUTTONS ---
            st.markdown("---")
            foot1, foot2, foot3 = st.columns([1, 2, 1])
            
            with foot1:
                if st.session_state.page_offset > 0:
                    if st.button("⬅️ Précédent"):
                        st.session_state.page_offset -= items_per_page
                        st.rerun()
            
            with foot3:
                if current_idx + items_per_page < len(results):
                    if st.button("Suivant ➡️"):
                        st.session_state.page_offset += items_per_page
                        st.rerun()
                        

    except Exception as e:
        st.error(f"Search error: {e}")