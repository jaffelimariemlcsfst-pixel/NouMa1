import os
import uuid
import streamlit as st
import pandas as pd
from fpdf import FPDF
import streamlit.components.v1 as components
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
URL = os.environ["QDRANT_URL"]
API_KEY = os.environ["QDRANT_API_KEY"]

# ============================================================================
# AUTHENTICATION & SAVED SEARCHES DATABASE LAYER
# ============================================================================
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

DB_PATH = "users.db"

def hash_password(password: str, salt: str = None) -> Tuple[str, str]:
    """Hash password using PBKDF2 with SHA256"""
    if salt is None:
        salt = secrets.token_hex(32)
    pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000).hex()
    return pwd_hash, salt

def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verify password against stored hash"""
    pwd_hash, _ = hash_password(password, salt)
    return pwd_hash == stored_hash

def init_db():
    """Initialize database tables"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        password_salt TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        session_token TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE)''')
    c.execute('''CREATE TABLE IF NOT EXISTS saved_searches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        filters JSON NOT NULL,
        keywords TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_used TIMESTAMP,
        is_favorite INTEGER DEFAULT 0,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
        UNIQUE(user_id, name))''')
    conn.commit()
    conn.close()

def register_user(email: str, password: str) -> Tuple[bool, str]:
    """Register new user"""
    if not email or not password or len(password) < 8 or '@' not in email:
        return False, "Invalid email or password (min 8 chars)"
    init_db()
    try:
        pwd_hash, salt = hash_password(password)
        conn = sqlite3.connect(DB_PATH)
        conn.execute('INSERT INTO users (email, password_hash, password_salt) VALUES (?, ?, ?)',
                    (email.lower(), pwd_hash, salt))
        conn.commit()
        conn.close()
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, "Email already registered"
    except Exception as e:
        return False, f"Error: {str(e)}"

def login_user(email: str, password: str) -> Tuple[bool, str, Optional[str]]:
    """Authenticate user and create session"""
    init_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT id, password_hash, password_salt FROM users WHERE email = ?', (email.lower(),))
        user = c.fetchone()
        if not user or not verify_password(password, user[1], user[2]):
            return False, "Invalid credentials", None
        user_id, _, _ = user
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(days=7)
        c.execute('INSERT INTO sessions (user_id, session_token, expires_at) VALUES (?, ?, ?)',
                 (user_id, session_token, expires_at))
        c.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?', (user_id,))
        conn.commit()
        conn.close()
        return True, "Login successful!", session_token
    except Exception as e:
        return False, f"Error: {str(e)}", None

def verify_session(session_token: str) -> Optional[Dict]:
    """Verify session token"""
    init_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT u.id, u.email, s.expires_at FROM sessions s JOIN users u ON s.user_id = u.id WHERE s.session_token = ?',
                 (session_token,))
        result = c.fetchone()
        conn.close()
        if not result or datetime.fromisoformat(result[2]) < datetime.now():
            return None
        return {'user_id': result[0], 'email': result[1]}
    except:
        return None

def logout_user(session_token: str) -> bool:
    """Logout user"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute('DELETE FROM sessions WHERE session_token = ?', (session_token,))
        conn.commit()
        conn.close()
        return True
    except:
        return False

def create_saved_search(user_id: int, name: str, filters: Dict, keywords: str = None) -> Tuple[bool, str, Optional[int]]:
    """Create saved search"""
    if not name or len(name) > 100:
        return False, "Invalid search name", None
    init_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('INSERT INTO saved_searches (user_id, name, filters, keywords) VALUES (?, ?, ?, ?)',
                 (user_id, name.strip(), json.dumps(filters), keywords))
        conn.commit()
        search_id = c.lastrowid
        conn.close()
        return True, f"Search '{name}' saved!", search_id
    except sqlite3.IntegrityError:
        return False, f"Search '{name}' already exists", None
    except Exception as e:
        return False, f"Error: {str(e)}", None

def get_saved_searches(user_id: int):
    """Get all saved searches for user"""
    init_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT id, name, filters, keywords, created_at, is_favorite FROM saved_searches WHERE user_id = ? ORDER BY is_favorite DESC, last_used DESC, created_at DESC',
                 (user_id,))
        results = c.fetchall()
        conn.close()
        return [{'id': r[0], 'name': r[1], 'filters': json.loads(r[2]), 'keywords': r[3], 'created_at': r[4], 'is_favorite': r[5]} for r in results]
    except:
        return []

def delete_saved_search(user_id: int, search_id: int) -> Tuple[bool, str]:
    """Delete saved search"""
    init_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('DELETE FROM saved_searches WHERE id = ? AND user_id = ?', (search_id, user_id))
        if c.rowcount == 0:
            conn.close()
            return False, "Search not found"
        conn.commit()
        conn.close()
        return True, "Search deleted!"
    except Exception as e:
        return False, f"Error: {str(e)}"

def toggle_favorite(user_id: int, search_id: int) -> bool:
    """Toggle favorite status"""
    init_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT is_favorite FROM saved_searches WHERE id = ? AND user_id = ?', (search_id, user_id))
        result = c.fetchone()
        if result:
            new_status = 1 - result[0]
            c.execute('UPDATE saved_searches SET is_favorite = ? WHERE id = ? AND user_id = ?', (new_status, search_id, user_id))
            conn.commit()
            conn.close()
            return True
        conn.close()
        return False
    except:
        return False





# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Tunisia Smartphone Search - Find Your Perfect Device",
    layout="wide",
    page_icon="📱",
    initial_sidebar_state="collapsed"
)
st.markdown("""
    <style>
    /* Add a professional card look to each product container */
    [data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
        background-color: #ffffff;
        border: 1px solid #f0f2f6;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    [data-testid="stVerticalBlock"] > div:has(div.stMarkdown):hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE AUTHENTICATION SESSION STATE
# ============================================================================
if 'auth_session_token' not in st.session_state:
    st.session_state['auth_session_token'] = None
if 'auth_user' not in st.session_state:
    st.session_state['auth_user'] = None
if 'show_saved_searches_page' not in st.session_state:
    st.session_state['show_saved_searches_page'] = False
if 'loaded_search' not in st.session_state:
    st.session_state['loaded_search'] = None

if 'compare_list' not in st.session_state:
    st.session_state.compare_list = {}
if 'page_offset' not in st.session_state:
    st.session_state.page_offset = 0
if 'total_searches' not in st.session_state:
    st.session_state.total_searches = 0

# --- 2. UTILITY FUNCTIONS ---
# ============================================================================
# AUTHENTICATION UI FUNCTIONS
# ============================================================================
def get_current_user():
    """Get authenticated user from session"""
    if st.session_state.get('auth_session_token'):
        user = verify_session(st.session_state.get('auth_session_token'))
        if user:
            st.session_state.auth_user = user
            return user
        else:
            st.session_state['auth_session_token'] = None
            st.session_state.auth_user = None
    return None

def render_auth_sidebar():
    """Render authentication UI in sidebar"""
    user = get_current_user()
    
    with st.sidebar:
        st.markdown("---")
        if user:
            st.success(f"✅ Logged in: {user['email']}")
            if st.button("🚪 Logout", use_container_width=True, key="logout_btn"):
                logout_user(st.session_state.get('auth_session_token'))
                st.session_state['auth_session_token'] = None
                st.session_state.auth_user = None
                st.session_state.show_saved_searches_page = False
                st.rerun()
            
            if st.button("📋 My Saved Searches", use_container_width=True, key="saved_searches_btn"):
                st.session_state.show_saved_searches_page = not st.session_state.show_saved_searches_page
                st.rerun()
        else:
            st.info("ℹ️ Not logged in")

def render_login_page():
    """Render login/register page"""
    st.markdown("## 🔐 Authentication")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Login")
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", key="login_btn", use_container_width=True):
            if login_email and login_password:
                success, message, session_token = login_user(login_email, login_password)
                if success:
                    st.session_state['auth_session_token'] = session_token
                    st.session_state['auth_user'] = verify_session(session_token)
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.warning("Enter email and password")
    
    with col2:
        st.markdown("### Register")
        reg_email = st.text_input("Email", key="reg_email")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
        if st.button("Register", key="register_btn", use_container_width=True):
            if not reg_email or not reg_password:
                st.warning("Fill all fields")
            elif reg_password != reg_confirm:
                st.error("Passwords don't match")
            else:
                success, message = register_user(reg_email, reg_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)

def render_saved_searches_page():
    """Render saved searches management page"""
    user = get_current_user()
    if not user:
        st.warning("Please log in")
        return
    
    st.markdown("## 📋 My Saved Searches")
    saved_searches = get_saved_searches(user['user_id'])
    
    if not saved_searches:
        st.info("No saved searches yet!")
        return
    
    for search in saved_searches:
        with st.expander(f"{'⭐' if search['is_favorite'] else '🔍'} {search['name']}", expanded=False):
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.markdown(f"**Created:** {search['created_at']}")
                if search['keywords']:
                    st.markdown(f"**Keywords:** {search['keywords']}")
                st.json(search['filters'])
            
            with col2:
                if st.button("🔄 Use", key=f"use_{search['id']}", use_container_width=True):
                    st.session_state.loaded_search = search
                    st.session_state.show_saved_searches_page = False
                    st.rerun()
            
            with col3:
                if st.button("⭐ Fav" if not search['is_favorite'] else "✨ Fav", key=f"fav_{search['id']}", use_container_width=True):
                    toggle_favorite(user['user_id'], search['id'])
                    st.rerun()
            
            with col4:
                if st.button("🗑️ Del", key=f"del_{search['id']}", use_container_width=True):
                    success, msg = delete_saved_search(user['user_id'], search['id'])
                    if success:
                        st.success("Deleted!")
                        st.rerun()

def render_save_search_widget(filters: dict, keywords: str = None):
    """Render save search widget"""
    user = get_current_user()
    if not user:
        return
    
    with st.expander("💾 Save This Search"):
        search_name = st.text_input("Search name", placeholder="e.g., Budget Android Phones", key=f"save_search_name_{id(filters)}")
        if st.button("Save", key=f"save_btn_{id(filters)}", use_container_width=True):
            if search_name:
                success, message, _ = create_saved_search(user['user_id'], search_name, filters, keywords)
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.warning("Enter a name")

# --- 2. UTILITY FUNCTIONS ---
def scroll_to_top():
    """Scroll to top smoothly using JavaScript"""
    components.html(
        "<script>window.parent.document.querySelector('.main').scrollTo({top: 0, behavior: 'smooth'});</script>",
        height=0
    )

def get_stars(rating):
    """Convert numeric rating to stars"""
    try:
        r = int(float(rating))
        return "⭐" * min(max(r, 1), 5)
    except:
        return "N/A"

def format_price_display(price, currency="DT"):
    """Format price with proper styling"""
    try:
        return f"{float(price):,.0f} {currency}"
    except:
        return f"{price} {currency}"

def generate_comparison_pdf(data_dict):
    """Generate a professional landscape PDF with comparison table"""
    from fpdf import FPDF
    import pandas as pd
    
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.add_page()
    
    # Color scheme
    PINK = (233, 30, 99)
    TURQUOISE = (64, 224, 208)
    NAVY = (10, 22, 40)
    LILAC = (200, 162, 200)

    # Title
    pdf.set_font("Arial", "B", 24)
    pdf.set_text_color(*PINK)
    pdf.cell(0, 15, "Tunisia Smartphone Search", ln=True, align="C")
    
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(*TURQUOISE)
    pdf.cell(0, 10, "Product Comparison Report", ln=True, align="C")
    pdf.ln(5)

    # Create DataFrame
    df = pd.DataFrame(data_dict).T
    
    # Define required columns
    required_cols = ['name', 'display_price', 'brand', 'color', 'availability']
    for col in required_cols:
        if col not in df.columns:
            df[col] = "N/A"
    
    features = [
        ('name', 'Product Name'), 
        ('brand', 'Brand'),
        ('display_price', 'Price'), 
        ('color', 'Color'),
        ('availability', 'Availability')
    ]

    # Header row
    pdf.set_font("Arial", "B", 10)
    pdf.set_fill_color(*LILAC)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(45, 12, "Feature", border=1, fill=True, align='C')
    
    for name in df['name']:
        pdf.cell(48, 12, str(name)[:20], border=1, fill=True, align='C')
    pdf.ln()

    # Data rows
    pdf.set_text_color(*NAVY)
    for key, label in features:
        pdf.set_font("Arial", "B", 9)
        pdf.set_fill_color(245, 247, 250)
        pdf.cell(45, 10, label, border=1, fill=True)
        
        pdf.set_font("Arial", "", 9)
        for val in df[key]:
            clean_val = str(val) if val is not None else "N/A"
            pdf.cell(48, 10, clean_val[:22], border=1)
        pdf.ln()

    # Footer
    pdf.ln(10)
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, "Generated by Tunisia Smartphone Search | Your Smart Shopping Assistant", align='C')

    # FIXED: Handle different FPDF versions
    output = pdf.output()
    if isinstance(output, bytes):
        return output
    else:
        return output.encode('latin-1')

# --- 3. LOAD CSS ---
if os.path.exists("style.css"):
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- 4. HEADER IMAGE & BRANDING ---
image_path = r"c:\Users\user\Pictures\Screenshots\Capture d'écran 2026-01-23 142433.png"

if os.path.exists(image_path):
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(image_path, use_container_width=True)
else:
    # Fallback header if image not found
    st.markdown("""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #E91E63 0%, #40E0D0 100%); border-radius: 20px; margin-bottom: 2rem;'>
            <h1 style='color: white; font-size: 3rem; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>📱 Tunisia Smartphone Search</h1>
            <p style='color: white; font-size: 1.2rem; margin-top: 0.5rem;'>Find Your Perfect Device at the Best Price</p>
        </div>
    """, unsafe_allow_html=True)

# --- 5. LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    client = QdrantClient(url=URL, api_key=API_KEY, timeout=60)
    model = SentenceTransformer('clip-ViT-B-32')
    return client, model
# --- 4.5 AUTHENTICATION ---
render_auth_sidebar()

# Handle page navigation
if st.session_state.show_saved_searches_page:
    render_saved_searches_page()
    st.stop()

# Check if user is authenticated
if not get_current_user():
    render_login_page()
    st.stop()

# --- 5. LOAD RESOURCES ---
client, model = load_resources()
collection_name = "products2"

# --- 6. HERO SECTION ---
st.markdown("""
    <div style='text-align: center; margin: 2rem 0;'>
        <h1 style='font-size: 3.5rem; font-weight: 800; background: linear-gradient(135deg, #E91E63 0%, #40E0D0 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;'>
            WELCOME TO TUNISIA SMARTPHONE SEARCH 📱
        </h1>
        <p class='subtitle' style='font-size: 1.3rem; color: #718096; font-weight: 400;'>
            🔍 Smart Search · 💰 Best Prices · ⚡ Lightning Fast · 🎯 AI-Powered
        </p>
    </div>
""", unsafe_allow_html=True)

# --- 7. SEARCH UI ---
st.markdown("<h2 style='color: #E91E63; margin-top: 2rem;'>🔎 Find Your Perfect Product</h2>", unsafe_allow_html=True)

# First row: Search and Budget
col1, col2, col3 = st.columns([3, 2, 2])
with col1:
    query = st.text_input(
        "Search by name, brand, or model...", 
        placeholder="Example: iPhone 15, Samsung Galaxy, Xiaomi Redmi...",
        help="Enter any keyword to search across all products"
    )
with col2:
    budget = st.number_input(
        "Maximum Budget (DT)", 
        min_value=0, 
        max_value=20000, 
        value=5000, 
        step=100,
        help="Set your budget limit"
    )
with col3:
    category_filter = st.selectbox(
        "Category", 
        ["Tous", "Smartphone", "Ordinateur", "Accessoires","Réfrigérateur", "Casque & Écouteurs","Électroménager"],
        help="Filter by product category"
    )

# Second row: Color and Image upload
col4, col5 = st.columns([1, 1])
with col4:
    color_filter = st.selectbox(
        "Preferred Color 🎨", 
        ["Toutes", "Noir", "Bleu", "Vert", "Rouge", "Blanc", "Gold", "Silver", "Violet"],
        help="Filter by color preference"
    )
with col5:
    uploaded_file = st.file_uploader(
        "Or Search by Image 📸", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a phone image to find similar models"
    )

# --- 8. STATS DASHBOARD ---
if st.session_state.total_searches > 0:
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #E91E63 0%, #FF6B9D 100%); padding: 1.5rem; border-radius: 16px; text-align: center; box-shadow: 0 4px 15px rgba(233, 30, 99, 0.3);'>
                <h3 style='color: white; margin: 0; font-size: 2rem;'>{st.session_state.total_searches}</h3>
                <p style='color: white; margin: 0; opacity: 0.9;'>Total Searches</p>
            </div>
        """, unsafe_allow_html=True)
    with stat_col2:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #40E0D0 0%, #7FEFDC 100%); padding: 1.5rem; border-radius: 16px; text-align: center; box-shadow: 0 4px 15px rgba(64, 224, 208, 0.3);'>
                <h3 style='color: white; margin: 0; font-size: 2rem;'>{len(st.session_state.compare_list)}</h3>
                <p style='color: white; margin: 0; opacity: 0.9;'>Products Comparing</p>
            </div>
        """, unsafe_allow_html=True)
    with stat_col3:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); padding: 1.5rem; border-radius: 16px; text-align: center; box-shadow: 0 4px 15px rgba(255, 215, 0, 0.3);'>
                <h3 style='color: white; margin: 0; font-size: 2rem;'>{budget:,.0f} DT</h3>
                <p style='color: white; margin: 0; opacity: 0.9;'>Your Budget</p>
            </div>
        """, unsafe_allow_html=True)

# --- 9. COMPARISON SECTION ---
if st.session_state.compare_list:
    st.markdown("---")
    st.markdown("<h2 style='color: #E91E63;'>⚖️ Product Comparison</h2>", unsafe_allow_html=True)
    
    with st.expander("📊 View Detailed Comparison Table", expanded=True):
        comp_df = pd.DataFrame([
            {
                "🏷️ Product": p.get('name', 'N/A')[:45] + "...",
                "🏢 Brand": p.get('brand', 'N/A'),
                "💰 Price": p.get('display_price', 'N/A'),
                "🎨 Color": p.get('color', 'N/A'),
                "📦 Stock": p.get('availability', 'N/A')
            } for p in st.session_state.compare_list.values()
        ])
        
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
        
        with btn_col1:
            pdf_data = generate_comparison_pdf(st.session_state.compare_list)
            st.download_button(
                "📥 Download PDF Report", 
                data=pdf_data, 
                file_name=f"smartphone_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf", 
                mime="application/pdf",
                use_container_width=True
            )
        
        with btn_col2:
            # Export as CSV
            csv = comp_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📊 Export as CSV",
                data=csv,
                file_name=f"comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with btn_col3:
            if st.button("🗑️ Clear Comparison List", use_container_width=True, type="secondary"):
                st.session_state.compare_list = {}
                st.rerun()

st.markdown("---")

# --- 10. SEARCH EXECUTION ---
search_vector = None
search_performed = False

if uploaded_file is not None:
    from PIL import Image
    img = Image.open(uploaded_file)
    # Load saved search if user clicked "Use" on a saved search
    if st.session_state.loaded_search:
        loaded = st.session_state.loaded_search
        query = loaded.get('keywords', '')
        budget = loaded['filters'].get('budget', budget)
        category_filter = loaded['filters'].get('category', category_filter)
        color_filter = loaded['filters'].get('color', color_filter)
        st.info(f"✓ Using saved search: **{loaded['name']}**")
        st.session_state.loaded_search = None
    
    img_col1, img_col2, img_col3 = st.columns([1, 2, 1])
    with img_col2:
        st.image(img, caption="🔍 Searching for similar phones...", use_container_width=True)
    
    search_vector = model.encode(img).tolist()
    search_performed = True
    st.session_state.total_searches += 1
    
elif query:
    search_vector = model.encode(query).tolist()
    search_performed = True
    st.session_state.total_searches += 1

if search_vector:
    try:
        # Build filter conditions
        filter_conditions = [
            models.FieldCondition(
                key="price", 
                range=models.Range(lte=float(budget))
            )
        ]
        
        if category_filter != "Tous":
            filter_conditions.append(
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value=category_filter)
                )
            )
        
        if color_filter != "Toutes":
            filter_conditions.append(
                models.FieldCondition(
                    key="color",
                    match=models.MatchValue(value=color_filter)
                )
            )
        
        # Execute search - Using scroll with filter and manual similarity scoring
        from qdrant_client.models import ScrollRequest
        
        # First, get all points matching the filter (up to limit)
        scroll_result = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(must=filter_conditions),
            limit=500,  # Get more to ensure we have enough after filtering
            with_payload=True,
            with_vectors=True
        )
        
        # Calculate similarity scores manually
        import numpy as np
        
        results = []
        search_vec = np.array(search_vector)
        
        # Check if this is a text query (not image search)
        query_text = query.lower() if query else ""
        
        for point in scroll_result[0]:  # scroll returns (points, next_offset)
            # Calculate cosine similarity
            point_vec = np.array(point.vector)
            similarity = np.dot(search_vec, point_vec) / (np.linalg.norm(search_vec) * np.linalg.norm(point_vec))
            
            # Boost score for exact/partial text matches
            boost = 0
            if query_text:
                product_name = point.payload.get('name', '').lower()
                product_brand = point.payload.get('brand', '').lower()
                
                # Exact name match - huge boost
                if query_text in product_name:
                    boost += 0.3
                
                # Brand match - medium boost
                if query_text in product_brand:
                    boost += 0.2
                
                # Word-by-word match
                query_words = query_text.split()
                name_words = product_name.split()
                matching_words = sum(1 for word in query_words if word in name_words)
                boost += (matching_words / len(query_words)) * 0.15 if query_words else 0
            
            final_score = similarity + boost
            
            # Create a result object similar to search results
            class ScoredPoint:
                def __init__(self, id, score, payload):
                    self.id = id
                    self.score = final_score
                    self.payload = payload
            
            results.append(ScoredPoint(point.id, final_score, point.payload))
        
        # Sort by combined score (highest first)
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:100]  # Keep top 100
        TOP_K = 7  # or 5

        def get_price(p):
            try:
                return float(p.payload.get("price", budget))
            except:
                return budget
        # --- Step 1: keep similarity order, but adjust first TOP_K by price closeness
        top_similar = results[:TOP_K]

        top_similar.sort(
        key=lambda x: abs(get_price(x) - budget)
        )

        # --- Step 2: remaining products (still under budget)
        remaining = results[TOP_K:]

        remaining.sort(
            key=lambda x: get_price(x)
        )

        # --- Final ordered list
        results = top_similar + remaining

        
        # --- Display Results ---
        if len(results) == 0:
            st.info("🔍 No products found matching your criteria. Try adjusting your filters!")
        else:
            # Results header
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, rgba(233, 30, 99, 0.1) 0%, rgba(64, 224, 208, 0.1) 100%); 
                            padding: 1.5rem; border-radius: 16px; margin: 2rem 0; text-align: center; border: 2px solid #E91E63;'>
                    <h2 style='color: #E91E63; margin: 0;'>🎯 Found {len(results)} Products Matching Your Search!</h2>
                </div>
            """, unsafe_allow_html=True)

            # Save search widget
            current_filters = {
                'query': query,
                'budget': budget,
                'category': category_filter,
                'color': color_filter
            }
            render_save_search_widget(current_filters, query)

            # Setup Pagination
            items_per_page = 9
            total_pages = (len(results) // items_per_page) + (1 if len(results) % items_per_page > 0 else 0)
            current_idx = st.session_state.get('page_offset', 0)
            page_results = results[current_idx : current_idx + items_per_page]

            # Product Grid
            for i in range(0, len(page_results), 3):
                cols = st.columns(3)
                for j, hit in enumerate(page_results[i:i+3]):
                    with cols[j]:
                        with st.container():
                            p = hit.payload
                            p_id = hit.id
                            
                            # 1. Product Image
                            img_url = str(p.get('image', ''))
                            if img_url and img_url.lower() != 'nan' and img_url.startswith('http'):
                                st.markdown(f"""
                                    <div style="display: flex; justify-content: center; align-items: center; height: 200px; margin-bottom: 10px;">
                                        <img src="{img_url}" style="max-height: 100%; max-width: 100%; object-fit: contain;">
                                     </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                    <div style='background: linear-gradient(135deg, #E91E63 0%, #40E0D0 100%); 
                                                padding: 4rem; text-align: center; border-radius: 16px;'>
                                        <p style='color: white; font-size: 3rem; margin: 0;'>📱</p>
                                    </div>
                                """, unsafe_allow_html=True)

                            # 2. Discount Badge
                            if p.get('has_discount'):
                                discount_pct = p.get('discount_percentage', 0)
                                st.markdown(f"""
                                    <div style="background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%); 
                                                color: white; padding: 8px; border-radius: 12px; text-align: center; 
                                                margin: 10px 0; font-weight: bold; box-shadow: 0 4px 10px rgba(255, 68, 68, 0.3);
                                                animation: pulse 2s infinite;">
                                        🔥 SPECIAL OFFER -{discount_pct}%
                                    </div>
                                """, unsafe_allow_html=True)

                            # 3. Availability Status
                            status = str(p.get('availability', 'En Stock'))
                            if status == "En Stock":
                                st.success(f"✅ {status}")
                            elif "Rupture" in status or "Out of Stock" in status:
                                st.error(f"❌ {status}")
                            else:
                                st.warning(f"⏳ {status}")
                            
                            # 4. Product Name
                            product_name = p.get('name', 'Product Name')[:50]
                            st.markdown(f"<h3 style='color: #1A202C; font-size: 1.1rem; font-weight: 700; margin: 0.5rem 0;'>{product_name}...</h3>", unsafe_allow_html=True)
                            
                            # 5. Price Display
                            if p.get('original_price') and p.get('has_discount'):
                                price_html = f"""
                                    <div style='margin: 1rem 0;'>
                                        <span style="text-decoration: line-through; color: #999; font-size: 0.95rem; display: block;">
                                            {p.get('display_original_price', '')}
                                        </span>
                                        <span style="color: #E91E63; font-size: 1.8rem; font-weight: 800; font-family: 'Space Mono', monospace;">
                                            {p.get('display_price', 'N/A')}
                                        </span>
                                    </div>
                                """
                            else:
                                price_html = f"""
                                    <div style="color: #E91E63; font-size: 1.6rem; font-weight: 800; margin: 1rem 0; font-family: 'Space Mono', monospace;">
                                        {p.get('display_price', 'N/A')}
                                    </div>
                                """
                            
                            st.markdown(price_html, unsafe_allow_html=True)
                            
                            # 6. Product Details
                            detail_col1, detail_col2 = st.columns(2)
                            with detail_col1:
                                st.markdown(f"""
                                    <div style='background: #1A202C; color: white; padding: 0.4rem; border-radius: 6px; 
                                    font-size: 0.8rem; text-align: center; font-weight: 600; margin-bottom: 5px;'>
                                        🏢 {p.get('brand', 'Brand')}
                                     </div>""", unsafe_allow_html=True)
                            with detail_col2:
                                st.markdown(f"""
                                    <div style='background: #E91E63; color: white; padding: 0.4rem; border-radius: 6px; 
                                    font-size: 0.8rem; text-align: center; font-weight: 600; margin-bottom: 5px;'>
                                        🎨 {p.get('color', 'Color')}
                                    </div>""", unsafe_allow_html=True)
                            
                            # 7. Action Buttons
                            compare_label = "✅ Selected for Comparison" if p_id in st.session_state.compare_list else "⚖️ Add to Compare"
                            button_type = "secondary" if p_id in st.session_state.compare_list else "primary"
                            
                            if st.button(compare_label, key=f"compare_{p_id}", use_container_width=True, type=button_type):
                                if p_id in st.session_state.compare_list:
                                    del st.session_state.compare_list[p_id]
                                    st.toast("❌ Removed from comparison", icon="🗑️")
                                else:
                                    if len(st.session_state.compare_list) < 5:
                                        st.session_state.compare_list[p_id] = p
                                        st.toast("✅ Added to comparison!", icon="⚖️")
                                    else:
                                        st.warning("⚠️ Maximum 5 products for comparison")
                                st.rerun()
                            
                            # 8. View Offer Button
                            offer_url = p.get('url', '#')
                            if offer_url and offer_url != '#':
                                st.link_button("🛒 View Full Details & Buy", offer_url, use_container_width=True)
                            else:
                                st.button("🛒 View Details (Coming Soon)", use_container_width=True, disabled=True)

            # Pagination Navigation
            st.markdown("---")
            st.markdown("<div style='margin: 3rem 0;'>", unsafe_allow_html=True)
            
            nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
            
            with nav_col1:
                if st.button("⬅️ Previous Page", use_container_width=True, disabled=(st.session_state.page_offset < items_per_page)):
                    st.session_state.page_offset -= items_per_page
                    scroll_to_top()
                    st.rerun()
            
            with nav_col2:
                current_page = (st.session_state.page_offset // items_per_page) + 1
                st.markdown(
                    f"""<div style='text-align: center; padding: 1rem; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                        <p style='margin: 0; font-size: 1.3rem; font-weight: 700; font-family: "Space Mono", monospace; color: #E91E63;'>
                            Page <span style='color: #40E0D0;'>{current_page}</span> of <span style='color: #40E0D0;'>{total_pages}</span>
                        </p>
                    </div>""", 
                    unsafe_allow_html=True
                )
            
            with nav_col3:
                if st.button("Next Page ➡️", use_container_width=True, disabled=((st.session_state.page_offset + items_per_page) >= len(results))):
                    st.session_state.page_offset += items_per_page
                    scroll_to_top()
                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
                        
    except Exception as e:
        st.error(f"❌ Search Error: {e}")
        st.info("💡 Try adjusting your search criteria or contact support if the problem persists.")

elif not search_performed:
    # Welcome message when no search is performed
    st.markdown("""
        <div style='text-align: center; padding: 4rem 2rem; background: white; border-radius: 20px; margin: 3rem 0; box-shadow: 0 10px 30px rgba(0,0,0,0.1);'>
            <div style='font-size: 5rem; margin-bottom: 1rem;'>🔍</div>
            <h2 style='color: #E91E63; margin-bottom: 1rem;'>Ready to Find Your Perfect Smartphone?</h2>
            <p style='font-size: 1.2rem; color: #718096; max-width: 600px; margin: 0 auto;'>
                Enter a search term above or upload an image to discover the best smartphone deals in Tunisia!
            </p>
            <div style='margin-top: 2rem; padding: 1.5rem; background: linear-gradient(135deg, rgba(233, 30, 99, 0.1) 0%, rgba(64, 224, 208, 0.1) 100%); border-radius: 12px;'>
                <p style='margin: 0; font-size: 1rem; color: #1A202C;'>
                    <strong>💡 Pro Tips:</strong> Search by brand (Samsung, iPhone), model (Galaxy S24), or even upload a photo of the phone you like!
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- 11. FOOTER ---
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #0A1628 0%, #1A2B4A 100%); border-radius: 20px; margin-top: 3rem;'>
        <h3 style='color: white; margin-bottom: 1rem;'>Tunisia Smartphone Search</h3>
        <p style='color: rgba(255,255,255,0.7); margin-bottom: 1rem;'>Your AI-Powered Smart Shopping Assistant</p>
        <p style='color: rgba(255,255,255,0.5); font-size: 0.9rem;'>
            🚀 Powered by Advanced AI · 🇹🇳 Made for Tunisia · 💰 Saving You Money
        </p>
    </div>
""", unsafe_allow_html=True)