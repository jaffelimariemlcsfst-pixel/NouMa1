import requests
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from sentence_transformers import SentenceTransformer
import time
import logging
import uuid
import json  # <--- Added for JSON support
from config import URL, API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TunisianetScraper:
    def __init__(self):
        self.headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3',
        }
        self.session = requests.Session()
        self.client = QdrantClient(url=URL, api_key=API_KEY, timeout=60)
        self.collection_name = "products2"
        self.json_file = "products_backup.json" # <--- Our local landing zone

        # Ensure Collection & Indexes
        if not self.client.collection_exists(self.collection_name):
            logger.info(f"📦 Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )
        
        # Verify Indexes (Fixes the 400 error)
        try:
            self.client.create_payload_index(self.collection_name, "price", PayloadSchemaType.FLOAT)
            self.client.create_payload_index(self.collection_name, "category", PayloadSchemaType.KEYWORD)
            self.client.create_payload_index(self.collection_name, "color", PayloadSchemaType.KEYWORD)
        except Exception:
            pass # Indexes already exist
            
        self.model = SentenceTransformer('clip-ViT-B-32')
        logger.info("✅ Scraper and Qdrant client initialized")

    def scrape_products(self, url, brand):
        try:
            # We add a slight delay to look more 'human'
            time.sleep(2) 
            response = self.session.get(url, headers=self.headers, timeout=15)
            # Add this inside scrape_products right after response = self.session.get(...)
            if brand == "MyTek" and not soup.select(container):
                with open("debug_mytek.html", "w", encoding='utf-8') as f:
                     f.write(response.text)
                logger.warning("🔍 Debug: MyTek HTML saved to 'debug_mytek.html'. Open this file in your browser to see what MyTek is actually sending!")
            if response.status_code != 200:
                logger.error(f"❌ {brand} blocked us! Status: {response.status_code}")
                return []

            soup = BeautifulSoup(response.content, 'html.parser')
            products = []

            # --- SELECTORS ---
            if brand == "Tunisianet":
                container, name_s, price_s, img_s = '.item-product', 'h2.product-title a', 'span[itemprop="price"]', '.product-thumbnail img'
            elif brand == "MyTek":
                # Broadened selectors to be more flexible
                container = '.product-item' # Removed the 'li' to be less restrictive
                name_s = 'a.product-item-link'
                price_s = '[data-price-type="finalPrice"] .price, .price' # Tries two ways to find price
                img_s = 'img.product-image-photo'
            else:
                return []

            items = soup.select(container)
            if not items:
                logger.warning(f"⚠️ {brand}: No items found with selector '{container}'")
                return []

            for item in items:
                try:
                    name_elem = item.select_one(name_s)
                    if not name_elem: continue
                    
                    name = name_elem.text.strip()
                    link = name_elem['href']
                    
                    price_elem = item.select_one(price_s)
                    price = price_elem.text.strip() if price_elem else "0"
                    
                    img_elem = item.select_one(img_s)
                    # MyTek often uses 'src', but check 'data-src' just in case of lazy loading
                    image = ""
                    if img_elem:
                        image = img_elem.get('src') or img_elem.get('data-src') or ""

                    products.append({'name': name, 'price': price, 'url': link, 'image': image})
                except Exception:
                    continue

            logger.info(f"✅ {brand}: Found {len(products)} products.")
            return products

        except Exception as e:
            logger.error(f"❌ Scrape Error: {e}")
            return []

    def save_to_json(self, products):
        """Saves the scraped list to a local file"""
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(products, f, ensure_ascii=False, indent=4)
        logger.info(f"💾 Data saved locally to {self.json_file}")

    # We add 'brand_name' here so it has a "slot" to catch the brand you are sending!
    def upload_to_qdrant(self, products, brand_name="Tunisianet"):
        """Uploads from the provided list, or from the JSON file if list is empty"""
        if not products:
            try:
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    products = json.load(f)
                logger.info(f"📂 Loaded {len(products)} items from JSON for upload")
            except FileNotFoundError:
                logger.error("❌ No data found to upload!")
                return 0

        product_names = [p['name'] for p in products]
        vectors = self.model.encode(product_names, show_progress_bar=True).tolist()
        
        points = []
        for product, vector in zip(products, vectors):
            p_name = product['name'].lower()
            category = "Smartphone"
            if any(x in p_name for x in ['pc', 'laptop', 'ordinateur', 'macbook']): category = "Ordinateur"
            elif any(x in p_name for x in ['case', 'charger', 'cable', 'écouteur']): category = "Accessoires"
            
            color = "Non spécifié"
            colors = {'Noir':['noir','black'], 'Bleu':['blue','bleu'], 'Blanc':['white','blanc'], 'Gold':['gold','doré'], 'Rose':['rose'], 'Vert':['vert','green'], 'Rouge':['rouge','red']}
            for c_name, keywords in colors.items():
                if any(k in p_name for k in keywords):
                    color = c_name
                    break

            price_str = product.get('price', '0')
            price_clean = "".join(filter(lambda x: x.isdigit() or x == '.', price_str.replace(',', '.')))
            price_float = float(price_clean) if price_clean else 0.0
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, product['url']))

            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    'name': product['name'], 'price': price_float, 'url': product['url'],
                    'image': product['image'], 'display_price': product.get('price', 'N/A'),
                    'category': category, 'brand': brand_name, 'color': color
                }
            ))
        
        total = 0
        for i in range(0, len(points), 100):
            self.client.upsert(self.collection_name, points[i:i+100], wait=True)
            total += len(points[i:i+100])
        return total

    def run_multi_site(self, targets, pages_per_site=10):
        all_sites_products = [] # We collect everything here
        total_uploaded = 0
        
        for target in targets:
            brand = target["brand"]
            for base_url in target["urls"]:
                logger.info(f"🚀 Starting {brand} scrape...")
                
                for p in range(1, pages_per_site + 1):
                    # 1. Define the URL specifically for THIS page
                    # Logic: MyTek uses ?p=, others use ?page=
                    param = "p" if brand == "MyTek" else "page"
                    url = f"{base_url}?{param}={p}" # <--- 'url' is defined HERE
                    
                    logger.info(f"🔍 Fetching {brand} Page {p}: {url}")
                    
                    # 2. Scrape THIS specific page
                    prods = self.scrape_products(url, brand) 
                    
                    if prods:
                        all_sites_products.extend(prods)
                        logger.info(f"✅ Found {len(prods)} products on page {p}")
                    else:
                        logger.warning(f"⚠️ No products found on {brand} page {p}")
                        break # Stop if we hit an empty page
                        
                    time.sleep(2) # Be polite to the servers

        # 3. After ALL loops are done, save and upload the big list
        if all_sites_products:
            self.save_to_json(all_sites_products)
            # Pass "Multi-Source" or the brand if you prefer
            total_uploaded = self.upload_to_qdrant(all_sites_products) 
        
        return len(all_sites_products), total_uploaded

if __name__ == "__main__":
    # 1. Define our targets with their Brand Names
    # This allows the scraper to tag each product with the correct source
    targets = [
        {
            "brand": "Tunisianet",
            "urls": [
                'https://www.tunisianet.com.tn/596-smartphone-tunisie',
                'https://www.tunisianet.com.tn/376-telephonie-tablette',
                'https://www.tunisianet.com.tn/377-telephone-portable-tunisie',
                'https://www.tunisianet.com.tn/301-pc-portable-tunisie',
                'https://www.tunisianet.com.tn/525-refrigerateur-tunisie',
                'https://www.tunisianet.com.tn/338-casque-ecouteurs'
            ]
            "brand": "spacenet",
            "urls": ['https://spacenet.tn/193-lave-vaisselle-tunisie']  
        },
        
    ]
    
    # 2. Initialize the Scraper
    scraper = TunisianetScraper()
    
    # 3. Run the multi-site logic
    # We pass the targets list and tell it how many pages to scrape per URL
    scraped, uploaded = scraper.run_multi_site(targets, 10)
    
    print(f"""
    ================================
    🎉 Multi-Site Scraping Complete!
    ================================
    Total Scraped: {scraped}
    Total Uploaded: {uploaded}
    Check your Qdrant Dashboard to see the sources!
    ================================
    """)