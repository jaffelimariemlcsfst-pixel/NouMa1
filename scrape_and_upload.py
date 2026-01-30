import requests
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from sentence_transformers import SentenceTransformer
import time
import logging
import uuid
import json
import os

URL = os.environ["QDRANT_URL"]
API_KEY = os.environ["QDRANT_API_KEY"]

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
        self.json_file = "products_backup.json"

        if not self.client.collection_exists(self.collection_name):
            logger.info(f"📦 Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )
        
        try:
            self.client.create_payload_index(self.collection_name, "price", PayloadSchemaType.FLOAT)
            self.client.create_payload_index(self.collection_name, "category", PayloadSchemaType.KEYWORD)
            self.client.create_payload_index(self.collection_name, "color", PayloadSchemaType.KEYWORD)
        except Exception:
            pass
            
        self.model = SentenceTransformer('clip-ViT-B-32')
        logger.info("✅ Scraper and Qdrant client initialized")

    def scrape_products(self, url, brand):
        try:
            time.sleep(2)
            response = self.session.get(url, headers=self.headers, timeout=15)
            
            if response.status_code != 200:
                logger.error(f"❌ Bad status code: {response.status_code}")
                return []

            soup = BeautifulSoup(response.content, 'html.parser')
            products = []

            # Define selectors per brand
            if brand == "Tunisianet":
                container = '.item-product'
                name_selector = 'h2.product-title a'
                price_selector = 'span[itemprop="price"]'
                img_selector = '.product-thumbnail img'
            elif brand == "spacenet":
                # Based on your original working code
                container = '.product-miniature'
                name_selector = 'a.product-title'  # Changed from '.href' 
                price_selector = '.price'
                img_selector = 'img'  # Changed from '.cover_image'
            else:
                logger.warning(f"Unknown brand: {brand}")
                return []

            items = soup.select(container)
            logger.info(f"🔍 {brand}: Found {len(items)} containers")
            
            if not items:
                logger.warning(f"⚠️ No containers found with selector: {container}")
                return []

            for idx, item in enumerate(items):
                try:
                    # Get name and link - try multiple approaches
                    name_elem = item.select_one(name_selector)
                    
                    # Fallback: if selector doesn't work, try finding ANY link with text
                    if not name_elem:
                        name_elem = item.find('a', class_='product-title') or \
                                   item.find('a', href=True, string=True) or \
                                   item.select_one('h3 a')
                    
                    if not name_elem:
                        logger.debug(f"Item {idx}: No name element found")
                        continue
                    
                    name = name_elem.get_text(strip=True)
                    link = name_elem.get('href', '')
                    
                    # Skip if no actual name
                    if not name or len(name) < 3:
                        continue
                    
                    # Make link absolute
                    if link and not link.startswith('http'):
                        if link.startswith('//'):
                            link = 'https:' + link
                        elif link.startswith('/'):
                            base = '/'.join(url.split('/')[:3])
                            link = base + link
                        else:
                            base_url = url.split('?')[0].rsplit('/', 1)[0]
                            link = base_url + '/' + link.lstrip('/')
                    
                    # Get price - try the selector and fallbacks
                    price_elem = item.select_one(price_selector)
                    if not price_elem:
                        price_elem = item.find(attrs={'itemprop': 'price'}) or \
                                    item.find(class_='product-price')
                    
                    price = price_elem.get_text(strip=True) if price_elem else "0"
                    
                    # Get image - try the selector and look for any img
                    img_elem = item.select_one(img_selector)
                    if not img_elem:
                        img_elem = item.find('img')
                    
                    image = ""
                    if img_elem:
                        image = (img_elem.get('src') or 
                                img_elem.get('data-src') or 
                                img_elem.get('data-lazy-src') or "")
                        
                        # Make image absolute
                        if image and not image.startswith('http'):
                            if image.startswith('//'):
                                image = 'https:' + image
                            elif image.startswith('/'):
                                base = '/'.join(url.split('/')[:3])
                                image = base + image

                    # Only add if we have minimum required data
                    if name and link:
                        products.append({
                            'name': name, 
                            'price': price, 
                            'url': link, 
                            'image': image,
                            'brand': brand  # Track which brand this came from
                        })
                        
                        # Log first few for debugging
                        if idx < 3:
                            logger.debug(f"  ✓ Product {idx+1}: {name[:40]}... | Price: {price}")
                    
                except Exception as e:
                    logger.debug(f"Error parsing item {idx}: {e}")
                    continue

            logger.info(f"✅ {brand}: Successfully scraped {len(products)} products")
            return products

        except Exception as e:
            logger.error(f"❌ Error scraping {brand}: {e}")
            return []

    def save_to_json(self, products):
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(products, f, ensure_ascii=False, indent=4)
        logger.info(f"💾 Data saved locally to {self.json_file}")

    def upload_to_qdrant(self, products):
        if not products:
            try:
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    products = json.load(f)
                logger.info(f"📂 Loaded {len(products)} items from JSON for upload")
            except FileNotFoundError:
                logger.error("❌ No data found to upload!")
                return 0

        if not products:
            return 0

        product_names = [p['name'] for p in products]
        vectors = self.model.encode(product_names, show_progress_bar=True).tolist()
        
        points = []
        for product, vector in zip(products, vectors):
            p_name = product['name'].lower()
            
            # Enhanced category detection
            category = "Autre"
            if any(x in p_name for x in ['smartphone', 'iphone', 'samsung', 'galaxy', 'mobile']):
                category = "Smartphone"
            elif any(x in p_name for x in ['pc', 'laptop', 'ordinateur', 'macbook', 'lenovo', 'hp', 'dell', 'asus']):
                category = "Ordinateur"
            elif any(x in p_name for x in ['casque', 'écouteur', 'earphone', 'headphone', 'charger', 'cable', 'câble']):
                category = "Accessoires"
            elif any(x in p_name for x in ['lave', 'vaisselle', 'machine', 'laver', 'réfrigérateur', 'climatiseur', 'frigo']):
                category = "Électroménager"
            
            color = "Non spécifié"
            colors = {
                'Noir': ['noir', 'black', 'negro'],
                'Bleu': ['blue', 'bleu', 'azul'],
                'Blanc': ['white', 'blanc', 'blanco'],
                'Gold': ['gold', 'doré', 'or'],
                'Rose': ['rose', 'pink'],
                'Vert': ['vert', 'green'],
                'Rouge': ['rouge', 'red'],
                'Gris': ['gris', 'gray', 'grey']
            }
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
                    'name': product['name'],
                    'price': price_float,
                    'url': product['url'],
                    'image': product['image'],
                    'display_price': product.get('price', 'N/A'),
                    'category': category,
                    'brand': product.get('brand', 'Unknown'),  # Use brand from product data
                    'color': color
                }
            ))
        
        total = 0
        for i in range(0, len(points), 100):
            batch = points[i:i+100]
            self.client.upsert(self.collection_name, batch, wait=True)
            total += len(batch)
        
        logger.info(f"📤 Uploaded {total} products to Qdrant")
        return total

    def run_multi_site(self, targets, pages_per_site=100):
        all_sites_products = []
        
        for target in targets:
            brand = target["brand"]
            logger.info(f"\n{'='*60}\n🚀 Starting {brand} scrape\n{'='*60}")
            
            for base_url in target["urls"]:
                for p in range(1, pages_per_site + 1):
                    param = "p" if brand == "MyTek" else "page"
                    url = f"{base_url}?{param}={p}"
                    
                    logger.info(f"🔍 Page {p}: {url}")
                    
                    prods = self.scrape_products(url, brand)
                    
                    if prods:
                        all_sites_products.extend(prods)
                        logger.info(f"✅ Page {p}: +{len(prods)} products (Total: {len(all_sites_products)})")
                    else:
                        logger.warning(f"⚠️ Page {p}: No products - stopping pagination")
                        break
                        
                    time.sleep(2)

        total_uploaded = 0
        if all_sites_products:
            logger.info(f"\n{'='*60}\n💾 FINAL: {len(all_sites_products)} products scraped\n{'='*60}")
            self.save_to_json(all_sites_products)
            total_uploaded = self.upload_to_qdrant(all_sites_products)
        else:
            logger.error("❌ No products scraped!")
        
        return len(all_sites_products), total_uploaded

if __name__ == "__main__":
    targets = [
        {
            "brand": "Tunisianet",
            "urls": [
                'https://www.tunisianet.com.tn/596-smartphone-tunisie',
                'https://www.tunisianet.com.tn/376-telephonie-tablette',
                'https://www.tunisianet.com.tn/301-pc-portable-tunisie',
                'https://www.tunisianet.com.tn/650-smartwatch',
                'https://www.tunisianet.com.tn/462-telephone-fixe'
            ]
        },
        {
            "brand": "spacenet",
            "urls": ['https://spacenet.tn/193-lave-vaisselle-tunisie',
                     'https://spacenet.tn/218-accessoires-gamer-tunisie',
                     'https://spacenet.tn/221-clavier-gamer',
                     'https://spacenet.tn/180-electromenager-cuisine',
                     'https://spacenet.tn/8-imprimante-tunisie',
                     'https://spacenet.tn/74-pc-portable-tunisie',
                     'https://spacenet.tn/148-smartwatch-tunisie',
                     ]  
        },
    ]
    
    scraper = TunisianetScraper()
    scraped, uploaded = scraper.run_multi_site(targets, pages_per_site=100)
    
    print(f"""
    {'='*50}
    🎉 Multi-Site Scraping Complete!
    {'='*50}
    Total Scraped:  {scraped}
    Total Uploaded: {uploaded}
    {'='*50}
    """)

