import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
import os

def get_headers():
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
    ]
    return {
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

def extract_bhk(text):
    if not text: return "N/A"
    match = re.search(r'(\d+)\s*BHK', text, re.IGNORECASE)
    if match:
        return f"{match.group(1)} BHK"
    return "N/A"

def extract_bathrooms(text):
    if not text: return "N/A"
    match = re.search(r'(\d+)\s*Bath', text, re.IGNORECASE)
    if match:
        return match.group(1)
    return "N/A"

def extract_furnishing(text):
    if not text: return "N/A"
    text = text.lower()
    if 'unfurnished' in text or 'un-furnished' in text:
        return 'Unfurnished'
    elif 'semi-furnished' in text or 'semifurnished' in text:
        return 'Semi-Furnished'
    elif 'fully furnished' in text or 'furnished' in text:
        return 'Furnished'
    return "N/A"

def extract_seller_type(text):
    if not text: return "N/A"
    text = text.lower()
    if 'owner' in text:
        return 'Owner'
    elif 'agent' in text or 'dealer' in text or 'broker' in text:
        return 'Agent'
    elif 'builder' in text or 'developer' in text:
        return 'Builder'
    return "N/A"

# --- SOURCE 1: HOUSING.COM ---
def scrape_housing_locality(city_name, locality_name, max_pages=20):
    print(f"  Scraping Housing.com: {locality_name}...")
    all_properties = []
    
    for page in range(1, max_pages + 1):
        url = f"https://housing.com/in/buy/{city_name.lower()}/{locality_name.lower().replace(' ', '-')}-ahmedabad?page={page}"
        try:
            response = requests.get(url, headers=get_headers(), timeout=10)
            if response.status_code != 200: break
            
            soup = BeautifulSoup(response.content, 'html.parser')
            listings = soup.find_all('article')
            if not listings: listings = soup.find_all('div', class_=lambda x: x and 'article' in x)
            
            if not listings: break
            
            for listing in listings:
                try:
                    data = {}
                    raw_text = listing.get_text(separator=' | ', strip=True)
                    
                    title_tag = listing.find('h2') or listing.find('h3') or listing.find('div', {'data-q': 'title'})
                    data['Property Title'] = title_tag.get_text(strip=True) if title_tag else "N/A"
                    
                    price_tag = listing.find('div', {'data-q': 'price'})
                    data['Price'] = price_tag.get_text(strip=True) if price_tag else "N/A"
                    
                    area_tag = listing.find('div', {'data-q': 'area'})
                    data['Area'] = area_tag.get_text(strip=True) if area_tag else "N/A"
                    
                    data['BHK'] = extract_bhk(data['Property Title'])
                    if data['BHK'] == "N/A": data['BHK'] = extract_bhk(raw_text)
                    
                    data['Bathrooms'] = extract_bathrooms(raw_text)
                    data['Furnishing'] = extract_furnishing(raw_text)
                    data['Seller Type'] = extract_seller_type(raw_text)
                    data['Locality'] = locality_name
                    data['City'] = city_name
                    data['Source'] = 'Housing.com'
                    data['Raw_Details'] = raw_text
                    all_properties.append(data)
                except: continue
            
            # print(f"    Page {page}: {len(listings)} items")
            time.sleep(random.uniform(0.5, 1.0))
            
        except Exception: break
    
    print(f"  > Collected {len(all_properties)} from {locality_name}")
    return all_properties

# --- SOURCE 2: MAGICBRICKS ---
def scrape_magicbricks(city_name, max_pages=50):
    print(f"  Scraping MagicBricks for {city_name}...")
    all_properties = []
    base_url = "https://www.magicbricks.com/property-for-sale/residential-real-estate"
    
    for page in range(1, max_pages + 1):
        url = f"{base_url}?proptype=Multistorey-Apartment,Builder-Floor-Apartment,Penthouse,Studio-Apartment&cityName={city_name}&page={page}"
        try:
            response = requests.get(url, headers=get_headers(), timeout=10)
            if response.status_code != 200: break
            
            soup = BeautifulSoup(response.content, 'html.parser')
            listings = soup.find_all('div', class_='mb-srp__card')
            if not listings: break
            
            for listing in listings:
                try:
                    data = {}
                    raw_text = listing.get_text(separator=' | ', strip=True)
                    
                    title_tag = listing.find('h2', class_='mb-srp__card--title')
                    data['Property Title'] = title_tag.get_text(strip=True) if title_tag else "N/A"
                    
                    price_tag = listing.find('div', class_='mb-srp__card__price--amount')
                    data['Price'] = price_tag.get_text(strip=True) if price_tag else "N/A"
                    
                    area_tag = listing.find('div', class_='mb-srp__card__summary--value')
                    data['Area'] = area_tag.get_text(strip=True) if area_tag else "N/A"
                    
                    data['Locality'] = "Ahmedabad"
                    if title_tag and " in " in title_tag.text:
                        data['Locality'] = title_tag.text.split(" in ")[-1].strip()
                        
                    data['BHK'] = extract_bhk(data['Property Title'])
                    data['Bathrooms'] = extract_bathrooms(raw_text)
                    data['Furnishing'] = extract_furnishing(raw_text)
                    data['Seller Type'] = extract_seller_type(raw_text)
                    data['City'] = city_name
                    data['Source'] = 'MagicBricks'
                    data['Raw_Details'] = raw_text
                    all_properties.append(data)
                except: continue
            
            if page % 5 == 0: print(f"    MB Page {page}: {len(all_properties)} total so far")
            time.sleep(random.uniform(1.0, 2.0))
        except: break
        
    return all_properties

# --- SOURCE 3: PROPTIGER ---
def scrape_proptiger(city_name, max_pages=50):
    print(f"  Scraping PropTiger for {city_name}...")
    all_properties = []
    
    for page in range(1, max_pages + 1):
        url = f"https://www.proptiger.com/{city_name.lower()}/property-sale?page={page}"
        try:
            response = requests.get(url, headers=get_headers(), timeout=10)
            if response.status_code != 200: break
            
            soup = BeautifulSoup(response.content, 'html.parser')
            listings = soup.find_all('div', class_=lambda x: x and 'project-card-main' in x) # Approximate class
            if not listings: 
                # Fallback to article or other common tags
                listings = soup.find_all('article')
            
            if not listings: break
            
            for listing in listings:
                try:
                    data = {}
                    raw_text = listing.get_text(separator=' | ', strip=True)
                    
                    # PropTiger structure varies, using generic extraction for robustness
                    data['Property Title'] = "Property in " + city_name
                    title_elem = listing.find('a', class_='project-name')
                    if title_elem: data['Property Title'] = title_elem.get_text(strip=True)
                    
                    price_elem = listing.find('div', class_='price')
                    data['Price'] = price_elem.get_text(strip=True) if price_elem else "N/A"
                    
                    area_elem = listing.find('div', class_='size')
                    data['Area'] = area_elem.get_text(strip=True) if area_elem else "N/A"
                    
                    data['BHK'] = extract_bhk(raw_text)
                    data['Bathrooms'] = extract_bathrooms(raw_text)
                    data['Furnishing'] = extract_furnishing(raw_text)
                    data['Seller Type'] = extract_seller_type(raw_text)
                    data['Locality'] = city_name # Refine if possible
                    loc_elem = listing.find('div', class_='loc-name')
                    if loc_elem: data['Locality'] = loc_elem.get_text(strip=True)
                    
                    data['City'] = city_name
                    data['Source'] = 'PropTiger'
                    data['Raw_Details'] = raw_text
                    all_properties.append(data)
                except: continue
                
            if page % 5 == 0: print(f"    PT Page {page}: {len(all_properties)} total so far")
            time.sleep(random.uniform(1.0, 2.0))
        except: break
    return all_properties

# --- SOURCE 4: MAKAAN.COM ---
def scrape_makaan(city_name, max_pages=20):
    print(f"  Scraping Makaan.com for {city_name}...")
    all_properties = []
    
    for page in range(1, max_pages + 1):
        url = f"https://www.makaan.com/{city_name.lower()}-residential-property/buy-property-in-{city_name.lower()}-city?page={page}"
        try:
            response = requests.get(url, headers=get_headers(), timeout=10)
            if response.status_code != 200: break
            
            soup = BeautifulSoup(response.content, 'html.parser')
            listings = soup.find_all('div', class_='cardLayout')
            
            if not listings: break
            
            for listing in listings:
                try:
                    data = {}
                    raw_text = listing.get_text(separator=' | ', strip=True)
                    
                    title_tag = listing.find('a', class_='typelink') or listing.find('div', class_='title-line')
                    data['Property Title'] = title_tag.get_text(strip=True) if title_tag else "N/A"
                    
                    price_tag = listing.find('div', {'data-type': 'price-link'})
                    data['Price'] = price_tag.get_text(strip=True) if price_tag else "N/A"
                    
                    area_tag = listing.find('td', class_='lbl', string='Area')
                    if area_tag:
                        val_tag = area_tag.find_next_sibling('td', class_='val')
                        data['Area'] = val_tag.get_text(strip=True) if val_tag else "N/A"
                    else:
                        data['Area'] = "N/A"
                    
                    data['BHK'] = extract_bhk(data['Property Title'])
                    data['Bathrooms'] = extract_bathrooms(raw_text)
                    data['Furnishing'] = extract_furnishing(raw_text)
                    data['Seller Type'] = extract_seller_type(raw_text)
                    
                    loc_tag = listing.find('span', itemprop='addressLocality')
                    data['Locality'] = loc_tag.get_text(strip=True) if loc_tag else city_name
                    
                    data['City'] = city_name
                    data['Source'] = 'Makaan.com'
                    data['Raw_Details'] = raw_text
                    all_properties.append(data)
                except: continue
            
            if page % 5 == 0: print(f"    Makaan Page {page}: {len(all_properties)} total so far")
            time.sleep(random.uniform(1.0, 2.0))
        except: break
    return all_properties

# --- SOURCE 5: COMMONFLOOR.COM ---
def scrape_commonfloor(city_name, max_pages=20):
    print(f"  Scraping CommonFloor for {city_name}...")
    all_properties = []
    
    for page in range(1, max_pages + 1):
        url = f"https://www.commonfloor.com/{city_name.lower()}-property/for-sale?page={page}"
        try:
            response = requests.get(url, headers=get_headers(), timeout=10)
            if response.status_code != 200: break
            
            soup = BeautifulSoup(response.content, 'html.parser')
            listings = soup.find_all('div', class_='snb-tile')
            
            if not listings: break
            
            for listing in listings:
                try:
                    data = {}
                    raw_text = listing.get_text(separator=' | ', strip=True)
                    
                    title_tag = listing.find('div', class_='st_title') or listing.find('h2')
                    data['Property Title'] = title_tag.get_text(strip=True) if title_tag else "N/A"
                    
                    price_tag = listing.find('span', class_='s_p')
                    data['Price'] = price_tag.get_text(strip=True) if price_tag else "N/A"
                    
                    area_tag = listing.find('div', class_='s_a')
                    data['Area'] = area_tag.get_text(strip=True) if area_tag else "N/A"
                    
                    data['BHK'] = extract_bhk(data['Property Title'])
                    data['Bathrooms'] = extract_bathrooms(raw_text)
                    data['Furnishing'] = extract_furnishing(raw_text)
                    data['Seller Type'] = extract_seller_type(raw_text)
                    
                    # Locality extraction might be tricky, often in title
                    data['Locality'] = city_name
                    if " in " in data['Property Title']:
                         data['Locality'] = data['Property Title'].split(" in ")[-1].strip()

                    data['City'] = city_name
                    data['Source'] = 'CommonFloor'
                    data['Raw_Details'] = raw_text
                    all_properties.append(data)
                except: continue
                
            if page % 5 == 0: print(f"    CF Page {page}: {len(all_properties)} total so far")
            time.sleep(random.uniform(1.0, 2.0))
        except: break
    return all_properties

def main():
    target_city = 'Ahmedabad'
    print(f"STARTING HIGH-VOLUME SCRAPER FOR {target_city.upper()}")
    all_data = []
    
    # Load existing data to preserve it
    filename = "ahmedabad_real_estate_raw_data.csv"
    if os.path.exists(filename):
        try:
            existing_df = pd.read_csv(filename)
            print(f"Loaded {len(existing_df)} existing rows.")
        except Exception as e:
            print(f"Could not load existing data: {e}")
            existing_df = pd.DataFrame()
    else:
        existing_df = pd.DataFrame()

    # --- REORDERED EXECUTION: NEW SOURCES FIRST ---
    
    # 4. Makaan.com (New Source)
    print("\n--- Source 1: Makaan.com ---")
    makaan_data = scrape_makaan(target_city, max_pages=30) # Reduced pages for speed
    if makaan_data:
        all_data.extend(makaan_data)
        save_incremental(all_data, filename) # Save immediately
    
    # 5. CommonFloor (New Source)
    print("\n--- Source 2: CommonFloor ---")
    cf_data = scrape_commonfloor(target_city, max_pages=30)
    if cf_data:
        all_data.extend(cf_data)
        save_incremental(all_data, filename)

    # 3. PropTiger
    print("\n--- Source 3: PropTiger ---")
    pt_data = scrape_proptiger(target_city, max_pages=30)
    if pt_data:
        all_data.extend(pt_data)
        save_incremental(all_data, filename)

    # 2. MagicBricks
    print("\n--- Source 4: MagicBricks ---")
    mb_data = scrape_magicbricks(target_city, max_pages=30)
    if mb_data:
        all_data.extend(mb_data)
        save_incremental(all_data, filename)

    # 1. Housing.com (Reduced scope for speed)
    print("\n--- Source 5: Housing.com ---")
    # Reduced list of localities to just a few key ones for variety
    localities = ['Bopal', 'Satellite', 'Gota', 'Thaltej', 'Vastrapur', 'Maninagar', 'Naroda'] 
    for loc in localities:
        housing_data = scrape_housing_locality(target_city, loc, max_pages=10) 
        all_data.extend(housing_data)
    save_incremental(all_data, filename)
    
    print("\n" + "="*60)
    print(f"SUCCESS! Data Collection Complete")
    print(f"Total Rows Collected in this run: {len(all_data)}")
    print("="*60)

def save_incremental(new_data, filename):
    if not new_data: return
    
    df = pd.DataFrame(new_data)
    
    # Standardize columns
    cols = ['Property Title', 'Price', 'Area', 'BHK', 'Bathrooms', 'Furnishing', 'Seller Type', 'Locality', 'City', 'Source', 'Raw_Details']
    for col in cols:
        if col not in df.columns: df[col] = "N/A"
    df = df[cols]
    
    if os.path.exists(filename):
        try:
            existing_df = pd.read_csv(filename)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
        except:
            combined_df = df
    else:
        combined_df = df
    
    # Deduplicate
    combined_df = combined_df.drop_duplicates(subset=['Property Title', 'Price', 'Area'])
    
    try:
        combined_df.to_csv(filename, index=False)
        print(f"  > Saved {len(combined_df)} unique rows to {filename}")
    except PermissionError:
        print(f"  > Warning: Could not save to {filename} (Permission Denied)")

if __name__ == "__main__":
    main()
