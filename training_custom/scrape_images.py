from playwright.sync_api import sync_playwright
from PIL import Image
from io import BytesIO

import os
import base64
import requests

cwd = os.getcwd()
output_folder = cwd + '/downloads'
subfolder = None

def scrape_images(keyword: str, page_limit: int=1):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto('https://images.google.com/')
        
        search_input = page.locator('textarea[title="Search"]')
        search_input.fill(keyword)
        search_input.press('Enter')

        page.wait_for_load_state('load')

        ticker = 0
        seen_srcs = set()

        for _ in range(page_limit):
            images = page.query_selector_all('img')
            for image in images:
                src = image.get_attribute('src')
                if src is None:
                    continue

                if src in seen_srcs:
                    continue

                if "data:image/jpeg;base64" in src:
                    b64_data = src.split(',')[1]
                    img = Image.open(BytesIO(base64.b64decode(b64_data)))
                    img.save(f'{subfolder}/{keyword.replace(" ", "_")}_{ticker:04}.jpg')
                    ticker += 1
                    seen_srcs.add(src)
                elif "http" in src:
                    # skip any src that contains "favicon", "logo"
                    if "favicon" in src or "logo" in src:
                        continue

                    img = requests.get(src)
                    try:
                        img = Image.open(BytesIO(img.content))
                    except:
                        print(f"Failed to download {src}")


                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(f'{subfolder}/{keyword.replace(" ", "_")}_{ticker:04}.jpg')
                    ticker += 1
                    seen_srcs.add(src)
                else:
                    continue

            # if there's a "Show more results" button, click it, else scroll down
            show_more_button = page.query_selector('input[value="Show more results"]')
            if show_more_button.is_visible():
                show_more_button.click()
                page.wait_for_load_state('load')
            else:
                page.evaluate('window.scrollBy(0, window.innerHeight)')
                page.wait_for_load_state('load')

        browser.close()

if __name__ == "__main__":
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    keyword = input("Enter keyword: ")
    page_limit = int(input("Enter page limit: "))

    subfolder = output_folder + '/' + keyword.replace(" ", "_")
    if not os.path.exists(subfolder):
        os.mkdir(subfolder)

    scrape_images(keyword, page_limit)
    print("Done scraping images!")
    print(f"Images are saved in {subfolder}")
    print("Check the images as this scraper will download unrelated images as well.")
