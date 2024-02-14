from playwright.sync_api import sync_playwright
from PIL import Image
from io import BytesIO

import os
import base64
import requests

cwd = os.getcwd()
output_folder = cwd + '/downloads'
subfolder = None

def scrape_images(keyword: str, page_limit: int=1, start_page: int=-1):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto('https://images.google.com/')
        
        search_input = page.locator('textarea[title="Search"]')
        search_input.fill(keyword)
        search_input.press('Enter')

        page.wait_for_load_state('load')

        # we need to turn off safe search because of the image blur
        safe_search_button = page.get_by_role('button', name='SafeSearch')
        safe_search_button.click()

        # turn off safe search
        safe_search_off_button = page.get_by_role('menuitem', name='Off')
        safe_search_off_button.click()

        # wait for the page to load
        page.wait_for_load_state('load')
        ticker = 0
        seen_srcs = set()
        
        if start_page > 0:
            for _ in range(start_page):
                show_more_button = page.query_selector('input[value="Show more results"]')
                if show_more_button != None and show_more_button.is_visible():
                    show_more_button.click()
                    page.wait_for_load_state('load')
                else:
                    page.evaluate('window.scrollBy(0, window.innerHeight)')
                    page.wait_for_load_state('load')

        for _ in range(page_limit):
            images = page.query_selector_all('img')
            for image in images:
                src = image.get_attribute('src')
                if src is None:
                    continue

                if src in seen_srcs:
                    continue

                filename = f'{subfolder}/{keyword.replace(" ", "_")}_{ticker:04}.jpg'

                if os.path.exists(filename):
                    # increment the ticker until the filename is unique
                    while os.path.exists(filename):
                        ticker += 1
                        filename = f'{subfolder}/{keyword.replace(" ", "_")}_{ticker:04}.jpg'

                if "data:image/jpeg;base64" in src:
                    b64_data = src.split(',')[1]
                    img = Image.open(BytesIO(base64.b64decode(b64_data)))
                    img.save(filename)
                    ticker += 1
                    seen_srcs.add(src)
                elif "http" in src:
                    # skip any src that contains "favicon", "logo"
                    if "favicon" in src or "logo" in src:
                        continue

                    img = None

                    try:
                        img = requests.get(src)
                    except:
                        print(f"Failed to download {src}, error: {img}")

                    try:
                        img = Image.open(BytesIO(img.content))
                    except:
                        print(f"Failed to download {src}, downloaded content might be corrupted.")


                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(filename)
                    ticker += 1
                    seen_srcs.add(src)
                else:
                    continue

            # if there's a "Show more results" button, click it, else scroll down
            show_more_button = page.query_selector('input[value="Show more results"]')
            if show_more_button != None and show_more_button.is_visible():
                show_more_button.click()
                page.wait_for_load_state('load')
            else:
                page.evaluate('window.scrollBy(0, window.innerHeight)')
                page.wait_for_load_state('load')

            print(f"Page {_+1} of {page_limit} done\t {ticker} images saved")

        browser.close()

if __name__ == "__main__":
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    keyword = input("Enter keyword: ")
    page_limit = int(input("Enter page limit: "))
    start_page = int(input("Enter start page: "))

    subfolder = output_folder + '/' + keyword.replace(" ", "_")
    if not os.path.exists(subfolder):
        os.mkdir(subfolder)

    scrape_images(keyword, page_limit, start_page)
    print("Done scraping images!")
    print(f"Images are saved in {subfolder}")
    print("Check the images as this scraper will download unrelated images as well.")
