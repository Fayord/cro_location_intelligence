from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from thefuzz import fuzz
from tqdm.notebook import tqdm
import time
import json


def create_driver(options=None):
    # Set up Selenium options
    if options is None:
        options = Options()
        # options.add_argument("--headless")  # Run in headless mode if you don't need to see the browser
        options.add_argument(
            "--disable-blink-features=AutomationControlled", "--disable-dev-shm-usage"
        )  # Avoid detection

    # options.add_argument("--no-sandbox")
    # options.add_argument("--disable-dev-shm-usage")
    # options.add_argument("--start-maximized")

    # Path to your ChromeDriver

    # Create a WebDriver instance
    cService = webdriver.ChromeService(executable_path=ChromeDriverManager().install())
    driver = webdriver.Chrome(service=cService, options=options)
    return driver


def setup_chrome_driver(window_size="1920,1080"):
    # keyword = "เจ๊ม่วย ก๋วยเตี๋ยวเป็ดสุดซอย"
    # weblink = (
    #     f"https://www.google.com/maps/search/{keyword}/@12.5596999, 99.9615208,17z"
    # )
    weblink = "https://www.google.com/maps/search/ร้านเจ้ม๋วยก๋วยเตี๋ยวเป็ด"
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--force-device-scale-factor=1.5")
    options.add_argument("--disable-dev-shm-usage")
    driver = create_driver(options)
    # Open the website
    driver.get(weblink)

    # Wait for Cloudflare's challenge to be bypassed
    time.sleep(1)  # Adjust the sleep time as needed based on the challenge duration

    # Example: Scraping the page title
    page_title = driver.title
    # print(f"Page Title: {page_title}")

    # Example: Scraping specific elements
    feeds = driver.find_elements(By.XPATH, "//div[@role='feed']")
    poi_title = None
    if len(feeds) > 0:
        feed_divs = feeds[0].find_elements(
            By.XPATH,
            "//div[contains(@jsaction, 'mouseover:pane') and contains(@jsaction, 'mouseout:pane')]",
        )
        count = 0

        for feed_div in feed_divs:
            if count == 1:
                feed_div.click()
                break

            if keyword in feed_div.text:
                count += 1

        poi_title = feed_divs[0].text

    review_starts = driver.find_elements(
        By.XPATH, "//div[contains(@jsaction, '.moreReviews')]"
    )
    if len(review_starts) > 0:
        review_starts[0].click()

    anchor_tag = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.XPATH, "//div[@role='tablist']"))
    )
    button_tags = anchor_tag.find_elements(By.TAG_NAME, "button")

    dead_letter = 0
    while len(button_tags) == 2 and dead_letter <= 10:
        time.sleep(0.2)
        button_tags = anchor_tag.find_elements(By.TAG_NAME, "button")
        dead_letter += 1

    if len(button_tags) == 3:
        driver.execute_script(
            "arguments[0].scrollIntoView({block: 'center'});", button_tags[1]
        )
        button_tags[1].click()
    if len(button_tags) == 4:
        driver.execute_script(
            "arguments[0].scrollIntoView({block: 'center'});", button_tags[2]
        )
        button_tags[2].click()

    relevant_element = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.XPATH, "//button[@aria-label='เกี่ยวข้องที่สุด']"))
    )
    relevant_element.click()

    menu_items = []
    WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.XPATH, "//div[@role='menuitemradio']"))
    )
    menu_items = driver.find_elements(By.XPATH, "//div[@role='menuitemradio']")
    menu_items[1].click()

    prev_len_scroll_element = -1
    sleep = 0.2
    scroll_elements = []
    while True:
        jslog = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located(
                (By.XPATH, "//div[@jslog='26354;mutable:true;']")
            )
        )
        scroll_elements = jslog.find_elements(By.TAG_NAME, "span")
        if len(scroll_elements) == prev_len_scroll_element:
            div_elements = jslog.find_elements(By.TAG_NAME, "div")
            if div_elements[-2].text == " " or sleep > 120:
                break
            time.sleep(sleep)
            sleep = sleep * 2
            continue
        sleep = 0.2
        driver.execute_script("arguments[0].scrollIntoView(true);", scroll_elements[-1])
        prev_len_scroll_element = len(scroll_elements)
        time.sleep(0.2)

    review_elements = driver.find_elements(By.CSS_SELECTOR, "div.fontBodyMedium")
    years = review_elements[-1].find_elements(By.TAG_NAME, "span")
    year_text = [year.text for year in years]

    all_reviews = [
        i.text.replace("\n", "|").replace("\t", ";") for i in review_elements
    ]
    last_review_text = ":".join(year_text).replace("\n", "|").replace("\t", ";")
    print(year_text)
    return driver


# Example usage
if __name__ == "__main__":
    # Create the driver\
    setup_chrome_driver()

    # Navigate to a website
    # driver.get("https://www.example.com")

    # Your automation code here

    # Don't forget to close the browser when done
    # driver.quit()
