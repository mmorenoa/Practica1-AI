from selenium import webdriver
from selenium.webdriver.common.by import By

driver = webdriver.Chrome()
driver.get("https://www.computer.org/csdl/journal/tp/2023/03")
titulo = driver.find_elements(By.CLASS_NAME, "article-title")
enlaces = []
for item in titulo:
    enlaces = item.get_attribute("href")
driver.quit()
for item in enlaces:
    print(item)
    """driver = webdriver.Chrome()
    driver.get(enlace.text)
    titulo = driver.find_element(By.CLASS_NAME, 'article-title')
    metadatos = driver.find_element(By.CLASS_NAME, 'article-metadata')
    keywords = driver.find_element(By.TAG_NAME, 'csdl-article-keywords')
    abstract = driver.find_element(By.TAG_NAME, 'article')
    print(titulo.text)
    print(metadatos.text)
    print(keywords.text)
    print(abstract.text)
    driver.quit()"""