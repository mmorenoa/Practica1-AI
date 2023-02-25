from selenium import webdriver
from selenium.webdriver.common.by import By
import re
from datetime import datetime

driver = webdriver.Chrome()
driver.get("https://www.computer.org/csdl/journal/tp/2023/03")
titulo = driver.find_elements(By.CLASS_NAME, "article-title")
aux = 1
for item in titulo:
    enlaces = item.get_attribute("href")
    print("\n----------")
    print(aux, " enlace: ", enlaces) 
    aux += 1
    print("----------")
    driver = webdriver.Chrome()
    driver.get(enlaces)
    patron = '.* ....,'
    titulo = driver.find_element(By.CLASS_NAME, 'article-title')
    metadatos = driver.find_element(By.CLASS_NAME, 'article-metadata')
    keywords = driver.find_element(By.TAG_NAME, 'csdl-article-keywords')
    abstract = driver.find_element(By.TAG_NAME, 'article')
    m = re.search(patron, metadatos.text)
    dates = m.group(0) #Queremos que sea 202303
    date_format = "%Y%m"
    date_object = datetime.strptime(dates, "%B %Y,")
    print("\nTITULO: ", titulo.text)
    #print("\nMETADATOS: ", metadatos.text) CREO QUE ESTO NO LO PIDEN, pero se usa para la fecha. Podriamos a lo mejor extraer solo los autores de aqui
    print("\nFECHA: ", date_object.strftime(date_format))
    print("\nABSTRACT: ", abstract.text)
    print("\nKEYWORDS: ", keywords.text)
    driver.quit()
    
