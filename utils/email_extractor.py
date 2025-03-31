import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse

def is_valid(url):
    """Check if the URL is valid."""
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def get_all_website_links(url, domain_name, visited):
    """Extract all internal links from a page."""
    urls = set()
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        return urls

    soup = BeautifulSoup(response.text, "html.parser")
    # Use find_all instead of deprecated findAll
    for a_tag in soup.find_all("a", href=True):
        href = a_tag.attrs.get("href")
        if not href:
            continue
        href = urljoin(url, href)
        parsed_href = urlparse(href)
        href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path
        # Stay within the same domain
        if domain_name not in href:
            continue
        if href not in visited:
            urls.add(href)
    return urls

def extract_emails(url):
    """Extract email addresses from the given URL's content."""
    emails = set()
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        return emails

    soup = BeautifulSoup(response.text, "html.parser")

    # 1. Extract emails from mailto: links
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if href.startswith("mailto:"):
            email = href.split("mailto:")[1].split('?')[0].strip()
            emails.add(email)

    # 2. Extract emails from individual text nodes
    text_nodes = soup.find_all(text=True)
    full_text = " ".join(text_nodes)
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    found_emails = re.findall(email_pattern, full_text)
    for email in found_emails:
        emails.add(email.strip())

    return emails

def crawl_website(start_url, max_pages=100):
    """Crawl the website starting at start_url and extract emails."""
    visited = set()
    emails_found = set()
    domain_name = urlparse(start_url).netloc
    pages_to_visit = [start_url]

    while pages_to_visit and len(visited) < max_pages:
        url = pages_to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)
        print("Crawling:", url)
        emails = extract_emails(url)
        new_emails = emails - emails_found
        if new_emails:
            print("New emails found:", new_emails)
        emails_found.update(new_emails)
        links = get_all_website_links(url, domain_name, visited)
        pages_to_visit.extend(links - visited)
    return emails_found

if __name__ == "__main__":
    # Replace with your college's website URL
    start_url = input("Enter the website URL: ")
    all_emails = crawl_website(start_url, max_pages=200)
    print("All emails found:", all_emails)
