import requests

SEARXNG_URL = "http://localhost:8888/search"  

def search_searxng(query, category="general", num_results=5):
    """Query SearXNG and return results."""
    params = {
        "q": query,
        "format": "json",
        "categories": category,
        "safesearch": 1, 
        "language": "en"
    }
    try:
        response = requests.get(SEARXNG_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", [])[:num_results]:
            results.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("content")
            })
        return results

    except Exception as e:
        print(f"Error: {e}")
        return []

if __name__ == "__main__":
    query = "federated learning"
    results = search_searxng(query, category="science", num_results=5)

    print(f"\n🔎 Top results for: {query}\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}\n")
