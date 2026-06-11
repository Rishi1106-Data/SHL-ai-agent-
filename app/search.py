from data_loader import load_catalog


def search_assessments(query):
    data = load_catalog()

    results = []

    query = query.lower()

    for item in data:
        text = (
            item.get("name", "") +
            " " +
            item.get("description", "")
        ).lower()

        if query in text:
            results.append(item)

    return results[:5]


if __name__ == "__main__":
    results = search_assessments("java")

    print(f"Found {len(results)} matches\n")

    for r in results:
        print(r["name"])