def concat_markdown_content(markdown_content: list[dict[str, str]]) -> str:
    """
    Concatenates multiple markdown content entries into one formatted string.
    """
    final_content = ""
    separator = "=" * 40
    for content_dict in markdown_content:
        url = content_dict["url"]
        markdown_text = content_dict["markdown_text"]
        title = content_dict["title"]
        final_content += (
            f"{separator}\n URL: {url}\n Page Title: {title}\n Markdown:\n"
            f"{separator}\n{markdown_text}\n"
        )
    return final_content
