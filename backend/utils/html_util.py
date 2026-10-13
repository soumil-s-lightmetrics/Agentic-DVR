import re

def convert_p_to_div_with_style(html: str) -> str:
    # Replace opening <p> tags (with or without attributes)
    html = re.sub(r'<p(\s[^>]*)?>', r'<div style="margin: 0 0 1em 0;">', html, flags=re.IGNORECASE)

    # Replace closing </p> tags
    html = re.sub(r'</p>', r'</div>', html, flags=re.IGNORECASE)

    return html
