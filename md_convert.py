# import markdown

# def convert_md_to_html_string(md_text):
#     # Convert markdown to HTML body content only
#     html_body = markdown.markdown(md_text, extensions=['fenced_code', 'codehilite', 'tables'])
#     return html_body

import markdown

def convert_md_to_html_string(md_text):
    html_body = markdown.markdown(md_text, extensions=['fenced_code', 'codehilite', 'tables'])

    # Add basic styles for readability
    styled_html = f"""
    <style>
        body {{
            font-family: 'Segoe UI', sans-serif;
            line-height: 1.6;
            font-size: 16px;
            color: #f8fafc;
        }}
        p {{
            margin-bottom: 16px;
        }}
        h1, h2, h3 {{
            margin-top: 24px;
            margin-bottom: 12px;
        }}
        ul, ol {{
            margin-left: 20px;
            margin-bottom: 16px;
        }}
        code {{
            background-color: #1e293b;
            padding: 2px 4px;
            border-radius: 4px;
            font-family: monospace;
        }}
        pre code {{
            display: block;
            padding: 12px;
            background-color: #1e293b;
            border-radius: 6px;
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 16px;
        }}
        th, td {{
            border: 1px solid #334155;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #334155;
        }}
    </style>
    <body>{html_body}</body>
    """

    return styled_html
