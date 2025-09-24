#!/usr/bin/env python3
"""
Ghost to Jekyll converter - Single post test
Converts the "Metrics on FloydHub" post from Ghost JSON export to Jekyll markdown
Includes image downloading functionality
"""

import json
import re
from datetime import datetime
from html2text import HTML2Text
import os
import requests
from urllib.parse import urlparse
import time

def convert_ghost_urls(content, base_url="https://floydhub.github.io"):
    """Convert Ghost __GHOST_URL__ placeholders to actual URLs"""
    return content.replace("__GHOST_URL__", base_url)

def convert_html_to_markdown(html_content):
    """Convert HTML content to markdown using html2text"""
    h = HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    h.ignore_emphasis = False
    h.body_width = 0  # Don't wrap lines

    # Clean up Ghost HTML artifacts
    html_content = html_content.replace("<!--kg-card-begin: markdown-->", "")
    html_content = html_content.replace("<!--kg-card-end: markdown-->", "")

    markdown = h.handle(html_content)

    # Clean up extra whitespace
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)

    return markdown.strip()

def format_date_for_jekyll(date_string):
    """Convert Ghost date to Jekyll date format"""
    # Parse Ghost date format: "2018-03-30T06:25:28.000Z"
    dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
    return dt.strftime('%Y-%m-%d %H:%M:%S %z')

def create_jekyll_filename(date_string, slug):
    """Create Jekyll filename format: YYYY-MM-DD-slug.md"""
    dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
    return f"{dt.strftime('%Y-%m-%d')}-{slug}.md"

def download_image(url, local_path, ghost_base_url="https://floydhub.ghost.io"):
    """Download an image from Ghost blog to local assets directory"""
    try:
        # Convert floydhub.github.io URLs back to ghost.io for downloading
        if "floydhub.github.io" in url:
            download_url = url.replace("https://floydhub.github.io", ghost_base_url)
        else:
            download_url = url

        print(f"Downloading: {download_url}")
        print(f"Saving to: {local_path}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download with headers to avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

        response = requests.get(download_url, headers=headers, timeout=30)
        response.raise_for_status()

        # Write image to file
        with open(local_path, 'wb') as f:
            f.write(response.content)

        print(f"✅ Downloaded: {os.path.basename(local_path)}")
        return True

    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return False

def process_images_in_content(content, ghost_base_url="https://floydhub.ghost.io"):
    """Download images and update their references in content"""

    # Find all image references
    img_pattern = r'!\[([^\]]*)\]\((https://floydhub\.github\.io/content/images/[^)]+)\)'
    images = re.findall(img_pattern, content)

    downloaded_images = []

    for alt_text, img_url in images:
        # Parse the image path
        parsed_url = urlparse(img_url)
        img_path = parsed_url.path

        # Create local path: /content/images/... -> assets/images/content/images/...
        local_path = f"assets/images{img_path}"

        # Download the image
        if download_image(img_url, local_path, ghost_base_url):
            # Update the content to use local path
            jekyll_path = f"/assets/images{img_path}"
            content = content.replace(img_url, jekyll_path)
            downloaded_images.append({
                'original': img_url,
                'local': local_path,
                'jekyll': jekyll_path
            })

        # Small delay between downloads
        time.sleep(1)

    return content, downloaded_images

def extract_test_post():
    """Extract the Metrics on FloydHub post from Ghost JSON"""

    # Load Ghost export
    with open('floydhub-blog.ghost.2025-09-24-21-16-37.json', 'r', encoding='utf-8') as f:
        ghost_data = json.load(f)

    # Find the Metrics post
    posts = ghost_data['db'][0]['data']['posts']
    metrics_post = None

    for post in posts:
        if post['title'] == "Metrics on FloydHub":
            metrics_post = post
            break

    if not metrics_post:
        print("Metrics post not found!")
        return

    print(f"Found post: {metrics_post['title']}")
    print(f"Published: {metrics_post['published_at']}")
    print(f"Slug: {metrics_post['slug']}")
    print(f"Feature image: {metrics_post.get('feature_image', 'None')}")

    # Extract content - prefer HTML over mobiledoc
    content_html = metrics_post.get('html', '')
    if not content_html:
        print("No HTML content found, trying mobiledoc...")
        mobiledoc = json.loads(metrics_post.get('mobiledoc', '{}'))
        # For this test, we'll use the HTML version
        content_html = metrics_post.get('html', '')

    # Convert content
    content_html = convert_ghost_urls(content_html)
    markdown_content = convert_html_to_markdown(content_html)

    # Process and download images
    print(f"\n🖼️  Processing images in post...")
    markdown_content, downloaded_images = process_images_in_content(markdown_content)

    if downloaded_images:
        print(f"✅ Downloaded {len(downloaded_images)} images:")
        for img in downloaded_images:
            print(f"   • {img['original']} → {img['jekyll']}")
    else:
        print("ℹ️  No images found to download")

    # Create Jekyll frontmatter
    frontmatter = {
        'layout': 'post',
        'title': metrics_post['title'],
        'date': format_date_for_jekyll(metrics_post['published_at']),
        'slug': metrics_post['slug'],
        'excerpt': metrics_post.get('custom_excerpt') or metrics_post.get('plaintext', '')[:200] + '...',
        'feature_image': metrics_post.get('feature_image'),
        'tags': ['metrics', 'monitoring', 'deep-learning']  # Add some relevant tags
    }

    # Remove None values
    frontmatter = {k: v for k, v in frontmatter.items() if v is not None}

    # Create Jekyll post file
    filename = create_jekyll_filename(metrics_post['published_at'], metrics_post['slug'])
    filepath = f"_posts/{filename}"

    # Generate Jekyll post content
    jekyll_content = "---\n"
    for key, value in frontmatter.items():
        if isinstance(value, str) and '"' in value:
            # Escape quotes in YAML strings
            escaped_value = value.replace('"', '\\"')
            jekyll_content += f'{key}: "{escaped_value}"\n'
        else:
            jekyll_content += f'{key}: "{value}"\n'
    jekyll_content += "---\n\n"
    jekyll_content += markdown_content

    # Write the file
    os.makedirs('_posts', exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(jekyll_content)

    print(f"\n✅ Created Jekyll post: {filepath}")
    print(f"Content preview:\n{markdown_content[:500]}...")

    return filepath

if __name__ == "__main__":
    try:
        filepath = extract_test_post()
        if filepath:
            print(f"\n🎉 Test post conversion completed: {filepath}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()