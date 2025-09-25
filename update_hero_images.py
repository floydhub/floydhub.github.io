#!/usr/bin/env python3
"""
Update all existing Jekyll posts with hero images from Ghost export
Downloads hero images and updates frontmatter
"""

import json
import os
import re
import yaml
import requests
from urllib.parse import urlparse
import time

def download_image(url, local_path):
    """Download an image from URL to local path"""
    try:
        # Convert Ghost URLs
        if "floydhub.github.io" in url:
            url = url.replace("https://floydhub.github.io", "https://floydhub.ghost.io")
        elif url.startswith("__GHOST_URL__"):
            url = url.replace("__GHOST_URL__", "https://floydhub.ghost.io")

        print(f"Downloading: {url}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download with headers to avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # Write image to file
        with open(local_path, 'wb') as f:
            f.write(response.content)

        print(f"✅ Downloaded: {os.path.basename(local_path)}")
        return True

    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return False

def update_post_with_hero(post_file, hero_image_url, post_slug):
    """Update Jekyll post with hero image"""
    try:
        # Read the existing post
        with open(post_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split frontmatter and content
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                frontmatter_str = parts[1]
                post_content = parts[2]
            else:
                print(f"❌ Invalid frontmatter in {post_file}")
                return False
        else:
            print(f"❌ No frontmatter found in {post_file}")
            return False

        # Parse frontmatter
        frontmatter = yaml.safe_load(frontmatter_str)

        # Skip if already has a local hero image
        if frontmatter.get('feature_image', '').startswith('/assets/images/hero/'):
            print(f"ℹ️  Post already has local hero image: {post_file}")
            return True

        # Download hero image
        if hero_image_url:
            # Get file extension
            parsed_url = urlparse(hero_image_url)
            file_ext = os.path.splitext(parsed_url.path)[1] or '.jpg'

            # Create local filename
            hero_filename = f"{post_slug}-hero{file_ext}"
            local_path = f"assets/images/hero/{hero_filename}"
            jekyll_path = f"/assets/images/hero/{hero_filename}"

            if download_image(hero_image_url, local_path):
                # Update frontmatter
                frontmatter['feature_image'] = jekyll_path

                # Write updated post
                new_content = "---\n"
                new_content += yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
                new_content += "---" + post_content

                with open(post_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)

                print(f"✅ Updated {post_file} with hero image")
                return True
            else:
                print(f"❌ Failed to download hero image for {post_file}")
                return False
        else:
            print(f"ℹ️  No hero image URL for {post_file}")
            return True

    except Exception as e:
        print(f"❌ Error updating {post_file}: {e}")
        return False

def main():
    # Load Ghost export to get hero images
    print("Loading Ghost export...")
    with open('floydhub-blog.ghost.2025-09-24-21-16-37.json', 'r', encoding='utf-8') as f:
        ghost_data = json.load(f)

    posts = ghost_data['db'][0]['data']['posts']

    # Create hero images directory
    os.makedirs('assets/images/hero', exist_ok=True)

    # Process each post
    updated_count = 0
    total_count = 0

    for ghost_post in posts:
        post_slug = ghost_post['slug']
        hero_image_url = ghost_post.get('feature_image')

        # Find corresponding Jekyll post
        jekyll_files = [f for f in os.listdir('_posts') if f.endswith(f'-{post_slug}.md')]

        if not jekyll_files:
            print(f"⚠️  No Jekyll file found for slug: {post_slug}")
            continue

        jekyll_file = f"_posts/{jekyll_files[0]}"
        total_count += 1

        print(f"\n📝 Processing: {jekyll_file}")

        if update_post_with_hero(jekyll_file, hero_image_url, post_slug):
            updated_count += 1

        # Small delay between downloads
        time.sleep(0.5)

    print(f"\n🎉 Completed! Updated {updated_count}/{total_count} posts with hero images.")

if __name__ == "__main__":
    main()