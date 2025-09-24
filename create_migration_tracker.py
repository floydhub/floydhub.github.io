#!/usr/bin/env python3
"""
Create migration tracker from Ghost export
"""

import json
from datetime import datetime

def get_author_info(post_id, ghost_data):
    """Get author information for a post"""
    db_data = ghost_data['db'][0]['data']

    # Find author ID from posts_authors relationship
    author_id = None
    for relationship in db_data['posts_authors']:
        if relationship['post_id'] == post_id:
            author_id = relationship['author_id']
            break

    if not author_id:
        return 'Unknown'

    # Find author details from users
    for user in db_data['users']:
        if user['id'] == author_id:
            return user.get('name', 'Unknown Author')

    return 'Unknown'

def create_migration_tracker():
    with open('floydhub-blog.ghost.2025-09-24-21-16-37.json', 'r') as f:
        ghost_data = json.load(f)

    # Get posts and sort by published date (newest first)
    posts = ghost_data['db'][0]['data']['posts']
    posts = sorted(posts, key=lambda x: x['published_at'], reverse=True)

    print(f'Total posts: {len(posts)}')

    # Create markdown content
    md_content = []
    md_content.append('# FloydHub Blog Migration Tracker')
    md_content.append('')
    md_content.append(f'**Total Posts**: {len(posts)}  ')
    md_content.append('**Status**: 🔄 In Progress  ')
    md_content.append('**Updated**: ' + datetime.now().strftime('%Y-%m-%d %H:%M'))
    md_content.append('')
    md_content.append('## Progress Overview')
    md_content.append('')
    md_content.append('- ✅ **Completed**: 2 posts')
    md_content.append('- 🔄 **In Progress**: 0 posts')
    md_content.append('- ⏳ **Pending**: 81 posts')
    md_content.append('')
    md_content.append('## Instructions')
    md_content.append('')
    md_content.append('- Update status: ⏳ → 🔄 → ✅')
    md_content.append('- Add notes for any special handling needed')
    md_content.append('- Completed posts should have Jekyll URLs working')
    md_content.append('')
    md_content.append('## Post List')
    md_content.append('')
    md_content.append('| Status | Title | Slug | Author | Date | Jekyll URL | Notes |')
    md_content.append('|--------|-------|------|--------|------|------------|-------|')

    completed_count = 0

    # Add each post
    for post in posts:
        # Parse date
        dt = datetime.fromisoformat(post['published_at'].replace('Z', '+00:00'))
        date_str = dt.strftime('%Y-%m-%d')

        # Get author
        author = get_author_info(post['id'], ghost_data)

        # Determine status
        if post['slug'] == 'metrics-on-floydhub':
            status = '✅'
            notes = 'Test post with images'
            jekyll_url = f'[/{post["slug"]}/](https://floydhub.github.io/{post["slug"]}/)'
            completed_count += 1
        elif post['slug'] == 'gpt2':
            status = '✅'
            notes = 'Test post with external images'
            jekyll_url = f'[/{post["slug"]}/](https://floydhub.github.io/{post["slug"]}/)'
            completed_count += 1
        else:
            status = '⏳'
            notes = ''
            jekyll_url = ''

        # Clean title for markdown (escape pipes)
        title = post['title'].replace('|', '\\|')

        md_content.append(f'| {status} | {title} | `{post["slug"]}` | {author} | {date_str} | {jekyll_url} | {notes} |')

    # Update progress overview
    md_content[8] = f'- ✅ **Completed**: {completed_count} posts'
    md_content[10] = f'- ⏳ **Pending**: {len(posts) - completed_count} posts'

    # Write to file
    with open('MIGRATION_TRACKER.md', 'w') as f:
        f.write('\n'.join(md_content))

    print(f'✅ Created MIGRATION_TRACKER.md')
    print(f'   - Total posts: {len(posts)}')
    print(f'   - Completed: {completed_count}')
    print(f'   - Pending: {len(posts) - completed_count}')

if __name__ == "__main__":
    create_migration_tracker()