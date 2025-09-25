# FloydHub Blog

A modern Jekyll-powered blog for FloydHub's deep learning content, featuring SEO optimization, responsive design, and comprehensive content management.

## Features

- **SEO Optimized**: Includes jekyll-sitemap, jekyll-seo-tag, and robots.txt for search engine visibility
- **Responsive Design**: Modern, mobile-first design with hero images and card-based layouts
- **Content Migration**: Automated Ghost-to-Jekyll migration with image handling
- **Error Handling**: Custom 404 page with navigation and recent posts
- **Social Sharing**: Open Graph and Twitter Card meta tags for enhanced social media sharing

## Local Development

### Quick Start
```bash
./bin/run
```

This will:
- Install Jekyll dependencies
- Start local server at http://localhost:4000
- Enable live reload for instant updates
- Show the migrated post at http://localhost:4000/metrics-on-floydhub/

### Manual Setup
If you prefer manual control:

```bash
# Install dependencies
bundle install

# Start Jekyll server
bundle exec jekyll serve --livereload
```

### Troubleshooting

**Bundle install errors**: The Gemfile uses Jekyll 3.9 directly instead of `github-pages` to avoid native compilation issues on macOS.

**Port conflicts**: Jekyll runs on port 4000 by default. Kill any existing processes or modify the port in `bin/run`.

## Migration from Ghost

The `ghost_to_jekyll.py` script converts Ghost blog posts to Jekyll format:

- Downloads images locally to `assets/images/`
- Converts HTML to Markdown
- Creates proper Jekyll frontmatter
- Updates internal links to floydhub.github.io

## Project Structure

```
├── _config.yml          # Jekyll configuration with plugins
├── _layouts/
│   ├── default.html     # Main layout with SEO tags
│   └── post.html        # Blog post layout with hero images
├── _posts/              # Blog posts in Markdown format
├── assets/
│   ├── images/          # Site images and migrated content
│   └── css/             # Additional stylesheets
├── 404.html             # Custom error page
├── robots.txt           # Search engine crawler instructions
└── ghost_to_jekyll.py   # Migration script from Ghost CMS
```

## Jekyll Plugins

The site uses the following Jekyll plugins for enhanced functionality:

- **jekyll-feed**: Generates RSS/Atom feeds for blog posts
- **jekyll-sitemap**: Creates XML sitemap for search engines
- **jekyll-seo-tag**: Adds comprehensive SEO meta tags and structured data

## SEO Features

- **Sitemap**: Automatically generated at `/sitemap.xml`
- **Robots.txt**: Guides search engine crawlers
- **Meta Tags**: Open Graph, Twitter Cards, and JSON-LD structured data
- **Canonical URLs**: Prevents duplicate content issues
- **Rich Snippets**: Enhanced search result appearance

## Deployment

Changes pushed to `gh-pages` branch automatically deploy to https://floydhub.github.io

The site rebuilds automatically on GitHub Pages with the configured Jekyll plugins.
