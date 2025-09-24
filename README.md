# FloydHub Blog

Jekyll-powered blog for FloydHub deep learning content.

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

## Deployment

Changes pushed to `gh-pages` branch automatically deploy to https://floydhub.github.io
