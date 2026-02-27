---
name: web-research
description: Gather and summarize information from web pages. Use when user asks for current events, external docs, or URL-based analysis.
---

# Web Research Skill

## Workflow
1. Identify key facts to verify.
2. Fetch relevant pages via `web_request`.
3. Compare multiple sources when possible.
4. Call `respond_to_user` with concise findings and any uncertainty.

## Best Practices
- Prefer primary sources.
- Include caveats when sources conflict.
- Quote critical lines briefly rather than pasting entire pages.
