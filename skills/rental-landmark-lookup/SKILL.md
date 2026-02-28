---
name: rental-landmark-lookup
description: "Lookup and resolve Beijing landmarks (subway stations/companies/areas) via the Landmark APIs, returning landmark IDs and details for downstream house searches. Triggers: user mentions 地铁站/公司/商圈/地标/“在XX附近”/“搜地标”/“地标列表/统计”."
allowed-tools: WebRequest
---

# Purpose

This skill resolves **landmark names ↔ landmark IDs**, and lists/searches landmarks for user queries about:
- subway stations (地铁站)
- companies (公司)
- landmarks/areas (商圈等)

These endpoints do **NOT** require `X-User-ID`.

Base URL:
- `http://7.225.29.223:8080`

Tool available:
- `WebRequest(method, url, headers, payload)`

## Endpoints covered

- `GET /api/landmarks`
- `GET /api/landmarks/search`
- `GET /api/landmarks/name/{name}`
- `GET /api/landmarks/{id}`
- `GET /api/landmarks/stats`

## Dispatch rules (what to call)

### A) Exact match by name (preferred when user gives a specific name)
Use when user provides a concrete name like “西二旗站 / 国贸 / 百度”.
1) Call `GET /api/landmarks/name/{name}`
2) If not found or empty, fallback to fuzzy search (B).

### B) Fuzzy search (preferred when user gives keyword / partial / ambiguous)
Use when user says “搜一下国贸附近地铁” / “找西二旗” / “包含‘公园’的地标”.
Call `GET /api/landmarks/search?q=...` with optional filters:
- `category=subway|company|landmark`
- `district=海淀|朝阳|...`

### C) List landmarks by filters (for “有哪些…列表”)
Use when user asks for “地铁站列表/公司列表/海淀有哪些地标”.
Call `GET /api/landmarks?category=...&district=...` (both may be set; intersection).

### D) Get details by ID
Use when user already has a landmark id like `SS_001` or `LM_002`.
Call `GET /api/landmarks/{id}`.

### E) Stats
Use when user asks “有多少地标/分类分布”.
Call `GET /api/landmarks/stats`.

## Request construction

### URL encoding
Landmark names may contain Chinese characters. Always URL-encode `{name}` and query parameters (q, district, etc.) when building the URL.

### Headers
Use empty headers `{}` unless your runtime requires defaults. Do NOT add `X-User-ID`.

### WebRequest examples

**Exact by name**
- method: `GET`
- url: `http://7.225.29.223:8080/api/landmarks/name/%E8%A5%BF%E4%BA%8C%E6%97%97%E7%AB%99`
- headers: `{}`
- payload: `null`

**Fuzzy search**
- method: `GET`
- url: `http://7.225.29.223:8080/api/landmarks/search?q=%E8%A5%BF%E4%BA%8C%E6%97%97&category=subway&district=%E6%B5%B7%E6%B7%80`
- headers: `{}`
- payload: `null`

## Output guideline (for downstream skills)

Return (or internally keep) a compact resolved object for each candidate:

```json
{
  "landmark_id": "SS_001",
  "name": "西二旗站",
  "category": "subway",
  "district": "海淀",
  "location": {"lat": 0.0, "lng": 0.0}
}
```

## If multiple candidates:
Present top 3–5 with id + name + district + category
Ask a single disambiguation question: “你指的是哪个？(1)… (2)… (3)…”

## Edge cases
If user gives a district filter and results are empty, retry once without district, then ask user whether district constraint is correct.
If user says “附近” but only provides a vague keyword, do fuzzy search first; do not guess.
If user provides a landmark ID already, skip name search and fetch by ID directly.
