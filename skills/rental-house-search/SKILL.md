---
name: rental-house-search
description: "Search and filter rentable houses using platform/community endpoints; converts user constraints (budget, district, bedrooms, subway, elevator, area, decoration, availability, commute) into query params and returns candidate house IDs. Triggers: user says 找房/筛选/预算/户型/整租合租/地铁/电梯/朝向/装修/通勤/可入住/平台(链家/安居客/58)/小区名查询."
allowed-tools: WebRequest
---

# Purpose

This skill performs **house discovery**:
- main filtering search via `by_platform`
- community-based lookup/disambiguation via `by_community`

It ALWAYS sends `X-User-ID: d00640449`.

Base URL:
- `http://7.225.29.223:8080`

Tool:
- `WebRequest(method, url, headers, payload)`

Required header:
- `X-User-ID: d00640449`

## Endpoints covered

- `GET /api/houses/by_platform`
- `GET /api/houses/by_community`

(Do NOT perform rent/terminate/offline actions in this skill.)

## Dispatch triggers (when to activate)

Activate this skill when the user asks to:
- 找房 / 筛选房源 / 推荐房源
- 限定预算、户型、整租/合租、装修、朝向、电梯、面积
- 地铁相关筛选（线路/站点/离地铁距离）
- 可入住时间筛选、通勤时间（如“到西二旗X分钟内”）
- 指定平台（链家/安居客/58同城）或“全平台对比”
- 按小区名查询“某小区有哪些可租房源”

## High-level workflow (small-model friendly)

1) Extract constraints from user text (only what is stated).
2) Normalize constraints into query parameters (rules below).
3) Call `GET /api/houses/by_platform` with:
   - `page=1`
   - `page_size=10` (or 20 if user wants broader)
4) If results are too many or too vague, ask ONE follow-up targeting the biggest uncertainty (usually budget, district/commute, rental_type, bedrooms).
5) Return candidate `house_id` list + short summaries (price/area/bedrooms/community/subway distance if present).

## When to use which endpoint

### A) `by_platform` (default search)

Use for almost all “找房/筛选” requests, especially with filters:
- district / area / price range / bedrooms / rental_type / subway / elevator / available date / commute, etc.

### B) `by_community` (community exact lookup)

Use when user provides a **specific community name** (小区名) like “建清园(南区)”:
- to find houses in that community
- to support “这个小区有房吗 / 小区信息核对 / 小区名消歧”

If user asks “这个小区在哪个区/离地铁近吗”等，仍先用 `by_community` 找到房源，再由其他技能（详情/周边）补充。

## Query parameter normalization rules

### listing_platform

- If user explicitly says 平台=链家/安居客/58同城, set `listing_platform` accordingly.
- If user says “全平台都看看/对比三家”:
  - Make 3 separate calls to `by_platform` with `listing_platform=链家`, `安居客`, `58同城`.
  - Merge results by `house_id` (dedupe) in your presentation.
- If not specified, omit it (API defaults to 安居客).

### district, area

- Accept multiple: split on `,` `，` `、` whitespace and re-join with comma.
  - Example: “海淀、朝阳” → `district=海淀,朝阳`
  - Example: “西二旗 上地” → `area=西二旗,上地`

### price

- “≤8000 / 不超过8000” → `max_price=8000`
- “≥6000 / 至少6000” → `min_price=6000`
- “6000-8000 / 6k-8k” → `min_price=6000&max_price=8000`

### bedrooms

Map common phrases:
- “一居/1居/1室” → `bedrooms=1`
- “两居/2居/2室” → `bedrooms=2`
- “一到两居/1-2居” → `bedrooms=1,2`

If truly ambiguous (e.g., “一两居都行” is fine), use a comma list.

### rental_type

- “整租” → `rental_type=整租`
- “合租” → `rental_type=合租`

### elevator (string boolean)

- “必须有电梯/有电梯” → `elevator=true`
- “不要电梯/必须无电梯” → `elevator=false`
- “无所谓电梯/有无都行” → omit `elevator`

### area (size in m²)

- “≥60平/60平以上” → `min_area=60`
- “≤40平/40平以内” → `max_area=40`
- “40-60平” → `min_area=40&max_area=60`

### subway constraints

- If user gives line: “13号线” → `subway_line=13号线`
- If user gives station: “车公庄站” → `subway_station=车公庄站`
- If user says “近地铁/步行10分钟内”:
  - Prefer distance constraint: `max_subway_dist=800` (recommended default for “近地铁”)
  - If user gave a tighter requirement, use that (e.g., “500米内” → `max_subway_dist=500`)

### utilities_type

- “民水民电” → `utilities_type=民水民电`

### available_from_before (date)

- If user provides full `YYYY-MM-DD`, pass through.
- If user gives partial date (e.g., “3月10前”), prefer asking a follow-up for year if unclear.
- Do NOT guess dates if ambiguity is high.

### commute_to_xierqi_max

- If user says “到西二旗通勤不超过 X 分钟” → `commute_to_xierqi_max=X`

### sort

- “最便宜/按价格升序” → `sort_by=price&sort_order=asc`
- “面积最大” → `sort_by=area&sort_order=desc`
- “离地铁最近” → `sort_by=subway&sort_order=asc`

### pagination

Defaults:
- `page=1`
- `page_size=10` (avoid huge pages; max is 10000 but do not use unless explicitly needed)

## Request construction

### Headers (required)

Always set:

```json
{ "X-User-ID": "d00640449" }
```

### WebRequest examples

**Basic search: Haidian, 1–2 bedrooms, <=8000, near subway**

* method: `GET`
* url: `http://7.225.29.223:8080/api/houses/by_platform?district=%E6%B5%B7%E6%B7%80&bedrooms=1,2&max_price=8000&max_subway_dist=800&page=1&page_size=10&sort_by=price&sort_order=asc`
* headers: `{ "X-User-ID": "d00640449" }`
* payload: `null`

**Community lookup**

* method: `GET`
* url: `http://7.225.29.223:8080/api/houses/by_community?community=%E5%BB%BA%E6%B8%85%E5%9B%AD(%E5%8D%97%E5%8C%BA)&page=1&page_size=10`
* headers: `{ "X-User-ID": "d00640449" }`
* payload: `null`

## Output guideline

For each returned item, extract a compact “candidate card”:

* `house_id`
* `price`
* `area_m2`
* `bedrooms`
* `rental_type`
* `district/area/community` if present
* any subway fields if present

Also return:

* applied filters (echo back what was used)
* next-step suggestion: “pick 3 to compare details” or “tighten budget/district” if too many.

## Error handling

* If API returns empty results:
  1. Do NOT silently change hard constraints.
  2. Ask a single targeted question: “预算/区域/是否必须近地铁 哪个可以放宽？”
* If user requests “对比平台挂牌”，do not do it here; hand off to a detail/listings skill.
