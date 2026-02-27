---
name: rental-house-detail-and-compare
description: "Fetch a house’s full detail and compare its listings across platforms (链家/安居客/58同城). 
Produces compact “house cards” and cross-platform status/price comparisons for decision-making. 
Triggers: user provides 房源ID(HF_xxxx) or asks 房源详情/这个房子怎么样/对比平台挂牌/同一房源在链家安居客58的差异/看挂牌记录."
allowed-tools: WebRequest
---

# Purpose

This skill is for **enrichment** after discovery:
- Get a single house’s **full detail** by `house_id`
- Get that house’s **all platform listings** and compare them (status/price/etc.)

It ALWAYS sends `X-User-ID: d00640449`.

Base URL:
- `http://7.197.86.219:8080`

Tool:
- `WebRequest(method, url, headers, payload)`

## Endpoints covered

- `GET /api/houses/{house_id}`
- `GET /api/houses/listings/{house_id}`

(Do NOT perform state-changing actions like rent/terminate/offline here.)

## Inputs expected

- `house_id` like `HF_2001`
If the user does not provide a `house_id`, ask a short question:
- “你要看哪套房？请给我房源ID（例如 HF_2001）或从候选列表里选一套。”

## Workflow

### A) House detail (single house deep dive)
1) Call `GET /api/houses/{house_id}`
2) Extract key fields into a compact “house card” (see Output).

Use when user asks:
- “这套房具体信息”
- “户型/面积/装修/朝向/楼层/电梯/地铁距离/可入住时间”等

### B) Cross-platform listings (same house across platforms)
1) Call `GET /api/houses/listings/{house_id}`
2) Summarize:
   - which platforms exist (链家/安居客/58同城)
   - each listing’s status (可租/已租/下架, etc.)
   - price differences if present
3) If user asks “哪个平台更便宜/更可靠”:
   - only compare fields returned by API; do not invent external judgments.

Use when user asks:
- “这套房在链家/安居客/58的挂牌记录”
- “平台差异/同一房源不同平台价格”
- “为什么有的平台显示下架”

### C) Compare multiple house_ids (optional batch compare)
If user provides multiple IDs (e.g., “HF_2001 和 HF_2033 对比一下”):
- Fetch each house detail (A) for 2–5 items max.
- Present a comparison table-like summary (in plain text) with:
  price, area, bedrooms, subway distance, elevator, decoration, available date if present.

If more than 5 IDs, ask user to pick top 3–5.

## Request construction

### Headers (required)
Always include:
```json
{ "X-User-ID": "d00640449" }
```

### WebRequest examples

**House detail**

* method: `GET`
* url: `http://7.197.86.219:8080/api/houses/HF_2001`
* headers: `{ "X-User-ID": "d00640449" }`
* payload: `null`

**House listings across platforms**

* method: `GET`
* url: `http://7.197.86.219:8080/api/houses/listings/HF_2001`
* headers: `{ "X-User-ID": "d00640449" }`
* payload: `null`

## Output guideline

### “House card” (standardized summary)

Produce a compact summary object (internally) and a human-readable card (externally).

Include if present:

* `house_id`
* `community`
* `district`, `area` (商圈)
* `price` (monthly)
* `area_m2`
* `bedrooms`
* `rental_type` (整租/合租)
* `decoration`
* `orientation`
* `elevator` (true/false)
* `subway_line`, `subway_station`, `subway_dist_m`
* `utilities_type`
* `available_from` or similar availability info if present
* any notable tags returned by API

Example internal JSON shape:

```json
{
  "house_id": "HF_2001",
  "community": "建清园(南区)",
  "district": "海淀",
  "price": 6500,
  "area_m2": 45,
  "bedrooms": 1,
  "rental_type": "整租",
  "subway": {"station": "西二旗站", "line": "13号线", "dist_m": 650},
  "elevator": true,
  "decoration": "精装",
  "orientation": "南北"
}
```

### Listings comparison summary

From `/api/houses/listings/{house_id}` response (data = { total, page_size, items }):

* Summarize per platform:

  * platform name
  * listing status
  * price (if present)
  * any timestamps/attributes (if present)

Example presentation:

* 安居客：可租，¥6500/月
* 链家：下架（可能是平台侧下架）
* 58同城：可租，¥6300/月

Only state what API returns.

## Error handling

* 404 / not found:

  * Ask user to confirm ID.
  * If user only has a community name, suggest using `rental-house-search` to find IDs.
* Listings endpoint returns empty items:

  * Say “该房源暂无平台挂牌记录” and still provide house detail if available.

## Guardrails

* Do not attempt to rent/terminate/offline here.
* Do not merge cross-platform records unless they share the same `house_id`.
* Do not fabricate missing fields; if not present, omit and keep summary minimal.
