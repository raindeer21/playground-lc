---
name: rental-house-proximity
description: "Find houses near a landmark and answer “nearby amenities” questions (parks/shopping) 
            around a community. Triggers: user says 附近/离XX近/在XX周边/步行到XX/通勤到XX/周边有没有公园或商场/小区附近配套. 
            Uses landmark name→ID resolution when needed, then calls nearby house search."
allowed-tools: WebRequest
---

# Purpose

This skill handles **proximity-based** questions:

1) “以地标为圆心查附近房源”
- Uses: `GET /api/houses/nearby`
- Input: `landmark_id` (can be ID or name per API)
- Output: houses plus distance metrics (straight-line / walking distance / walking time)

2) “查询小区周边地标（公园/商场）”
- Uses: `GET /api/houses/nearby_landmarks`
- Input: `community` + optional `type` + optional `max_distance_m`
- Output: nearby amenities sorted by distance

This skill ALWAYS sends `X-User-ID: d00640449` for the **house** endpoints.

Base URL:
- `http://7.225.29.223:8080`

Tool:
- `WebRequest(method, url, headers, payload)`

## Endpoints covered

- House proximity:
  - `GET /api/houses/nearby`
  - `GET /api/houses/nearby_landmarks`
- Landmark resolution (helper calls; no X-User-ID required):
  - `GET /api/landmarks/name/{name}`
  - `GET /api/landmarks/search`

## Dispatch rules (what to call)

### A) Nearby houses around a landmark (most common)
Trigger phrases:
- “离XX近 / 在XX附近 / XX周边租房 / 步行到XX / 到XX通勤”
Do:
1) Obtain `landmark_id`:
   - If user gave an ID like `SS_001`, use it directly.
   - If user gave a name like “西二旗站/百度/国贸”:
     - Prefer exact: `GET /api/landmarks/name/{name}`
     - Fallback fuzzy: `GET /api/landmarks/search?q=...` (ask user to choose if multiple)
2) Call `GET /api/houses/nearby?landmark_id=...&max_distance=...&listing_platform=...&page=...&page_size=...`

Notes:
- API allows passing “地标 ID 或地标名称（支持按名称查找）”.
- Still prefer resolving to a stable ID when possible to reduce ambiguity.

### B) Nearby amenities around a community (parks/shopping)
Trigger phrases:
- “小区附近有没有公园/商场/配套/周边设施”
Do:
- Call `GET /api/houses/nearby_landmarks?community=...&type=...&max_distance_m=...`

Type mapping:
- “商场/商超/购物” → `type=shopping`
- “公园/绿地” → `type=park`
If user doesn’t specify, omit `type` and return mixed results.

## Parameter defaults (small-model friendly)

### /api/houses/nearby
- `max_distance`: default `2000` meters (API default). If user says “步行10分钟内”, prefer `max_distance=1200` (rule of thumb) ONLY if user explicitly wants walking closeness; otherwise keep default and rely on API walking time.
- `page=1`
- `page_size=10`
- `listing_platform`: if user specifies 链家/安居客/58同城, pass it; else omit (API default 安居客).

### /api/houses/nearby_landmarks
- `max_distance_m`: default `3000` meters (API default)
- `type`: only set when user specifies park/shopping clearly.

## Request construction

### Headers

For house endpoints (`/api/houses/...`), always include:
```json
{ "X-User-ID": "d00640449" }
```

For landmark helper endpoints (`/api/landmarks/...`), do not include `X-User-ID`:

```json
{}
```

### URL encoding

Always URL-encode:

* Chinese landmark names
* community names
* query parameters like district/type

## WebRequest examples

### 1) Resolve landmark by exact name (helper)

* method: `GET`
* url: `http://7.225.29.223:8080/api/landmarks/name/%E8%A5%BF%E4%BA%8C%E6%97%97%E7%AB%99`
* headers: `{}`
* payload: `null`

### 2) Nearby houses around a landmark

* method: `GET`
* url: `http://7.225.29.223:8080/api/houses/nearby?landmark_id=SS_001&max_distance=2000&page=1&page_size=10&listing_platform=%E5%AE%89%E5%B1%85%E5%AE%A2`
* headers: `{ "X-User-ID": "d00640449" }`
* payload: `null`

### 3) Nearby parks around a community

* method: `GET`
* url: `http://7.225.29.223:8080/api/houses/nearby_landmarks?community=%E5%BB%BA%E6%B8%85%E5%9B%AD(%E5%8D%97%E5%8C%BA)&type=park&max_distance_m=3000`
* headers: `{ "X-User-ID": "d00640449" }`
* payload: `null`

## Output guideline

### Nearby houses response

Return:

* resolved landmark: `landmark_id + name` (if available)
* top candidates (3–10):

  * `house_id`
  * `community`
  * `price`
  * `bedrooms`
  * `area_m2`
  * distance metrics (from API): straight-line distance / walking time if present
* If too many results:

  * ask ONE follow-up to tighten either `max_distance`, `listing_platform`, or price expectations (but do not invent price filters here; use rental-house-search for complex filtering).

### Nearby amenities response

Return:

* `community`
* requested type (park/shopping or all)
* top 5–10 landmarks with:

  * `name`
  * `type`
  * `distance_m` (if present)
* If none found within max distance:

  * increase `max_distance_m` only if user indicates flexibility; otherwise ask whether to broaden range.

## Edge cases

* Landmark name ambiguous (multiple results):

  * show top 3 with `id + name + district + category`, ask user to pick one.
* User gives only “附近” without reference:

  * ask: “你希望以哪个地标/地铁站/公司为中心？”
* User gives a community name but it might be misspelled:

  * recommend using `rental-house-search` → `by_community` to confirm canonical community name, then retry amenities query.
