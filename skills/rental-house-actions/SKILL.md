---
name: rental-house-actions
description: "Perform state-changing house actions: rent, terminate rental, 
    or take a listing offline for a specific platform. 
    Enforces strict parameter checks and explicit user confirmation. 
    Triggers: user says 租/我要租/确认租/退租/解除租约/下架/停止出租/把房源下架, 
    and provides or references a house_id (e.g., HF_2001)."
allowed-tools: WebRequest
---

# Purpose

This skill executes **state-changing operations** on a house:
- rent
- terminate rental (make available again)
- take offline (mark as offline)

It ALWAYS sends `X-User-ID: d00640449`.

Base URL:
- `http://7.197.86.219:8080`

Tool:
- `WebRequest(method, url, headers, payload)`

## Endpoints covered

- `POST /api/houses/{house_id}/rent` (requires query `listing_platform`)
- `POST /api/houses/{house_id}/terminate` (requires query `listing_platform`)
- `POST /api/houses/{house_id}/offline` (requires query `listing_platform`)

## Required inputs (hard gate)

You MUST have:
1) `house_id` (e.g., `HF_2001`)
2) `listing_platform` ∈ { `链家`, `安居客`, `58同城` }

If either is missing, ask a single direct question to obtain it.

## Safety rule: explicit confirmation required

Before sending any POST:
1) Summarize the exact action you will take:
   - action: `rent` / `terminate` / `offline`
   - house_id
   - listing_platform
2) Ask user to confirm by replying with an explicit confirmation string:

Confirmation format (choose one and require it verbatim):
- Chinese: `确认 <action_cn> <house_id> @ <listing_platform>`
  - examples:
    - `确认 租用 HF_2001 @ 安居客`
    - `确认 退租 HF_2001 @ 链家`
    - `确认 下架 HF_2001 @ 58同城`
- English fallback: `CONFIRM <action> <house_id> @ <listing_platform>`

Do NOT call POST unless the user replies with a matching confirmation.

## Dispatch rules (map user intent → endpoint)

- If user intent is “租 / 租下 / 我就要这套” → `rent`
  - `POST /api/houses/{house_id}/rent?listing_platform=...`

- If user intent is “退租 / 解除租约 / 恢复可租” → `terminate`
  - `POST /api/houses/{house_id}/terminate?listing_platform=...`

- If user intent is “下架 / 停止出租 / 不挂了” → `offline`
  - `POST /api/houses/{house_id}/offline?listing_platform=...`

If ambiguous between terminate vs offline:
- Ask: “你是要退租（恢复可租）还是下架（不再出租）？”

## Request construction

### Headers (required)
Always include:
```json
{ "X-User-ID": "d00640449" }
```

### Query param

`listing_platform` is required for all POST endpoints.

### Payload

No JSON body is specified; use `null`.

### WebRequest examples

**Rent on Anjuke**

* method: `POST`
* url: `http://7.197.86.219:8080/api/houses/HF_2001/rent?listing_platform=%E5%AE%89%E5%B1%85%E5%AE%A2`
* headers: `{ "X-User-ID": "d00640449" }`
* payload: `null`

**Terminate on Lianjia**

* method: `POST`
* url: `http://7.197.86.219:8080/api/houses/HF_2001/terminate?listing_platform=%E9%93%BE%E5%AE%B6`
* headers: `{ "X-User-ID": "d00640449" }`
* payload: `null`

**Take offline on 58**

* method: `POST`
* url: `http://7.197.86.219:8080/api/houses/HF_2001/offline?listing_platform=58%E5%90%8C%E5%9F%8E`
* headers: `{ "X-User-ID": "d00640449" }`
* payload: `null`

## Output guideline

After a successful POST:

* Confirm action success in one line
* Echo returned record fields (as provided by API), especially:

  * `house_id`
  * `listing_platform`
  * updated status fields (whatever API returns)
* Remind the user:

  * API updates all three platforms’ states together but returns the specified platform record.

If the API returns an error:

* Explain minimally (status code + message if available)
* Suggest next step:

  * verify house_id
  * verify listing_platform
  * if already rented/offline, suggest terminate or other appropriate action.

## Edge cases

* User asks “租这套” but no platform:

  * Ask: “你要在哪个平台操作？链家 / 安居客 / 58同城（必填）”
* User provides platform synonym:

  * Normalize:

    * “58” → `58同城`
    * “anjuke/安居” → `安居客`
    * “lianjia/链家网” → `链家`
* User tries to perform multiple actions in one message:

  * Handle one action at a time; ask which house/platform first.
