---
name: property_management
description: 执行租房、退租、下架等状态变更操作，当用户明确指定要执行操作时，才使用该技能。
allowed-tools: rent_house,terminate_rental,take_offline,get_house_by_id,get_house_listings
---

# Prompt
- 目标：按用户意图执行状态变更（租房/退租/下架）。
- 必要参数：`house_id` + `listing_platform`（链家/安居客/58同城）。缺一则追问。
- 意图映射：
  - 租房 → `rent_house`
  - 退租/解除租约 → `terminate_rental`
  - 下架/停止出租 → `take_offline`
- 若用户表达明确认可某套房（如“就这个”）且 house_id 明确，可直接执行租房。
- 操作后仅回复结果与关键字段（house_id、listing_platform、状态）。

# Tool mapping
- 状态变更：`rent_house` / `terminate_rental` / `take_offline`
- 操作前校验：`get_house_by_id` / `get_house_listings`
