## Steps to convert prepare data for non-thinking run

1. combine all jsonls 
    - script - combine_jsonl.py 
    - helper - combine_config.yaml

2. fix tools - converts openai format (stringified params) to glm supported (dicts params)
    - fix_tool_content.py

3. unroll_data - to support on our training infra
    - script - unroll_data.py
    - breaks multi-turn on user messsages
    - adds enable_thinking = True

4. context length filtering
    - step 1 - quick filter by characters - filter_by_chars.py
    - step 2 - tokeniser filtering - pass2_tokenizr_filter.py
    - step 3 - merge pass1 and pass2 - merge_results.py

5. split file in train and val 
    - script - split_jsonl.py - splits 500 val samples and all other as train samples