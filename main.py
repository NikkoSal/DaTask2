import pandas as pd
import json
from openai import OpenAI

client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1" 
)

df = pd.read_csv('Womens Clothing E-Commerce Reviews.csv')

df = df.dropna(subset=['Review Text'])
df = df.head(60)

results = []

for index, row in df.iterrows():
    review = row['Review Text']
    
    prompt = f"Проанализируй отзыв: {review}. Верни ответ строго в формате JSON с полями: sentiment (positive/negative/neutral), category (о чем отзыв, тема)."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    raw_json = response.choices[0].message.content
    data = json.loads(raw_json)
    
    results.append(data)

analysis_df = pd.DataFrame(results)
final_df = pd.concat([df.reset_index(drop=True), analysis_df], axis=1)

final_df.to_csv('result.csv', index=False, encoding='utf-8-sig')
