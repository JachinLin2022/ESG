# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
key = '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'
openai.api_key = key
openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "user", "content": "Who won the world series in 2020?"},
    ]
)