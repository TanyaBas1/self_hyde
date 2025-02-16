import time
import openai

class Generator:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key

    
    def generate(self):
        return ""


class OpenAIGenerator(Generator):
    def __init__(self, model_name, api_key, base_url=None, n=1, max_tokens=512, 
                 temperature=0.7, top_p=1, frequency_penalty=0.0, 
                 presence_penalty=0.0, stop=['\n\n\n'], wait_till_success=False):
        super().__init__(model_name, api_key)
        self.n = n
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.wait_till_success = wait_till_success
        self.base_url = base_url
        self._client_init()
    
    def _client_init(self):
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def parse_response(self, response):
        """Parse the response from the OpenAI API."""
        texts = []
        for choice in response.choices:
            texts.append(choice.message.content)
        return texts
    
    def generate(self, prompt):
        get_results = False
        while not get_results:
            try:
                result = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    top_p=self.top_p,
                    n=self.n,
                    stop=self.stop
                )
                get_results = True
            except Exception as e:
                if self.wait_till_success:
                    time.sleep(1)
                else:
                    raise e
        return self.parse_response(result)

