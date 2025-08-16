import yaml
from dotenv import load_dotenv
import json
import requests
import feedparser
import os, sys
from contextlib import contextmanager
from newspaper import Article
from llama_cpp import Llama
import openai

# -----------------------------
# Global
# -----------------------------
FAILURES = 0

# ---------------------------------
# Troubleshooting Section
# ---------------------------------
import time
start_time = time.time()
def track_step_time(step_name):
    global  start_time
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"{step_name} took {elapsed:.3f} seconds")
    start_time = end_time
    return end_time

def handle_exception(ex, additional_message=None):
    if additional_message:
        print(additional_message)
    print(f'Exception: {ex}')


# ---------------------------------
# Helper Methods
# ---------------------------------
def load_config():
    # Get environment values
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment file")

    # Load YAML config
    script_dir = os.path.dirname(os.path.abspath(__file__))  # directory of rss_agent.py
    config_path = os.path.join(script_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Inject API key into config
    config["openai_api_key"] = openai_api_key
    return config

def setup():
    prompt = None
    llm = None
    config = load_config()
    script_dir = os.path.dirname(os.path.abspath(__file__))  # directory of rss_agent.py
    prompt_path = os.path.join(script_dir, config["rss_prompt_file"])
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()

    if config["use_local_llm"]:
        @contextmanager
        def suppress_output():
            with open(os.devnull, "w") as devnull:
                old_out, old_err = sys.stdout, sys.stderr
                sys.stdout, sys.stderr = devnull, devnull
                try:
                    yield
                finally:
                    sys.stdout, sys.stderr = old_out, old_err
        with suppress_output():
            model_path = str(config["model_path"])
            local_model = str(config["local_model"])
            llm = Llama(model_path=os.path.join(model_path, local_model), n_ctx=2048, verbose=False)
        track_step_time("Load Local LLM")
    else:
        openai.api_key = config["openai_api_key"]
        track_step_time("Setup openAI")
    return config, prompt, llm

def cost_estimator(config, prompt_tokens, completion_tokens):
    if configuration["use_local_llm"]:
        return 0, 0

    input_cost = 2.50
    output_cost = 10.00
    price_factor = 1000000
    model = config['model']

    if model == "gpt-5-nano":
        input_cost = 0.05
        output_cost = 0.40
    elif model == "gpt-5-mini":
        input_cost = 0.25
        output_cost = 2.00
    elif model == "gpt-5":
        input_cost = 1.25
        output_cost = 10.00

    input_cost = round((prompt_tokens/price_factor) * input_cost,4)
    output_cost = round((completion_tokens / price_factor) * output_cost,4)
    return input_cost, output_cost


def format_ai_response(config, response, title, link, feed_url):
    summary = None
    try:
        if config['use_local_llm']:
            llm_response = response.choices[0].text.strip() or ""
            llm_response = parse_or_extract_json(llm_response)
            prompt_tokens = 0
            completion_tokens = 0
            model = config['local_model']
        else:
            llm_response = response.choices[0].message.content.strip() or ""
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            model = config['model']

        if llm_response == "":
            debug_response(response)

        raw_summary = json.loads(llm_response)
        input_cost, output_cost = cost_estimator(config, prompt_tokens, completion_tokens)
        total_cost = input_cost + output_cost
        relevancy_score = raw_summary['relevancy_score']
        urgency_score = raw_summary['urgency_score']
        overall_score = int((relevancy_score + (2 * urgency_score)) / 3)

        summary = {
            "title": title,
            "link": link,
            "summary": raw_summary['summary'],
            "reasons": raw_summary['reasons'],
            "tags": raw_summary['tags'],
            "relevancy_score": relevancy_score,
            "urgency_score": urgency_score,
            "overall_score": overall_score,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "feed_url": feed_url,
            "model": model
        }
    except Exception as ex:
        handle_exception(ex)

    return summary


def parse_or_extract_json(text):
    """
    Try to parse text as JSON. If it fails, try to extract JSON from a ```json block.
    Returns a Python dictionary if successful, otherwise None.
    """
    # First, try parsing the text directly
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass  # Not valid JSON, try extracting from ```json block

    # Try to extract JSON from ```json ... ```
    import re
    match = re.search(r"```json\s*(\{.*?})\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
        return json_str
    else:
        return ""

def article_exists(config: dict, link: str) -> bool:
    payload = {"link": link}
    try:
        response = requests.post(config['url_search'], json=payload)
        response.raise_for_status()  # raise exception for 4xx/5xx errors
        data = response.json()
        return bool(data)  # If the search returned at least one record, it exists
    except requests.RequestException as e:
        print(f"Error calling search API: {e}")
        return False

def process_summary(config, summary):
    global FAILURES
    if summary:
        # Convert tags list to comma-separated string
        if summary["tags"]:
            summary["tags"] = ",".join(summary["tags"])
        else:
            summary["tags"] = ""
        response = requests.post(config['url_create'], json=summary)

        # Check result
        if response.status_code == 200:
            print("Article inserted successfully!")
        else:
            print(f"Error {response.status_code}: {response.text}")
    else:
        FAILURES += 1
        print('There is no summary')

# -----------------------------
# AI Summarization
# -----------------------------
def summarize_text_ai(config, prompt, llm, article_text, title, link, feed_url):
    if not article_text.strip():
        return None

    if config['use_local_llm']:
        llm_prompt = f"""{prompt}
        ARTICLE TITLE: {title}
        ARTICLE LINK: {link}
        ARTICLE CONTENT:
        {article_text}
        """
        response = llm(prompt=llm_prompt, max_tokens=config['max_tokens'])
        summary = format_ai_response(config, response, title, link, feed_url)
    else:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": article_text}
        ]
        response = openai.ChatCompletion.create(
            model=config['model'],
            messages=messages,
            max_completion_tokens=config['max_tokens']
        )
        summary = format_ai_response(config, response, title, link, feed_url)
    track_step_time("AI Summarized Article")
    return summary

# -----------------------------
# Extract Full Article Text
# -----------------------------
def get_full_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        track_step_time(f"Retrieved Article: {url}")
        return article.text
    except Exception as ex:
        handle_exception(ex)
        return ""

# -----------------------------
# Fetch & Store Articles
# -----------------------------
def fetch_and_store_articles(config, prompt, llm):
    for feed_url in config['rss_feeds']:
        response = requests.get(feed_url)
        feed = feedparser.parse(response.content)

        for entry in feed.entries:
            link = None
            try:
                title = entry.title
                link = entry.link
                if not article_exists(config, link):
                    # Get full article text
                    full_text = get_full_article_text(link)

                    # If full article is empty, fallback to RSS summary
                    if full_text:
                        summary = summarize_text_ai(config, prompt, llm, full_text, title, link, feed_url)
                        process_summary(config, summary)
                else:
                    print(f'Article already exists: {link}')
            except Exception as ex:
                handle_exception(ex, f"Skipping Article due to Exception {link}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    overall_start = time.time()

    configuration, full_prompt, local_llm = setup()
    fetch_and_store_articles(configuration, full_prompt, local_llm)

    overall_end = time.time()
    print(f"RSS agent run took {overall_end - overall_start:.3f} seconds")
    print(f'There were {FAILURES} failures')