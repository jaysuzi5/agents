# Agents
This project if for various Python AI Agents

## rss_agent
AI agent that will pull articles from various RSS feeds, review them based upon my interest and rank them.  They
will then be loaded into a PostgreSQL database by calling an Articles API.  Before processing the records, the API
will be called to see if that article already exists in the database.