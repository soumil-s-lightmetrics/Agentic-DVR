import aiohttp
import os
from utils.html_util import convert_p_to_div_with_style
from utils.http_util import fetch_with_retry

class FreshdeskAPIUtil:
    def __init__(self):
        self.api_key = os.getenv("FRESHDESK_API_KEY")
        self.base_url = os.getenv("FRESHDESK_API_BASE_URL")
        if not self.api_key or not self.base_url:
            raise ValueError("Freshdesk API key or domain is not set in environment variables.")
        self.auth = aiohttp.BasicAuth(self.api_key, 'X')

    async def get_ticket_details(self, ticket_id):
        """
        Fetch details of a Freshdesk ticket by its ID.
        Args:
            ticket_id (int): The ID of the ticket to fetch.
        Returns:
            dict: The details of the ticket.
        """
        url = f"{self.base_url}/api/v2/tickets/{ticket_id}"
        return await fetch_with_retry(url, method="GET", auth=self.auth)

    async def add_note_to_ticket(self, ticket_id, note_content):
        """
        Add a note to a Freshdesk ticket.
        Args:
            ticket_id (int): The ID of the ticket to add the note to.
            note_content (str): The content of the note to be added.
        Returns:
            dict: The response from the Freshdesk API after adding the note.
        """
        url = f"{self.base_url}/api/v2/tickets/{ticket_id}/notes"
        headers = {'Content-Type': 'application/json'}
        payload = {"body": convert_p_to_div_with_style(note_content), "private": True}
        return await fetch_with_retry(url, method="POST", payload=payload, headers=headers, auth=self.auth)
    
    async def create_reply_to_ticket(self, ticket_id, reply_content):
        """
        Create a reply to a Freshdesk ticket.
        Args:
            ticket_id (int): The ID of the ticket to reply to.
            reply_content (str): The content of the reply.
        Returns:
            dict: The response from the Freshdesk API after creating the reply.
        """
        url = f"{self.base_url}/api/v2/tickets/{ticket_id}/reply"
        headers = {'Content-Type': 'application/json'}
        payload = {"body": convert_p_to_div_with_style(reply_content)}
        return await fetch_with_retry(url, method="POST", payload=payload, headers=headers, auth=self.auth)

    async def get_article_details(self, article_id):
        """
        Fetch details of a Freshdesk knowledge base article by its ID.
        Args:
            article_id (int or str): The ID of the article to fetch.
        Returns:
            dict: The details of the article including title, description, and description_text.
        """
        url = f"{self.base_url}/api/v2/solutions/articles/{article_id}"
        return await fetch_with_retry(url, method="GET", auth=self.auth)