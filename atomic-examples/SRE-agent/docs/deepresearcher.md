In the previous post, I have introduced you to the Atomic Agents Framework, a multi-agent framework inspired by Atomic Design principles. If you didn’t do so yet, you can read the article “Introducing Atomic Agents: A New Framework for Building Agentic AI with Modular Design”.

We have seen how Atomic Agents bring modularity, predictability, and control to AI agent development, much like how LEGO blocks allow you to build complex structures from simple, reusable pieces.

Today, we’re diving deeper into the practical applications of Atomic Agents. Specifically, I will be demonstrating how to create a Deep Research Multi-Agent System using this framework. This system will showcase the power of Atomic Agents in handling complex research tasks by breaking them down into smaller, manageable components with the smallest possible failure rate.

Before we get started, if you are looking for an experienced freelance architect / developer / lead or you are looking to develop a Proof of Concept or even if you simply would like strategic advice on implementing AI, I am your guy!

Get in touch with me on LinkedIn or even feel free to reach out to me by email at kenny.vaneetvelde@gmail.com for any projects you would like to have me involved in.
What We Will Build (& What We Won’t)

Imagine a scenario where you need to conduct extensive research on a particular topic. Traditionally, this would involve sifting through countless articles, extracting relevant information, and synthesizing it into a coherent report. With Atomic Agents, we can automate this entire process, ensuring consistency, reliability, and efficiency while still being able to fine-tune each step and each component to your liking.

Now, perhaps you could say: “Sounds like a good job for CrewAI” or whatever multi-agent framework you would initially have in mind, with a “Researcher” agent, an “Editor” agent, … Truth is, however that it will probably try a lot of things, fail at a lot of things, and eventually not quite give you what you were looking for, all while it costing a lot of money due to the extensive re-trying.

Instead of trying to let AI figure it out on its own, let’s look at how we would do it without AI. Without AI we would:

    Think of Queries: Based on the initial question or topic.
    Conduct Research: Use search engines or other resources to find information.
    Draft an Initial Answer: Create a preliminary answer in our heads.
    Select the Best Sources: Review the search results in more detail and pick the most relevant and reliable sources.
    Extract Key Information: Visit these URLs and extract the information you need.
    Refine your initial answer: Use the extracted information to improve and elaborate on the initial answer, making it more comprehensive and detailed.

This can be completely automated by coding it as a flow where multiple agents pass data to each other.

To aid in this process we will define a few Atomic Agents. In other words, we will look at this process and try to identify the smallest possible tasks that cannot be done more reliably with traditional code.

Step 1 will be replaced with a query generating agent, step 2 is traditionally coded and can be handled with SearxNG, Tavily, … Step 3 and 4 will each have their own specialized agent whereas for step 5 we will use a temporary in-memory vector DB, in this case we will use FAISS. Finally we will have an agent that specializes in refining an original summary or answer into a more complete answer.

The final automated process will look like this:

    Generate Queries for Research: Formulate queries based on the initial user input or research topic.
    Perform Search for Each Query: Execute searches using a search tool to gather information related to each query.
    Select Top URLs: Filter and select the most relevant URLs from the search results.
    Generate Initial Answer: Create an initial answer or summary based on the content from the top URLs.
    Ingest URLs into Vector Database: Store the content of the selected URLs into an in-memory database for further processing.
    Retrieve Relevant Chunks: Extract relevant pieces of information (chunks) from the in-memory database.
    Refine the Answer: Improve and refine the initial answer using the retrieved chunks of information.
    Display Refined Answer: Present the refined answer to the user.
    Repeat or Exit: Prompt the user for new input to continue the process or exit if no further input is provided.

Defining the Query Agent

First, let’s define the Query Agent. In the Atomic Agents Framework

import instructor
import openai
from atomic_agents.agents.base_chat_agent import BaseAgentIO, BaseChatAgent, BaseChatAgentConfig
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator, SystemPromptInfo
from atomic_agents.lib.tools.search.searx_tool import SearxNGSearchTool

class QueryAgentInputSchema(BaseAgentIO):
    instruction: str = 'A detailed instruction or request to generate deep research queries for.'
    num_queries: int = 'The number of queries to generate.'

# Create the query agent
query_agent = BaseChatAgent(
    BaseChatAgentConfig(
        client=instructor.from_openai(openai.OpenAI()), 
        model='gpt-3.5-turbo',
        system_prompt_generator=SystemPromptGenerator(
            SystemPromptInfo(
                background=[
                    "You are an intelligent query generation expert.",
                    "Your task is to generate a specified number of diverse and highly relevant queries based on a given instruction or request.",
                    "The queries should cover different aspects of the instruction to ensure comprehensive exploration."
                ],
                steps=[
                    "You will receive a detailed instruction or request and the number of queries to generate.",
                    "Generate the requested number of queries in a JSON format."
                ],
                output_instructions=[
                    "Ensure clarity and conciseness in each query.",
                    "Ensure each query is unique and as diverse as possible while remaining relevant to the instruction."
                ]
            )
        ),
        input_schema=QueryAgentInputSchema,
        output_schema=SearxNGSearchTool.input_schema
    )
)

Some important things to note here:

    We defined a custom input schema. In our demo, the instruction will come from the user and the number of queries to execute will be arbitrarily hardcoded, but it could just as well come from a web form, be determined by another agent together with the instruction, …
    The output schema is dynamically set to be the input schema of the SearxNGSearchTool, but it could be a SerperSearchTool, Tavily, … This means that if we want to switch to another search tool that happens to have a different input schema, we simply switch out the class and don’t have to refactor anything else.

Defining the URL Selector Agent

Similarly to the query agent, we will define an agent that specializes in picking the most important/promising URLs based on the initial user query, and the search results gathered by the query agent.

from typing import List
import instructor
import openai
from pydantic import BaseModel, Field, HttpUrl
from atomic_agents.agents.base_chat_agent import BaseAgentIO, BaseChatAgent, BaseChatAgentConfig
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator, SystemPromptInfo

class TopUrlsSelectorInputSchema(BaseAgentIO):
    user_input: str = Field(..., description='The user input or question.')
    num_urls: int = Field(..., description='The number of top URLs to select.')

class TopUrlsSelectorOutputSchema(BaseAgentIO):
    top_urls: List[HttpUrl] = Field(..., description='The list of top URLs selected based on the user input.')

# Create the top URLs selector agent
top_urls_selector_agent = BaseChatAgent(
    BaseChatAgentConfig(
        client=instructor.from_openai(openai.OpenAI()), 
        model='gpt-3.5-turbo',
        system_prompt_generator=SystemPromptGenerator(
            SystemPromptInfo(
                background=[
                    "You are an intelligent URL selection expert.",
                    "Your task is to select the best URLs from a list of search results that are most relevant to the user input or question."
                ],
                steps=[
                    "You will receive the user input or question, the list of search results, and the number of best URLs to select.",
                    "Analyze the search results and select the best URLs that are most relevant to the user input or question.",
                ],
                output_instructions=[
                    "Ensure each selected URL is highly relevant to the user input or question.",
                    "Ensure the selected URLs are diverse yet as relevant as possible."
                ]
            )
        ),
        input_schema=TopUrlsSelectorInputSchema,
        output_schema=TopUrlsSelectorOutputSchema
    )
)

It’s worth noting that we use Pydantic’s HttpUrl in the output schema. Once the schema is passed to the LLM, the LLM will know this has to be a string that is a valid URL, and the Python code will dynamically validate this as well so that you have full type safety.

Also note that we don’t yet pass in, or provide room for the actual search results from the previous step. We will add this context later in an easy plug-and-play fashion.
Defining the Question-Answering Agent and Info Refiner Agent

Next, we can define a special Atomic Agent that is an expert at answering questions given a context. Here again we will provide the context later in a plug-and-play way.

import instructor
import openai
from pydantic import BaseModel, Field
from atomic_agents.agents.base_chat_agent import BaseAgentIO, BaseChatAgent, BaseChatAgentConfig
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator, SystemPromptInfo

class AnswerAgentInputSchema(BaseAgentIO):
    question: str = Field(..., description='A question that needs to be answered based on the provided context.')

class AnswerAgentOutputSchema(BaseAgentIO):
    markdown_output: str = Field(..., description='The answer to the question in markdown format.')

# Create the answer agent
answer_agent = BaseChatAgent(
    BaseChatAgentConfig(
        client=instructor.from_openai(openai.OpenAI()), 
        model='gpt-3.5-turbo',
        system_prompt_generator=SystemPromptGenerator(
            SystemPromptInfo(
                background=[
                    "You are an intelligent answering expert.",
                    "Your task is to provide accurate and detailed answers to user questions based on the given context."
                ],
                steps=[
                    "You will receive a question and the context information.",
                    "Generate a detailed and accurate answer based on the context."
                ],
                output_instructions=[
                    "Ensure clarity and conciseness in each answer.",
                    "Ensure the answer is directly relevant to the question and context provided."
                ]
            )
        ),
        input_schema=AnswerAgentInputSchema,
        output_schema=AnswerAgentOutputSchema
    )
)

And finally, our refiner agent, which is almost the same, but will refine our initial answer later on based on more detailed context that we will get from the vector database. Its input schema will be both the original question and the initial answer, and its output will be the refined answer.

from pydantic import BaseModel, Field
from atomic_agents.agents.base_chat_agent import BaseAgentIO
import instructor
import openai
from atomic_agents.agents.base_chat_agent import BaseChatAgent, BaseChatAgentConfig
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator, SystemPromptInfo

class RefineAnswerInputSchema(BaseAgentIO):
    question: str = Field(..., description='The question that was asked.')
    answer: str = Field(..., description='The initial answer to the question.')

class RefineAnswerOutputSchema(BaseModel):
    refined_answer: str = Field(..., description='The refined answer to the question.')

# Create the refine answer agent
refine_answer_agent = BaseChatAgent(
    BaseChatAgentConfig(
        client=instructor.from_openai(openai.OpenAI()), 
        model='gpt-3.5-turbo',
        system_prompt_generator=SystemPromptGenerator(
            SystemPromptInfo(
                background=[
                    "You are an intelligent answer refinement expert.",
                    "Your task is to expand and elaborate on an existing answer to a question using additional context from vector DB chunks."
                ],
                steps=[
                    "You will receive a question or instruction, the initial answer, and additional context from vector DB chunks.",
                    "Expand and elaborate on the initial answer using the additional context to provide a more comprehensive and detailed response."
                ],
                output_instructions=[
                    "Ensure the refined answer is clear, concise, and well-structured.",
                    "Ensure the refined answer is directly relevant to the question and incorporates the additional context provided.",
                    "Add new information and details to make the final answer more elaborate and informative.",
                    "Do not make up any new information; only use the information present in the context."
                ]
            )
        ),
        input_schema=RefineAnswerInputSchema,
        output_schema=RefineAnswerOutputSchema
    )
)

Connecting all the dots

Now that we have created all the agents that we have to create, there is still a few more things left.

    The context providers to pass the search results & vector database results into an Agent
    The actual Vector DB implementation

Let’s start with the Vector DB implementation, I went for an in-memory FAISS implementation that re-initializes each run, but feel free to bring your own!

import openai
import faiss
import numpy as np
import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import markdownify

class InMemFaiss:
    def __init__(self, openai_api_key):
        openai.api_key = openai_api_key
        self.index = faiss.IndexFlatL2(1536)
        self.texts = []
        self.metadata = []

    async def download_content(self, url):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url)
            html_content = await page.content()
            await browser.close()
        
        # Clean HTML content
        cleaned_html = self.clean_html(html_content)
        
        # Convert cleaned HTML to Markdown
        markdown_content = markdownify.markdownify(cleaned_html, heading_style="ATX")
        return markdown_content

    def clean_html(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Example cleaning: remove script and style tags
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        
        # Additional cleaning can be done here
        cleaned_html = str(soup)
        return cleaned_html

    def split_text(self, text, chunk_size=1000, overlap=250):
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size].strip()
            chunks.append(chunk)
        return chunks

    async def generate_embeddings(self, texts):
        embeddings = []
        for text in texts:
            response = await openai.Embedding.create(input=text, model="text-embedding-3-small")
            embeddings.append(response['data'][0]['embedding'])
        return np.array(embeddings).astype('float32')

    async def ingest_urls(self, urls):
        tasks = [self.process_url(url) for url in urls]
        await asyncio.gather(*tasks)

    async def process_url(self, url):
        try:
            url = str(url)
            content = await self.download_content(url)
            chunks = self.split_text(content)
            embeddings = await self.generate_embeddings(chunks)
            self.index.add(embeddings)
            self.texts.extend(chunks)
            self.metadata.extend([{"url": url}] * len(chunks))
        except Exception as e:
            print(f"Failed to ingest URL {url}: {e}")

    async def retrieve_chunks(self, query, top_k=5):
        query_embedding = await self.generate_embeddings([query])
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.texts[idx] for idx in indices[0]]

The most important about the FAISS implementation is the methods ingest_urls and retrieve_chunks which will store data in our vector DB and retrieve them later to put them in the context of the Agent, respectively.

Now that we have all of that in place, the last thing for us to do is create what is called a “Context Provider” in Atomic Agents. We have two agents that want search results from our SearxNG server (the Question-Answering agent and the URL selector agent) and one that wants chunks from a vector DB, so we will create both.

from atomic_agents.lib.components.system_prompt_generator import SystemPromptContextProviderBase
    
class SearchResultsProvider(SystemPromptContextProviderBase):
    def __init__(self, title: str):
        super().__init__(title=title)
        self.search_results = []
        
    def get_info(self) -> str:
        return f'SEARCH RESULTS: {self.search_results}'

class VectorDBChunksProvider(SystemPromptContextProviderBase):
    def __init__(self, title: str):
        super().__init__(title=title)
        self.chunks = []

    def get_info(self) -> str:
        return f'VECTOR DB CHUNKS: {self.chunks}'

search_results_provider = SearchResultsProvider(title='Search results')
vector_db_chunks_provider = VectorDBChunksProvider(title='Vector DB chunks')

As everything in Atomic Agents, Context Providers are highly streamlined and self-consistent. They should always have a get_info() method that returns a single string of formatted context.

Here you could parse it out nicely, but I find that just stringifying them (which parses them into JSON automatically) does a great job already. But, you have full control, in case you want to parse it nicer, filter it, …

Great! Now we have everything defined, and we can finally stitch it all together in a main.py

import os
from examples.deep_research_multi_agent.query_agent import query_agent
from examples.deep_research_multi_agent.question_answering import answer_agent
from examples.deep_research_multi_agent.top_urls_selector import top_urls_selector_agent
from examples.deep_research_multi_agent.info_refiner import refine_answer_agent
from examples.deep_research_multi_agent.providers import search_results_provider, vector_db_chunks_provider
from examples.deep_research_multi_agent.in_memory_faiss import InMemFaiss
from atomic_agents.lib.tools.search.searx_tool import SearxNGSearchTool, SearxNGSearchToolConfig
from rich.console import Console
import asyncio

async def main():
    
    console = Console()

    # Initialize the search tool
    search_tool = SearxNGSearchTool(SearxNGSearchToolConfig(base_url=os.getenv('SEARXNG_BASE_URL'), max_results=30))
    in_mem_faiss = InMemFaiss(openai_api_key=os.getenv('OPENAI_API_KEY'))

    user_input = input("Enter your question (or type 'exit' to quit): ")
    
    # Register the context providers. This allows us to plug-n-play different info in our system prompt.
    answer_agent.register_context_provider('search_results', search_results_provider)
    top_urls_selector_agent.register_context_provider('search_results', search_results_provider)
    refine_answer_agent.register_context_provider('vector_db_chunks', vector_db_chunks_provider)

    while True:
        if user_input.lower() == 'exit':
            break

        console.print("[bold green]Getting queries...[/bold green]")
        queries = query_agent.run(query_agent.input_schema(instruction=user_input, num_queries=5))

        # Start loop here
        console.print("[bold green]Registering context providers...[/bold green]")

        console.print("[bold green]Getting search results...[/bold green]")
        search_results_provider.search_results = search_tool.run(search_tool.input_schema(**queries.model_dump()))

        # Select top URLs
        console.print("[bold green]Selecting top URLs...[/bold green]")
        top_urls = top_urls_selector_agent.run(top_urls_selector_agent.input_schema(
            user_input=user_input,
            num_urls=5
        ))

        # Call the answer agent
        console.print("[bold green]Calling the answer agent...[/bold green]")
        initial_answer = answer_agent.run(answer_agent.input_schema(question=user_input))
        console.print(initial_answer.markdown_output, markup=True)

        console.print("[bold green]=====================[/bold green]")

        console.print("[bold green]Ingesting URLs into in-memory FAISS...[/bold green]")
        await in_mem_faiss.ingest_urls(top_urls.top_urls)

        # Retrieve chunks
        console.print("[bold green]Retrieving chunks...[/bold green]")
        query = user_input
        top_k = 5
        chunks = in_mem_faiss.retrieve_chunks(query, top_k)

        vector_db_chunks_provider.chunks = chunks

        console.print("[bold green]Refining the answer...[/bold green]")
        refined_answer = refine_answer_agent.run(refine_answer_agent.input_schema(
            question=user_input,
            answer=initial_answer.markdown_output
        ))

        console.print("Refined Answer:")
        console.print(refined_answer.refined_answer, markup=True)

        # We don't need to keep the context for this agent after each run. This will save money.
        refine_answer_agent.reset_memory()

        user_input = input("Enter your question (or type 'exit' to quit): ")

# Run the main function
asyncio.run(main())

And that’s it! As promised earlier, we have registered all of the context providers here in a plug-n-play fashion, and we have even made a shared context provider to share data between the answering agent and the URL selector agent!
Wrapping Up

By now, you should have a good overview of how to work with Atomic Agents. For a somewhat deeper dive into what is going on behind the scenes, please refer back to my in-depth introduction.

Since Atomic Agents is completely open-source, any feedback, help, contributions, pull requests, … are very welcome!

I hope you enjoyed this article and that it inspires you to go out there and build some Agentic AI systems that are actually useful and not just made tot WOW a YouTube audience with “Generate an entire application from a one-sentence prompt WOOOOW OMG INDUSTRY SHAKING”

See you in the next one, and don’t forget to get in touch with me for business inquiries on My LinkedIn