from typing import Any, Awaitable, Callable, Coroutine, Optional, Union


from azure.search.documents.aio import SearchClient
from azure.storage.blob.aio import ContainerClient
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
)
from openai_messages_token_helper import build_messages, get_token_limit

from approaches.approach import ThoughtStep
from approaches.chatapproach import ChatApproach
from core.authentication import AuthenticationHelper
from core.imageshelper import fetch_image

class ChatReadRetrieveReadVisionApproach(ChatApproach):
    """
    A multi-step approach that first uses OpenAI to turn the user's question into a search query,
    then uses Azure AI Search to retrieve relevant documents, and then sends the conversation history,
    original user question, and search results to OpenAI to generate a response.
    """

    def __init__(
        self,
        *,
        search_client: SearchClient,
        blob_container_client: ContainerClient,
        openai_client: AsyncOpenAI,
        auth_helper: AuthenticationHelper,
        chatgpt_model: str,
        chatgpt_deployment: Optional[str],  # Not needed for non-Azure OpenAI
        gpt4v_deployment: Optional[str],  # Not needed for non-Azure OpenAI
        gpt4v_model: str,
        embedding_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
        embedding_model: str,
        embedding_dimensions: int,
        sourcepage_field: str,
        content_field: str,
        query_language: str,
        query_speller: str,
        vision_endpoint: str,
        vision_token_provider: Callable[[], Awaitable[str]]
    ):
        self.search_client = search_client
        self.blob_container_client = blob_container_client
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.chatgpt_model = chatgpt_model
        self.chatgpt_deployment = chatgpt_deployment
        self.gpt4v_deployment = gpt4v_deployment
        self.gpt4v_model = gpt4v_model
        self.embedding_deployment = embedding_deployment
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.query_language = query_language
        self.query_speller = query_speller
        self.vision_endpoint = vision_endpoint
        self.vision_token_provider = vision_token_provider
        self.chatgpt_token_limit = get_token_limit(gpt4v_model)

    @property
    def system_message_chat_conversation(self):
        return """
        You are an AI assistant specializing in agricultural issues, particularly crop diseases and farming practices. 
        Your role is to help agronomists answer queries from growers about various field-related issues, agro-chemicals, pest control, nutrients, fertilizers, etc. 
        Use the provided knowledge base of documents to answer questions. 
        The documents contain text, graphs, tables, and images.

        Your response should be structured as follows:

        1. If an image analysis is provided:
            a. Disease Identification: Briefly describe the identified disease based on the image analysis.
            b. Provide a concise explanation of the disease, its symptoms, and potential impacts.
            c. Solution Recommendation: Recommend products or treatments based strictly on the information found in the provided documents.

        2. If no image analysis is provided (text query only):
            a. Briefly restate the user's question to confirm understanding.
            b. Provide relevant information from the knowledge base to address the query.
            c. If applicable, suggest products, treatments, or best practices based strictly on the information found in the provided documents.

        Always include the source name from the image or text for each fact you use in the response in the format: [filename]
        Answer the following question using only the data provided in the sources below.
        If asking a clarifying question to the user would help, ask the question.
        Be brief in your answers.
        For tabular information return it as an html table. Do not return markdown format.
        The text and image source can be the same file name, don't use the image title when citing the image source, only use the file name as mentioned
        If you cannot answer using the sources below, say you don't know. Return just the answer without any input texts.

        {follow_up_questions_prompt}
        {injected_prompt}
        """

    async def analyze_image(self, image_url: str) -> str:
        import requests
        from io import BytesIO
        import base64
        # Step 1: Fetch the image from the URL
        response = requests.get(image_url)
        encoded_image = ""
        # Step 2: Ensure the request was successful
        if response.status_code == 200:
            # Step 3: Load the image into memory
            image_data = BytesIO(response.content)

            # Step 4: Convert image to base64
            # If you need the image in Base64 format
            encoded_image = base64.b64encode(response.content).decode('utf-8')

            #print("Base64 Encoded Image:")
            #print(encoded_image)
        else:
            print(f"Failed to retrieve image. Status code: {response.status_code}")

        """Analyze the image using Azure Computer Vision API for plant diseases."""
        deployment_name = self.gpt4v_deployment
        response = await self.openai_client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in identifying crop diseases. Provide concise, accurate identifications."},
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": "Analyze this plant image and provide:\n1. Crop name\n2. Disease name\n3. Key symptoms (in one sentence)"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                    }
                ]}
            ],
            max_tokens=100,
            temperature=0.3,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        assistant_response = response.choices[0].message.content
        print(assistant_response)
        return assistant_response


    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: bool = False,
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]]]:
        seed = overrides.get("seed", None)
        use_text_search = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        use_vector_search = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_ranker = True if overrides.get("semantic_ranker") else False
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top", 3)
        minimum_search_score = overrides.get("minimum_search_score", 0.0)
        minimum_reranker_score = overrides.get("minimum_reranker_score", 0.0)
        filter = self.build_filter(overrides, auth_claims)

        vector_fields = overrides.get("vector_fields", ["embedding"])
        send_text_to_gptvision = overrides.get("gpt4v_input") in ["textAndImages", "texts", None]
        send_images_to_gptvision = overrides.get("gpt4v_input") in ["textAndImages", "images", None]

        original_user_query = messages[-1]["content"]
        if not isinstance(original_user_query, str):
            raise ValueError("The most recent message content must be a string.")
        past_messages: list[ChatCompletionMessageParam] = messages[:-1]

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        image_received = overrides.get("image_url", "")
        image_analysis = ""
        print("Image received is ", image_received)
        if image_received:
            image_analysis = await self.analyze_image(image_received)
            #product_recommendations = await self.get_product_recommendations(image_analysis)
            #print("Product recommendations based on this are ", product_recommendations)

        # Generate search query based on image analysis
        if image_analysis:
            user_query_request = f"Generate a search query for the following plant disease analysis: {image_analysis}"
        else:
            user_query_request = original_user_query

        query_response_token_limit = 100
        query_model = self.chatgpt_model
        query_deployment = self.chatgpt_deployment
        query_messages = build_messages(
            model=query_model,
            system_prompt=self.query_prompt_template,
            few_shots=self.query_prompt_few_shots,
            past_messages=past_messages,
            new_user_content=user_query_request,
            max_tokens=self.chatgpt_token_limit - query_response_token_limit,
        )

        chat_completion: ChatCompletion = await self.openai_client.chat.completions.create(
            model=query_deployment if query_deployment else query_model,
            messages=query_messages,
            temperature=0.0,
            max_tokens=query_response_token_limit,
            n=1,
            seed=seed,
        )

        query_text = self.get_search_query(chat_completion, image_analysis)

        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query

        # If retrieval mode includes vectors, compute an embedding for the query
        vectors = []
        if use_vector_search:
            for field in vector_fields:
                vector = (
                    await self.compute_text_embedding(query_text)
                    if field == "embedding"
                    else await self.compute_image_embedding(query_text)
                )
                vectors.append(vector)

        results = await self.search(
            top,
            query_text,
            filter,
            vectors,
            use_text_search,
            use_vector_search,
            use_semantic_ranker,
            use_semantic_captions,
            minimum_search_score,
            minimum_reranker_score,
        )
        sources_content = self.get_sources_content(results, use_semantic_captions, use_image_citation=True)
        content = "\n".join(sources_content)

        # STEP 3: Generate a contextual and content specific answer using the search results and chat history

        # Allow client to replace the entire prompt, or to inject into the existing prompt using >>>
        system_message = self.get_system_prompt(
            overrides.get("prompt_template"),
            self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else "",
        )

        if image_analysis:
            user_content: list[ChatCompletionContentPartParam] = [
                {"text": f"Image Analysis:\n{image_analysis}\n\nUser Query: {original_user_query}", "type": "text"}
            ]
        user_content: list[ChatCompletionContentPartParam] = [{"text": image_analysis, "type": "text"}]
        image_list: list[ChatCompletionContentPartImageParam] = []

        if send_text_to_gptvision:
            user_content.append({"text": "\n\nSources:\n" + content, "type": "text"})
        if send_images_to_gptvision:
            for result in results:
                url = await fetch_image(self.blob_container_client, result)
                if url:
                    image_list.append({"image_url": url, "type": "image_url"})
            user_content.extend(image_list)
        if image_analysis:
            image_list.append({"image_url": image_analysis, "type": "image_url"})

        response_token_limit = 1024
        messages = build_messages(
            model=self.gpt4v_model,
            system_prompt=system_message,
            past_messages=messages[:-1],
            new_user_content=user_content,
            max_tokens=self.chatgpt_token_limit - response_token_limit,
        )

        data_points = {
            "text": sources_content,
            "images": [d["image_url"] for d in image_list],
        }

        extra_info = {
            "data_points": data_points,
            "thoughts": [
                ThoughtStep(
                    "Prompt to generate search query",
                    [str(message) for message in query_messages],
                    (
                        {"model": query_model, "deployment": query_deployment}
                        if query_deployment
                        else {"model": query_model}
                    ),
                ),
                ThoughtStep(
                    "Search using generated search query",
                    query_text,
                    {
                        "use_semantic_captions": use_semantic_captions,
                        "use_semantic_ranker": use_semantic_ranker,
                        "top": top,
                        "filter": filter,
                        "vector_fields": vector_fields,
                        "use_text_search": use_text_search,
                    },
                ),
                ThoughtStep(
                    "Search results",
                    [result.serialize_for_results() for result in results],
                ),
                ThoughtStep(
                    "Prompt to generate answer",
                    [str(message) for message in messages],
                    (
                        {"model": self.gpt4v_model, "deployment": self.gpt4v_deployment}
                        if self.gpt4v_deployment
                        else {"model": self.gpt4v_model}
                    ),
                ),
            ],
        }

        chat_coroutine = self.openai_client.chat.completions.create(
            model=self.gpt4v_deployment if self.gpt4v_deployment else self.gpt4v_model,
            messages=messages,
            temperature=overrides.get("temperature", 0.3),
            max_tokens=response_token_limit,
            n=1,
            stream=should_stream,
            seed=seed,
        )
        return (extra_info, chat_coroutine)
    
    async def get_product_recommendations(self, disease_info: str) -> str:
        # Use the disease_info to perform a targeted search in your document database
        search_results = await self.search(
            top=3,  # Adjust as needed
            query_text=disease_info,
            filter=None,  # Adjust if you need to filter by document type
            vectors=[],
            use_text_search=True,
            use_vector_search=False,  # Adjust based on your setup
            use_semantic_ranker=True,
            use_semantic_captions=False,
            minimum_search_score=0.5,
            minimum_reranker_score=0.0
        )

        # Extract relevant information from search results
        product_info = self.get_sources_content(search_results, use_semantic_captions=False, use_image_citation=True)

        # Use GPT to synthesize the product recommendations
        deployment_name = self.chatgpt_deployment
        response = await self.openai_client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are an AgTech product specialist. Provide concise, specific product recommendations based on the given information using only given data source. Don't use any extrenal data sources on web."},
                {"role": "user", "content": f"Based on the disease: {disease_info}, provide product recommendations using this information:\n\n{' '.join(product_info)}"}
            ],
            max_tokens=300,
            temperature=0.5,
        )
        
        return response.choices[0].message.content
