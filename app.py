from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from pydantic import BaseModel, Field, validator
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv
import tiktoken
import time
import openai
load_dotenv()
from unittest import result
from altair import CompositeMark
import gradio as gr

output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

raw_material_template= """You are an expert in Life-cycle assessment. You have been hired as a Life-cycle assessment consultant, to perform Life-cycle assesment of a product.
You have been given a product. The product name is given below:
***
# PRODUCT NAME: {product_name}
***
Analyse, Brainstorm the Identify all the Raw Materials that are involved in manufacturing this product.
{format_instructions}
"""

process_template="""You are an expert in Manufacturing and Industrial Engineering.
Given below is a list raw materials.
***
# Raw Materials: 
{raw_material}
***
Analyse and tell the manifacturing process involved in making each one of the Raw Materials. If you know the exact manufacturing process terminology then say it. If no process is involved and it is naturally occuring then just say no processing requireed.
Give the raw material alongside it's preocess that you identify on each new line, in the format "raw material" "==>" "manufacturing process"
"""

analysis_template="""You have done P.H.D in Product Lifecycle Assessment Analysis. And are now working as a sustainability consultant.
Given below is the product name. The list of raw materials required in manufacturing it and manufacturing processes involved for each raw material.
***
# Product: {product}
***
# Raw Material and Manufacturing Processes: {details}
***
Analyse the details and give top 5 recommendations on how to reduce the emissions across the supply chain for this product. Focus on newer green solutions that can be used.
Give recommendations in bullet points.
"""

material_prompt = PromptTemplate(
    template=raw_material_template,
    input_variables=["product_name"],
    partial_variables={"format_instructions": format_instructions}
)

process_prompt = PromptTemplate(
    template=process_template,
    input_variables=["raw_material"]
)

analysis_prompt = PromptTemplate(
    template=analysis_template,
    input_variables=["product","details"]
)

temperature = 0.5
llm = OpenAI(temperature=temperature,model_name="gpt-4")
material_chain = LLMChain(llm=llm, prompt=material_prompt,verbose=True)
process_chain = LLMChain(llm=llm, prompt=process_prompt,verbose=True)
analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt,verbose=True)

raw_material_data = {}

def get_materials_from_gpt(product_name):
    materials =  material_chain.predict(product_name=product_name)
    return output_parser.parse(materials)

def get_processes(raw_material):
    process = process_chain.predict(raw_material=raw_material)
    return process

def get_analysis(product, details):
    return analysis_chain.predict(product=product, details=details)
    

with gr.Blocks() as demo:
    with gr.Column(scale=3):
        gr.Markdown(
        """
        # AI LCA assistant
        """)
    with gr.Column(scale=1):
        product_name = gr.Textbox(label="Enter Product for LCA Analysis")
        # components = gr.Textbox(label="Enter individual components of the product")
        tell_me_materials = gr.Button("Identify Raw Materials")

    with gr.Column(scale=1):
        raw_materials = gr.Markdown()
    
    get_processes_btn = gr.Button("identify Processes Involved")
    processes = gr.Markdown()

    get_analysis_btn = gr.Button("Analysis and Recommendations")
    analysis = gr.Markdown()

    @tell_me_materials.click(inputs=product_name, outputs=raw_materials)
    def identify_raw_matrials(product_name):
        #GPT Call to identify source raw materials
        response = get_materials_from_gpt(product_name)
        result = "# Raw Materials\n"
        # components = components.split(",")
        # components = [c.lstrip(" ") for c in components]
        # response = ""
        # for component in components:
        #     raw_material_data[component] = []
        #     response += f"## The raw materials required for {component}:\n\n"
        #     materials = get_materials_from_gpt(product_name,component)
        #     for material in materials:
        #         response += "- "+material+"\n"
        #         raw_material_data[component].append(material)
        for i in response:
            result += "- " + i + "\n"
            raw_material_data[i] = []
        return result
    
    @get_processes_btn.click(inputs=raw_materials, outputs=processes)
    def identify_processes(raw_materials):
        result = "# Processes:\n"
        response = get_processes(raw_materials)
        # for raw_material in raw_material_data.keys():
        #     response = get_processes(raw_material)
        #     time.sleep(1)
        #     result += "- " + raw_material + " ==> " + response + "\n"
        return response

        return result
    
    @get_analysis_btn.click(inputs=[product_name,processes],outputs=analysis)
    def identify_analysis(product_name,processes):
        result = get_analysis(product_name,processes)
        return result
    

demo.launch()